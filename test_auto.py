from dataset.datasets import load_data_volume
import argparse
import numpy as np
import logging
from monai.losses import DiceCELoss, DiceLoss
from modeling.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
from functools import partial
import os
from utils.util import setup_logger
import surface_distance
from surface_distance import metrics
from monai.inferers import sliding_window_inference
import SimpleITK as sitk
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon","brain"]
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        nargs='+',
        type=str,
    )
    parser.add_argument(
        "--data_name",
        default="",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
    )
    parser.add_argument(
        "--device_ids",
        default="0,1",  # 默认使用前两张GPU
        type=str,
        help="GPU device ids, comma separated, e.g., '0,1'"
    )

    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument(
        "--checkpoint",
        default="best",
        type=str,
    )
    parser.add_argument("-tolerance", default=5, type=int)
    args = parser.parse_args()
    device_ids = [int(id) for id in args.device_ids.split(',')]
    device_0 = torch.device(f'cuda:{device_ids[0]}')
    device_1 = torch.device(f'cuda:{device_ids[1]}')
    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"
    # device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["colon", "pancreas", "lits", "kits","brain"]:
            args.rand_crop_size = (256, 256, 256)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    # args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    pred_save_dir=os.path.join(args.snapshot_path, "result",args.data_name)
    os.makedirs(pred_save_dir, exist_ok=True)
    

    setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))
    results=[]
    test_data = load_data_volume(
        data=args.data,
        batch_size=1,
        path_prefix=args.data_prefix,
        augmentation=False,
        split="test",
        rand_crop_spatial_size=args.rand_crop_size,
        convert_to_sam=False,
        do_test_crop=False,
        deterministic=True,
        num_worker=0
    )
    img_encoder = ImageEncoderViT_3d(
        device_0=device_0,
        device_1=device_1,
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice = 16)
    img_encoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file),map_location='cpu')["encoder_dict"], strict=True)
    # img_encoder.to(device)


    mask_decoder = VIT_MLAHead(device_0=device_0,img_size = 96)
    mask_decoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["decoder_dict"],
                          strict=True)
    mask_decoder.to(device_0)

    dice_loss = DiceLoss(include_background=False, softmax=False, to_onehot_y=True, reduction="none")
    def iou_3d(pred, target, threshold=0.5, epsilon=1e-6):
        # """
        # 计算 3D IoU。
        
        # :param pred: 预测结果，3D 张量（MetaTensor 或 Tensor）
        # :param target: 真实标签，3D 张量（Tensor）
        # :param threshold: 用于将预测概率转换为二进制的阈值，默认为 0.5
        # :param epsilon: 防止除零的一个小常数
        # :return: 3D IoU
        # """
        # 如果是 MetaTensor 类型，将其转换为 torch.Tensor
        if isinstance(pred, torch.Tensor) is False:
            pred = pred.tensor  # 获取 MetaTensor 的实际 tensor 数据

        # 先对预测结果应用 Sigmoid，如果是概率值，则二值化
        pred = torch.sigmoid(pred)  # 如果是多分类问题，通常需要先应用sigmoid或softmax
        pred = (pred > threshold).float()  # 将预测值二值化
        
        target = target.float()  # 确保目标是 float 类型

        # 计算交集和并集
        intersection = torch.sum(pred * target)  # 交集
        union = torch.sum(pred) + torch.sum(target) - intersection  # 并集

        # 计算 IoU
        iou = (intersection + epsilon) / (union + epsilon)  # 加上 epsilon 防止除零错误

        return iou
    img_encoder.eval()
    mask_decoder.eval()

    patch_size = args.rand_crop_size[0]

    def model_predict(img, img_encoder, mask_decoder):
        out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
        input_batch = out[0].transpose(0, 1)
        input_batch = input_batch.to(device_0)
        batch_features, feature_list = img_encoder(input_batch)
        feature_list.append(batch_features)

        new_feature = feature_list
        img_resize = F.interpolate(img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device_0), scale_factor=64/patch_size,
                                   mode="trilinear")
        new_feature.append(img_resize)
        masks = mask_decoder(new_feature, 2, patch_size//64)
        masks = masks.permute(0, 1, 4, 2, 3)
        masks = torch.softmax(masks, dim=1)
        masks = masks[:, 1:]
        return masks

    with torch.no_grad():
        loss_summary = []
        loss_nsd = []
        for idx, (img, seg, spacing) in enumerate(test_data):
            seg = seg.float()
            seg = seg.to(device_0)
            img = img.to(device_0)
            pred = sliding_window_inference(img, [256, 256, 256], overlap=0.5, sw_batch_size=1,
                                            mode="gaussian",
                                            predictor=partial(model_predict,
                                                              img_encoder=img_encoder,
                                                              mask_decoder=mask_decoder))
            pred = F.interpolate(pred, size=seg.shape[1:], mode="trilinear")
            seg = seg.unsqueeze(0)
            if torch.max(pred) < 0.5 and torch.max(seg) == 0:
                # loss_summary.append(1)
                # loss_nsd.append(1)
                masks=pred
                loss = 1 - dice_loss(masks, seg)
                iou=iou_3d(masks,seg)
                loss_summary.append(loss.detach().cpu().numpy())
                ssd = surface_distance.compute_surface_distances((seg == 1)[0, 0].cpu().numpy(),
                                                                 (masks==1)[0, 0].cpu().numpy(),
                                                                 spacing_mm=spacing[0].numpy())
                nsd = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)  # kits
                loss_nsd.append(nsd)
            else:
                masks = pred > 0.5
                loss = 1 - dice_loss(masks, seg)
                iou=iou_3d(masks,seg)
                loss_summary.append(loss.detach().cpu().numpy())
                ssd = surface_distance.compute_surface_distances((seg == 1)[0, 0].cpu().numpy(),
                                                                 (masks==1)[0, 0].cpu().numpy(),
                                                                 spacing_mm=spacing[0].numpy())
                nsd = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)  # kits
                loss_nsd.append(nsd)
            logger.info(
                " Case {} - Dice {:.6f} | NSD {:.6f}".format(
                    test_data.dataset.image_paths[idx], loss.item(), nsd
                ))
            results.append({
            "File": test_data.dataset.image_paths[idx],
            "Dice": loss.item(),
            "NSD":nsd,
            "IoU":iou.item()
            })
            csv_file_path = os.path.join(pred_save_dir, "loss_results.csv")
            with open(csv_file_path, mode="w", newline="") as csv_file:
                fieldnames = ["File", "Dice", "NSD", "IoU"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
            
            name=os.path.basename(test_data.dataset.image_paths[idx])
                        # 1. 将 img (MetaTensor) 转换为 NumPy 数组并保存
            target_size = img.shape[2:] 
            #img_resized = F.interpolate(img, size=target_size, mode="trilinear", align_corners=False)
            img_np = img.cpu().numpy()  # 将 MetaTensor 转换为 NumPy 数组
            
            img_sitk = sitk.GetImageFromArray(img_np)  # 将 NumPy 数组转换为 SimpleITK 图像
            Spacing=img_sitk.GetSpacing()
            #img_sitk.SetSpacing(Spacing)  # 设置空间分辨率
            sitk.WriteImage(img_sitk, os.path.join(pred_save_dir, name.replace('.npz', '.nii.gz')))

            # 2. 将 masks (MetaTensor) 转换为 NumPy 数组并保存
            masks=F.interpolate(masks.float(), size=target_size, mode="trilinear", align_corners=False)
            masks_np = masks.cpu().numpy()  # 将 MetaTensor 转换为 NumPy 数组
            masks_np = masks_np.astype(np.uint8)
            masks_sitk = sitk.GetImageFromArray(masks_np)  # 将 NumPy 数组转换为 SimpleITK 图像
            masks_sitk.SetSpacing(Spacing)  # 设置空间分辨率
            sitk.WriteImage(masks_sitk, os.path.join(pred_save_dir, name.replace('.npz', '_masks.nii.gz')))

            # 3. 将 seg (Tensor) 转换为 NumPy 数组并保存
            seg=F.interpolate(seg.float(), size=target_size, mode="trilinear", align_corners=False)
            seg_np = seg.cpu().numpy()  # 将 Tensor 转换为 NumPy 数组
            seg_np = seg_np.astype(np.uint8)
            seg_sitk = sitk.GetImageFromArray(seg_np)  # 将 NumPy 数组转换为 SimpleITK 图像
            seg_sitk.SetSpacing(Spacing)  # 设置空间分辨率
            sitk.WriteImage(seg_sitk, os.path.join(pred_save_dir, name.replace('.npz', '_gt.nii.gz')))
            
            
        logging.info("- Test metrics Dice: " + str(np.mean(loss_summary)))
        logging.info("- Test metrics NSD: " + str(np.mean(loss_nsd)))


if __name__ == "__main__":
    main()

