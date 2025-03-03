from dataset.datasets import load_data_volume
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
import sys
from monai.losses import DiceCELoss, DiceLoss
from modeling.image_encoder import  ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
from modeling.Hiera_encoder import  Hiera 
import torch.nn.functional as F
from modeling.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
import torch.nn as nn
from functools import partial
import os
from utils.util import setup_logger
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from sam2.build_sam import build_sam2
from modeling.Image_ecoder_hiera import ImageEncoder,FpnNeck

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
        "--img_files",
        default="",

        type=str,
    )
    parser.add_argument(
        "--seg_files",
        default="",

        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=256,
        nargs='+', type=int,
    )
    parser.add_argument(
        "--data_name",
        default="",
        type=str,
    )
    parser.add_argument(
        "--device_ids",
        default="0,1",  # 默认使用前两张GPU
        type=str,
        help="GPU device ids, comma separated, e.g., '0,1'"
    )
    parser.add_argument(
    "--resume1", 
    type=str,
    default=None,
    help="Resuming training from checkpoint"
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=6e-5, type=float)
    parser.add_argument("--max_epoch", default=200, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("-tolerance", default=5, type=int)

    args = parser.parse_args()
    device_ids = [int(id) for id in args.device_ids.split(',')]
    device_0 = torch.device(f'cuda:{device_ids[0]}')
    device_1 = torch.device(f'cuda:{device_ids[0]}')
    if args.rand_crop_size == 0:
        if args.data in ["pancreas", "lits", "colon", "kits","brain"]:
            args.rand_crop_size = (256, 256, 256)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    args.snapshot_path = os.path.join(args.snapshot_path,args.data_name+run_id)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))
    # train_data = load_data_volume(
    #     data=args.data,
    #     img_path=args.img_files,
    #     seg_path=args.seg_files,
    #     batch_size=1,
    #     # train_ratio=1,
    #     augmentation=True,
    #     split="train",
    #     rand_crop_spatial_size=args.rand_crop_size,
    #     num_worker = args.num_worker
    # )
    train_data = load_data_volume(
    data=args.data,
    path_prefix=args.data_prefix,
    batch_size=2,
    train_ratio=0.5,
    augmentation=True,
    split="train",
    rand_crop_spatial_size=args.rand_crop_size,
    num_worker = args.num_worker
)
    

    # val_data = load_data_volume(
    #     data=args.data,
    #     path_prefix=args.data_prefix,
    #     batch_size=1,
    #     train_ratio=1,
    #     augmentation=False,
    #     split="val",
    #     deterministic=True,
    #     rand_crop_spatial_size=args.rand_crop_size,
    #     num_worker = args.num_worker
    # )
    # sam = sam_model_registry["vit_b"](checkpoint="/data/pyhData/3DSAM-adapter-main/3DSAM-adapter/ckpt/sam_vit_b_01ec64.pth")
    sam=build_sam2("sam2_hiera_t.yaml","/data/pyhData/3DSAM-adapter-main/3DSAM-adapter/ckpt/sam2_hiera_tiny.pt")

    # mask_generator = SamAutomaticMaskGenerator(sam)
    # img_encoder=Hiera(
    #         device_0=device_0,
    #         device_1=device_1,
    #         embed_dim=768,
    #         img_size=1024,
    #         mlp_ratio=4,
    #         norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    #         num_heads=12,
    #         patch_size=16,
    #         qkv_bias=True,
    #         use_rel_pos=False,
    #         global_attn_indexes=[2, 5, 8, 11],
    #         window_size=14,
    #         cubic_window_size=8,
    #         out_chans=256,
    #         num_slice = 16)
    trunk = Hiera(
    embed_dim=96,
    num_heads=1,
    device_0=device_0,
    device_1=device_1,
    stages=[1, 2, 7, 2],
    global_att_blocks=[5, 7, 9],

    )

    neck = FpnNeck(
        device_1=device_1,
        d_model=256,
        backbone_channel_list=[768, 384, 192, 96],
        #backbone_channel_list=[6144, 3072, 1536, 768],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest"
    )
    img_encoder = ImageEncoder(trunk=trunk, neck=neck)

    # 使用转换后的权重加载
    converted_weights = convert_2d_to_3d_weights(sam.image_encoder.state_dict())
    img_encoder.load_state_dict(converted_weights, strict=False)


    # img_encoder.load_state_dict(sam.image_encoder.state_dict(), strict=False)
    del sam
    mask_decoder = VIT_MLAHead(device_0=device_0,img_size=96, num_classes=2)

    #     # 初始化 mask_decoder 的权重
    # def init_weights(m):
    #     if isinstance(m, nn.Conv3d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.InstanceNorm3d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)

    # mask_decoder.apply(init_weights)
    mask_decoder.to(device_0)


    

    for p in img_encoder.parameters():
        p.requires_grad = False
    # img_encoder.depth_embed.requires_grad = True
    for p in img_encoder.trunk.slice_embed.parameters():
        p.requires_grad = True
    # 分别处理两个GPU上的blocks
    for i in img_encoder.trunk.blocks_0:  # GPU 0上的blocks
        for p in i.norm1.parameters():
            p.requires_grad = True
        for p in i.adapter.parameters():
            p.requires_grad = True
        for p in i.norm2.parameters():
            p.requires_grad = True
    for module in img_encoder.neck.modules():
        for p in module.parameters():
            p.requires_grad = True
        # i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)

    for i in img_encoder.trunk.blocks_1:  # GPU 1上的blocks
        for p in i.norm1.parameters():
            p.requires_grad = True
        for p in i.adapter.parameters():
            p.requires_grad = True
        for p in i.norm2.parameters():
            p.requires_grad = True
    #    i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)


    

    # 获取 device_0 和 device_1 上的参数
    params_device_0 = [p for p in img_encoder.parameters() if p.device == device_0 and p.requires_grad]
    params_device_1 = [p for p in img_encoder.parameters() if p.device == device_1 and p.requires_grad]
        # 将两个设备的参数一起传给优化器
    encoder_opt = AdamW(
        params_device_0 + params_device_1,  # 合并两个设备的参数
        lr=args.lr, 
        weight_decay=0
    )


    #encoder_opt = AdamW([i for i in img_encoder.parameters() if i.requires_grad==True], lr=args.lr, weight_decay=0)
        # 重新初始化scheduler

    #encoder_scheduler = torch.optim.lr_scheduler.LinearLR(encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    decoder_opt = AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    #decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    # 创建 LinearLR 学习率调度器，使学习率在训练过程中逐渐从 start_factor（1.0）线性减少到 end_factor（0.01），并且在 500 个迭代步骤中完成

    encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    encoder_opt,
    T_max=500,
    eta_min=1e-8
    )
    decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        decoder_opt,
        T_max=500,
        eta_min=1e-8
    )
    start_epoch=0

    # 恢复训练状态
    if args.resume1 is not None:
        ## Map model to be loaded to specified single GPU
        print("=> loading checkpoint '{}'".format(args.resume1))
        checkpoint = torch.load(args.resume1, map_location="cpu")
        start_epoch = checkpoint["epoch"]
        img_encoder.load_state_dict(checkpoint['encoder_dict'])  # 恢复模型权重
        mask_decoder.load_state_dict(checkpoint['decoder_dict'])  # 恢复模型权重
        mask_decoder.to(device_0)
        encoder_opt.load_state_dict(checkpoint['encoder_opt'])  # 恢复优化器状态
        decoder_opt.load_state_dict(checkpoint['decoder_opt'])  # 恢复优化器状态
                # 加载checkpoint后修改学习率
        for param_group in encoder_opt.param_groups:
            param_group['lr'] = 1e-5  # 新的学习率
        for param_group in decoder_opt.param_groups:
            param_group['lr'] = 5e-5  # 新的学习率

        # 重新初始化scheduler
        encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            encoder_opt,
            T_max=500,
            eta_min=1e-7
        )
        decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            decoder_opt,
            T_max=500,
            eta_min=1e-7
        )


    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    # dice_loss: 使用 Dice 损失计算，常用于图像分割任务。include_background=False 表示不包括背景类，softmax=True 表示使用 softmax 函数，to_onehot_y=True 将标签转换为 one-hot 编码。
    # loss_cal: 定义了一个综合的损失函数，结合了 Dice 损失和交叉熵损失（lambda_dice=0.5 和 lambda_ce=0.5 用于加权）。
    best_loss = np.inf
    patch_size = args.rand_crop_size[0]#patch_size = args.rand_crop_size[0]：随机裁剪大小。
    losses=[]
    for epoch_num in range(start_epoch,args.max_epoch):
        loss_summary = []
        img_encoder.train()
        mask_decoder.train()
        for idx, (img, seg, spacing) in enumerate(tqdm(train_data)):
            print('seg: ', seg.sum())#seg.sum() 将返回所有标记为 1 的像素数量。
            #img.shape=torch.Size([1, 3, 256, 256, 256])
            out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
            #             将输入的 img 图像转换为浮动类型（float32）。
            # 根据给定的 scale_factor 进行图像的插值缩放。插值操作会计算出图像的新尺寸。
            # 使用 trilinear 插值方法来对图像进行三维插值（适用于 3D 图像）
            # input_batch = (out.cuda() - pixel_mean) / pixel_std
            #out.shape=torch.Size([1, 3, 512, 512, 512])
            input_batch = out.to(device_0)
            # input_batch = input_batch[0].transpose(0, 1)
            #torch.Size([512, 3, 512, 512])
            #  transpose(0, 1) 是 PyTorch 张量的一个操作，用来交换张量的第一个和第二个维度（即交换通道和高度/宽度的维度）。
            #  假设 input_batch[0] 的形状为 (channels, height, width)，那么 input_batch[0].transpose(0, 1) 会将其转换为 (height, channels, width)。也就是说，通道和高度的维度被交换了。
            batch_features, feature_list = img_encoder(input_batch)
            feature_list.append(batch_features)
            new_feature = feature_list
            img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device_0), scale_factor=64/patch_size,
                mode='trilinear')
            
            img[:, 0]
            # img 是一个包含多张图像的批次（batch）。通常 img 的形状是 (batch_size, channels, height, width)，其中：

            # batch_size 是批次大小。
            # channels 是图像的通道数（例如，RGB图像的通道数是 3，灰度图像的通道数是 1）。
            # height 和 width 是图像的高和宽。
            # img[:, 0] 是对 img 进行切片操作，选取批次中的每张图像的第一个通道（通常是图像的第一个模态或者通道，可能是灰度图像的通道）。

            # 这会将 img 的形状从 (batch_size, channels, height, width) 转换为 (batch_size, height, width)，即丢弃了通道维度，保留了每张图像的第一个通道。
            # 2. .permute(0, 2, 3, 1)
            # permute 是 PyTorch 中的一个函数，用于重新排列张量的维度。
            # img[:, 0].permute(0, 2, 3, 1) 将切片后的张量的维度顺序从 (batch_size, height, width) 改为 (batch_size, width, height)。
            # 0 对应的是批次维度 batch_size，保持不变。
            # 2 和 3 分别对应的是 height 和 width，交换了它们的位置。
            # 1 是原来的通道维度，这里因为只选择了第一个通道，所以维度会变成 (batch_size, width, height, 1)。
            # 3. .unsqueeze(1)
            # unsqueeze(1) 会在维度 1 的位置（即 width 和 height 之间）插入一个新的维度。
            # 假设 img[:, 0].permute(0, 2, 3, 1) 的形状为 (batch_size, width, height, 1)，unsqueeze(1) 会将其转换为 (batch_size, 1, width, height)，即为图像添加回一个单通道的维度。
            new_feature.append(img_resize)
            mask_decoder = mask_decoder.to(device_0)
            masks = mask_decoder(new_feature, 2, patch_size//64)
            masks = masks.permute(0, 1, 4, 2, 3)
            seg = seg.to(device_0)
            seg = seg.unsqueeze(1)
            loss = loss_cal(masks, seg)
            dice_loss1=dice_loss(masks, seg)
            dice=1-dice_loss1
            dice.detach().cpu().numpy()
            loss = loss.to(device_0)
            loss_summary.append(loss.detach().cpu().numpy())
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            loss.backward()
            # logger.info(
            #     'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) + ": loss:" + str(
            #         loss_summary[-1].flatten().item())+",dice:"+str(torch.mean(dice.item())))
            logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) +
                ": loss:" + str(torch.mean(torch.tensor(loss_summary[-1].flatten())).item()) +
                ", dice:" + str(torch.mean(torch.tensor(dice)).item())
            )
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 12.0)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 12.0)
            #             clip_grad_norm_：这是 PyTorch 中的一个函数，用来对模型参数的梯度进行裁剪。它会计算梯度的 L2 范数，如果梯度的 L2 范数超过了指定的阈值，则会按比例缩放梯度，使其 L2 范数等于指定的阈值，从而防止梯度爆炸。

            # img_encoder.parameters()：表示对 img_encoder 模型的所有可训练参数进行梯度裁剪。img_encoder 是你网络中的一个子模块（通常是模型的一部分），parameters() 获取该模型的所有参数。
            # 12.0：表示裁剪的阈值。具体来说，如果梯度的 L2 范数超过 12.0，函数会缩放梯度，使得它的 L2 范数为 12.0。这样可以避免梯度过大而导致的训练不稳定或梯度爆炸问题。
            # 这行代码的作用是对 img_encoder 模型的梯度进行裁剪，确保它们的 L2 范数不会超过 12.0。
            encoder_opt.step()
            decoder_opt.step()
        encoder_scheduler.step()
        decoder_scheduler.step()
        losses.append((np.mean(loss_summary)).item())

        logger.info("- Train metrics: " + str(np.mean(loss_summary)))

        img_encoder.eval()
        mask_decoder.eval()
        # with torch.no_grad():
        #     loss_summary = []
        #     for idx, (img, seg, spacing) in enumerate(val_data):
        #         print('seg: ', seg.sum())
        #         out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
        #         input_batch = out.to(device_0)
        #         input_batch = input_batch[0].transpose(0, 1)
        #         batch_features, feature_list = img_encoder(input_batch)
        #         feature_list.append(batch_features)
        #         new_feature = feature_list
        #         img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device_0), scale_factor=64/patch_size,
        #                                    mode='trilinear')
        #         new_feature.append(img_resize)
        #         masks = mask_decoder(new_feature, 2, patch_size//64)
        #         masks = masks.permute(0, 1, 4, 2, 3)
        #         seg = seg.to(device_0)
        #         seg = seg.unsqueeze(1)
        #         loss = dice_loss(masks, seg)
        #         loss_summary.append(loss.detach().cpu().numpy())
        #         logger.info(
        #             'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(val_data)) + ": loss:" + str(
        #                 loss_summary[-1].flatten()[0]))
        # logger.info("- Val metrics: " + str(np.mean(loss_summary)))


        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            is_best = True
        # save_checkpoint({"epoch": epoch_num + 1,
        #                 "best_val_loss": best_loss,
        #                  "encoder_dict": img_encoder.state_dict(),
        #                  "decoder_dict": mask_decoder.state_dict(),
        #                  "encoder_opt": encoder_opt.state_dict(),
        #                  "decoder_opt": decoder_opt.state_dict()
        #                  },
        #                 is_best=is_best,
        #                 checkpoint=args.snapshot_path)
        checkpoint={"epoch": epoch_num + 1,
                        "best_val_loss": best_loss,
                         "encoder_dict": img_encoder.state_dict(),
                         "decoder_dict": mask_decoder.state_dict(),
                         "encoder_opt": encoder_opt.state_dict(),
                         "decoder_opt": decoder_opt.state_dict()
                         }
        torch.save(checkpoint,os.path.join(args.snapshot_path,"last.pth"))
        if is_best:
            torch.save(checkpoint,os.path.join(args.snapshot_path,"best.pth"))

        # logger.info("- Val metrics best: " + str(best_loss))
# %% plot loss
    plt.plot(losses)
    plt.title("Dice Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.snapshot_path,  "train_loss.png"))
    plt.close()
def convert_2d_to_3d_weights(state_dict_2d):
    state_dict_3d = {}
    
    for k, v in state_dict_2d.items():
        if 'pos_embed_window' in k:
            # 处理位置嵌入
            if len(v.shape) == 4:  # [1, 96, 8, 8]
                state_dict_3d[k] = v.unsqueeze(-1).expand(-1, -1, -1, -1, 8)
        elif '.conv.weight' in k and len(v.shape) == 4:
            # 处理卷积权重
            state_dict_3d[k] = v.unsqueeze(-1)
        elif 'patch_embed.proj.weight' in k:
            # 特殊处理patch embed投影
            # 此处可能需要更复杂的插值或重采样
            # 简单示例:
            if v.shape == torch.Size([96, 3, 7, 7]):
                new_weight = torch.nn.functional.interpolate(
                    v.permute(1, 0, 2, 3).unsqueeze(0),  # [1, 3, 96, 7, 7]
                    size=(96, 16, 16),
                    mode='trilinear'
                ).squeeze(0).permute(1, 0, 2, 3)  # 返回到[96, 3, 16, 16]
                state_dict_3d[k] = new_weight
        else:
            # 对于其他参数，直接复制
            state_dict_3d[k] = v
    
    return state_dict_3d


if __name__ == "__main__":
    main()


     # python /data/pyhData/3DSAM-adapter-main/3DSAM-adapter/train_auto.py --data brain --snapshot_path "./work_dir/ped0213" --data_prefix "/data/pyhData/MedSAM-MedSAM2/data/npz_train/BraTS-PED"  --rand_crop_size 256 

