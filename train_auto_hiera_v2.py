import os
import torch
import argparse
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from monai.losses import DiceLoss, DiceCELoss
from dataset.datasets import load_data_volume
from sam2.build_sam import build_sam2
from utils.util import setup_logger
from modeling.Image_ecoder_hiera import ImageEncoder,FpnNeck
from modeling.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
from modeling.Hiera_encoder_v2 import  Hiera
from modeling.encoder_decoder import SAM2_3D

def convert_2d_to_3d_weights(state_dict_2d):
    """
    将 2D 权重转换为 3D 权重。

    参数:
        state_dict_2d (dict): 原始 2D 权重字典。

    返回:
        state_dict_3d (dict): 转换后的 3D 权重字典。
    """
    state_dict_3d = {}

    for k, v in state_dict_2d.items():
        if 'pos_embed_window' in k:
            if len(v.shape) == 4:  # [1, 96, 8, 8]
                # 扩展为 3D 并重复填充
                state_dict_3d[k] = v.unsqueeze(-1).expand(-1, -1, -1, -1, 8)
            else:
                state_dict_3d[k] = v
        elif 'pos_embed' in k:
            if len(v.shape) == 4 and v.shape[1] == 144 and v.shape[2] == 7 and v.shape[3] == 7:
                # 插值调整为 [1, 96, 64, 64]
                new_weight = torch.nn.functional.interpolate(
                    v,
                    size=(64, 64),
                    mode='bilinear',
                    align_corners=False
                )
                # 调整维度顺序为 [1, 64, 64, 96]
                new_weight = new_weight.permute(0, 2, 3, 1)
                state_dict_3d[k] = new_weight
            else:
                state_dict_3d[k] = v


        elif '.conv.weight' in k and len(v.shape) == 4:
            # 增加一个深度维度
            state_dict_3d[k] = v.unsqueeze(-1)

        elif 'patch_embed.proj.weight' in k:
            if v.shape == torch.Size([96, 3, 7, 7]):
                # 插值调整为 [96, 3, 16, 16]
                new_weight = torch.nn.functional.interpolate(
                    v.permute(1, 0, 2, 3).unsqueeze(0),  # [1, 3, 96, 7, 7]
                    size=(96, 16, 16),
                    mode='trilinear'
                ).squeeze(0).permute(1, 0, 2, 3)  # 返回到 [96, 3, 16, 16]
                state_dict_3d[k] = new_weight

        else:
            # 对于其他参数，直接复制
            state_dict_3d[k] = v

    return state_dict_3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="brain", type=str, choices=["kits", "pancreas", "lits", "colon", "brain"]
    )
    parser.add_argument(
        "--snapshot_path",
        default="/data/pyhData/3DSAM-adapter-main/3DSAM-adapter/work_dir",
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
        default=None,
        type=str,
    )
    parser.add_argument(
        "--seg_files",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=[256,256,256],
        nargs='+', type=list,
    )
    parser.add_argument(
        "--data_name",
        default="DDP_brain_ped_2021",
        type=str,
    )
    parser.add_argument(
        "--device_ids",
        default="0,1",
        type=str,
        help="GPU device ids, comma separated, e.g., '0,1'"
    )
    parser.add_argument(
        "--resume1", 
        type=str,
        default=None,
        help="Resuming training from checkpoint"
    )
    parser.add_argument("-bs", "--batch_size", default=4, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=6e-5, type=float)
    parser.add_argument("--max_epoch", default=200, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("--tolerance", default=5, type=int)
    parser.add_argument("--local_rank", default=-1, type=int, help="For distributed training")
    
    args = parser.parse_args()
    
    # 初始化分布式训练环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    # 根据设备设置
    device = torch.device(f'cuda:{args.local_rank}')
    
    # 设置随机裁剪大小
    if args.rand_crop_size == 0:
        if args.data in ["pancreas", "lits", "colon", "kits", "brain"]:
            args.rand_crop_size = (256, 256, 256)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
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
    
    # 创建快照路径并设置日志
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    args.snapshot_path = os.path.join(args.snapshot_path, args.data_name + run_id)
    if args.rank == 0:
        if not os.path.exists(args.snapshot_path):
            os.makedirs(args.snapshot_path)
        setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
        logger = logging.getLogger(f"train")
        logger.info(str(args))
    else:
        logger = logging.getLogger(f"train")
    
    # 加载训练数据
    train_sampler = DistributedSampler(dataset=train_data)
    
    train_data = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_worker,
        pin_memory=True
    )
    
    # 构建模型
    sam = build_sam2("sam2_hiera_l.yaml", "/data/pyhData/3DSAM-adapter-main/3DSAM-adapter/ckpt/sam2_hiera_large.pt")
    
    trunk = Hiera(
        embed_dim=144,
        num_heads=2,
        stages=[2, 6, 36, 4],
        global_att_blocks=[23, 33, 43],
        window_spec= [8, 4, 16, 8]
    )
    
    neck = FpnNeck(
        d_model=256,
        backbone_channel_list=[1152, 576, 288, 144],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest"
    )
    
    img_encoder = ImageEncoder(trunk=trunk, neck=neck)
    
    # 使用转换后的权重加载
    converted_weights = convert_2d_to_3d_weights(sam.image_encoder.state_dict())
    img_encoder.load_state_dict(converted_weights, strict=False)
    
    # 释放SAM模型内存
    del sam
    
    mask_decoder = VIT_MLAHead(img_size=96, num_classes=2)
    
    # 初始化模型
    model = SAM2_3D(
        trunk=trunk, 
        neck=neck, 
        mask_decoder=mask_decoder, 
        patch_size=args.rand_crop_size[0]
    )
    
    # 将模型组件移至GPU
    model = model.to(device)
    
    # 设置参数可训练状态
    for p in model.img_encoder.parameters():
        p.requires_grad = False
    
    # 启用特定组件的可训练状态
    for p in model.img_encoder.trunk.slice_embed.parameters():
        p.requires_grad = True
    
    for i in model.img_encoder.trunk.blocks_0:  
        for p in i.norm1.parameters():
            p.requires_grad = True
        for p in i.adapter.parameters():
            p.requires_grad = True
        for p in i.norm2.parameters():
            p.requires_grad = True
            
    for i in model.img_encoder.trunk.blocks_1:  
        for p in i.norm1.parameters():
            p.requires_grad = True
        for p in i.adapter.parameters():
            p.requires_grad = True
        for p in i.norm2.parameters():
            p.requires_grad = True
            
    # 启用neck模块所有参数的训练
    for module in model.img_encoder.neck.modules():
        for p in module.parameters():
            p.requires_grad = True
    
    # # 修正原代码中的缩进问题
    # for i in img_encoder.trunk.blocks_0:
    #     for p in i.norm1.parameters():
    #         p.requires_grad = True
    #     for p in i.adapter.parameters():
    #         p.requires_grad = True
    #     for p in i.norm2.parameters():
    #         p.requires_grad = True
    
    # 应用DDP包装
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    # 优化器设置
    params_encoder = [p for p in model.module.img_encoder.parameters() if p.requires_grad]
    encoder_opt = AdamW(
        params_encoder,
        lr=args.lr, 
        weight_decay=0
    )
    
    decoder_opt = AdamW([i for i in model.module.mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    
    # 学习率调度
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
    
    start_epoch = 0
    
    # 恢复训练状态
    if args.resume1 is not None:
        if args.rank == 0:
            print("=> loading checkpoint '{}'".format(args.resume1))
        checkpoint = torch.load(args.resume1, map_location=device)
        start_epoch = checkpoint["epoch"]
        model.module.img_encoder.load_state_dict(checkpoint['encoder_dict'])
        model.module.mask_decoder.load_state_dict(checkpoint['decoder_dict'])
        encoder_opt.load_state_dict(checkpoint['encoder_opt'])
        decoder_opt.load_state_dict(checkpoint['decoder_opt'])
        
        # 加载checkpoint后修改学习率
        for param_group in encoder_opt.param_groups:
            param_group['lr'] = 1e-5
        for param_group in decoder_opt.param_groups:
            param_group['lr'] = 5e-5
            
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
    
    # 损失函数设置
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    
    best_loss = np.inf
    patch_size = args.rand_crop_size[0]
    losses = []
    
    # 训练循环
    for epoch_num in range(start_epoch, args.max_epoch):
        train_sampler.set_epoch(epoch_num)  # 设置epoch，确保数据打乱
        loss_summary = []
        model.train()
        
        for idx, (img, seg, spacing) in enumerate(tqdm(train_data, disable=args.rank != 0)):
            img = img.to(device)
            seg = seg.to(device)
            
            if args.rank == 0:
                print('seg: ', seg.sum())
            
            # 前向传播
            masks = model(img)
            seg = seg.unsqueeze(1)
            loss = loss_cal(masks, seg)
            dice_loss1 = dice_loss(masks, seg)
            dice = 1 - dice_loss1
            
            # 修正空的to()调用
            # loss = loss.to()  # 原有错误
            
            loss_summary.append(loss.detach().cpu().numpy())
            
            # 反向传播
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            loss.backward()
            
            if args.rank == 0:
                logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) +
                    ": loss:" + str(torch.mean(torch.tensor(loss_summary[-1].flatten())).item()) +
                    ", dice:" + str(torch.mean(torch.tensor(dice)).item())
                )
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.module.img_encoder.parameters(), 12.0)
            torch.nn.utils.clip_grad_norm_(model.module.mask_decoder.parameters(), 12.0)
            
            # 优化步骤
            encoder_opt.step()
            decoder_opt.step()
            
        # 学习率调度步骤
        encoder_scheduler.step()
        decoder_scheduler.step()
        
        # 计算平均损失
        # 在所有进程上同步损失
        if len(loss_summary) > 0:
            avg_loss = np.mean(loss_summary).item()
            avg_loss_tensor = torch.tensor(avg_loss).to(device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / args.world_size
            losses.append(avg_loss)
            
            if args.rank == 0:
                logger.info("- Train metrics: " + str(avg_loss))
                
                # 检查是否为最佳模型
                is_best = False
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    is_best = True
                    
                # 保存检查点
                checkpoint = {
                    "epoch": epoch_num + 1,
                    "best_val_loss": best_loss,
                    "encoder_dict": model.module.img_encoder.state_dict(),
                    "decoder_dict": model.module.mask_decoder.state_dict(),
                    "encoder_opt": encoder_opt.state_dict(),
                    "decoder_opt": decoder_opt.state_dict()
                }
                
                torch.save(checkpoint, os.path.join(args.snapshot_path, "last.pth"))
                if is_best:
                    torch.save(checkpoint, os.path.join(args.snapshot_path, "best.pth"))

if __name__ == "__main__":
    # 确保相关库已导入

    
    # 启动函数
    main()