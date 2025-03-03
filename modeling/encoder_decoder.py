import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from modeling.Image_ecoder_hiera import ImageEncoder,FpnNeck

class SAM2_3D(nn.Module):
    def __init__(self, trunk, neck, mask_decoder, patch_size=256):
        super(SAM2_3D, self).__init__()
        
        # 初始化图像编码器
        self.img_encoder = ImageEncoder(trunk=trunk, neck=neck)
        
        # 初始化掩码解码器
        self.mask_decoder = mask_decoder

        
        # 其他参数
        self.patch_size = patch_size

    def forward(self, img):
        # 图像预处理：调整大小
        out = F.interpolate(img.float(), scale_factor=512 / self.patch_size, mode='trilinear')
        input_batch = out
        
        # 获取图像特征
        batch_features, feature_list = self.img_encoder(input_batch)
        feature_list.append(batch_features)
        
        # 添加额外特征
        img_resize = F.interpolate(
            img[:, 0].permute(0, 2, 3, 1).unsqueeze(1),
            scale_factor=64 / self.patch_size,
            mode='trilinear'
        )
        feature_list.append(img_resize)
        
        # 掩码预测
        masks = self.mask_decoder(feature_list, 2, self.patch_size // 64)
        masks = masks.permute(0, 1, 4, 2, 3)
        
        return masks