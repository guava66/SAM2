import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from segment_anything.modeling.common import LayerNorm2d, MLPBlock
from segment_anything.modeling.image_encoder import Attention, PatchEmbed, window_partition, window_unpartition
from functools import partial

class Adapter(nn.Module):
    def __init__(
            self,
            input_dim,
            mid_dim
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, mid_dim)
        self.conv = nn.Conv3d(in_channels = mid_dim, out_channels = mid_dim, kernel_size=3, padding=1, groups=mid_dim)
        self.linear2 = nn.Linear(mid_dim, input_dim)

    def forward(self, features):
        out = self.linear1(features)
        out = F.relu(out)
        out = out.permute(0, 4, 1, 2, 3)
        out = self.conv(out)
        out = out.permute(0, 2, 3, 4, 1)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = features + out
        return out

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class ImageEncoderViT_3d(nn.Module):
    def __init__(
            self,
            device_0,
            device_1,
            img_size: int = 1024,
            patch_size: int = 16,
            patch_depth: int=32,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            cubic_window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            num_slice = 1,
             
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.device_0 = device_0
        self.device_1 = device_1

        # 将patch_embed放在第一个GPU上
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        ).to(device_0)
        self.num_slice = num_slice
        if self.num_slice > 1:
            self.slice_embed = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim,
                                         kernel_size=(1,1,self.num_slice), stride=(1,1,self.num_slice),
                                         groups=embed_dim).to(device_0)

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            ).to(device_0)
            self.depth_embed = nn.Parameter(
                torch.zeros(1, patch_depth, embed_dim)
            ).to(device_0)

        self.blocks_0 = nn.ModuleList()
        self.blocks_1 = nn.ModuleList()
        for i in range(depth):
            block = Block_3d(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=cubic_window_size,
                res_size=window_size if i not in global_attn_indexes else img_size // patch_size,
                shift=cubic_window_size // 2 if i % 2 == 0 else 0
            )
            if i < 6:
                self.blocks_0.append(block.to(device_0))
            else:
                self.blocks_1.append(block.to(device_1))
            # self.blocks.append(block)

        self.neck_3d = nn.ModuleList()
        for i in range(4):
            neck = nn.Sequential(
                nn.Conv3d(768, out_chans, 1, bias=False),
                nn.InstanceNorm3d(out_chans),
                nn.ReLU(),
            )
            if i < 2:  # 前两个neck放在第一个GPU
                self.neck_3d.append(neck.to(device_0))
            else:  # 后两个neck放在第二个GPU
                self.neck_3d.append(neck.to(device_1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device_0)
        with torch.no_grad():
            x = self.patch_embed(x)
        if self.num_slice > 1:
            x = self.slice_embed(x.permute(3, 1, 2, 0).unsqueeze(0))
            x = x.permute(0, 2, 3, 4, 1)
        else:
            x = x.permute(1, 2, 0, 3).unsqueeze(0)


        if self.pos_embed is not None:
            pos_embed = F.avg_pool2d(self.pos_embed.permute(0,3,1,2), kernel_size=2).permute(0,2,3,1).unsqueeze(3)
            pos_embed = pos_embed + (self.depth_embed.unsqueeze(1).unsqueeze(1))
            x = x + pos_embed

        idx = 0
        feature_list = []
        for idx, blk in enumerate(self.blocks_0):
        # for blk in self.blocks[:6]:
            x = blk(x)
            idx += 1
            if idx % 3 == 0 and idx != 12:
                feature_list.append(self.neck_3d[idx//3-1](x.permute(0, 4, 1, 2, 3)))
        # 将数据转移到第二个GPU
        x = x.to(self.device_1)

# 处理后6个blocks (在device_1上)
        for idx, blk in enumerate(self.blocks_1):
            x = blk(x)
            idx += 1
            if idx % 3 == 0 and idx != 12:
                feature_list.append(self.neck_3d[idx//3-1](x.permute(0, 4, 1, 2, 3)))

        x = self.neck_3d[-1](x.permute(0, 4, 1, 2, 3))
        return x, feature_list

class ImageEncoderViT_3d_v2(nn.Module):
    def __init__(
            self,
            device_0,
            device_1,
            img_size: int = 1024,
            patch_size: int = 16,
            patch_depth: int=32,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            cubic_window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            num_slice = 1
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.mlp_ratio = 4.0 表示 MLP 层的隐藏层维度是 embed_dim 的 4 倍。当 embed_dim = 768，且 mlp_ratio = 4.0 时：hidden_dim=3072 也就是说，MLP 层的第一层 Linear 投影的输出维度是 3072，然后再通过 GELU 激活后投影回 768。
            qkv_bias (bool): If True, add a learnable bias to query, key, value.qkv_bias 参数的作用
            qkv_bias: bool = True 主要用于 自注意力机制（Self-Attention） 中的 查询（Query）、键（Key）、值（Value） 计算，如果设置为 True，则在计算 Q, K, V 时 添加一个可学习的偏置项（bias）。
            Q=XWQ+bQ
​
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.表示相对位置编码的初始值为零，训练过程中模型将逐渐学习这些位置编码。rel_pos_zero_init=False：相对位置编码的参数会被随机初始化为非零值。模型一开始就会有一些位置信息的偏见，这样可能更早地利用相对位置信息。
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.指定哪些层使用全局注意力，这有助于在保持计算效率的同时，增强模型捕捉全局信息的能力。
        """
        super().__init__()
        self.img_size = img_size
        self.device_0 = device_0
        self.device_1 = device_1

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        ).to(device_0)
        self.num_slice = num_slice
        if self.num_slice > 1:
            self.slice_embed = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim,
                                         kernel_size=(1,1,self.num_slice), stride=(1,1,self.num_slice),
                                         groups=embed_dim).to(device_0)

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            ).to(device_0)
            self.depth_embed = nn.Parameter(
                torch.ones(1, patch_depth, embed_dim)
            ).to(device_0)
        #patch_depth 表示图像在深度方向（例如 3D 图像中的 z 轴）上划分的 patch 数量。即深度方向上有多少个 patch。
        self.depth_embed = nn.Parameter(torch.zeros(1, patch_depth, embed_dim), requires_grad=True).to(device_0)

        self.blocks_0 = nn.ModuleList()
        self.blocks_1 = nn.ModuleList()
        for i in range(depth):
            block = Block_3d(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=cubic_window_size,
                res_size=window_size if i not in global_attn_indexes else img_size // patch_size,
                shift=cubic_window_size // 2 if i % 2 == 0 else 0
            )
            if i < 6:
                self.blocks_0.append(block.to(device_0))
            else:
                self.blocks_1.append(block.to(device_1))

        self.neck_3d = nn.ModuleList()
        for i in range(4):
            neck = nn.Sequential(
                nn.Conv3d(768, out_chans, 1, bias=False),
                LayerNorm3d(out_chans),
                nn.Conv3d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm3d(out_chans),
            )
            if i < 2:  # 前两个neck放在第一个GPU
                self.neck_3d.append(neck.to(device_0))
            else:  # 后两个neck放在第二个GPU
                self.neck_3d.append(neck.to(device_1))
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device_0)
        # print(x.shape)
        #x.shape=torch.Size([512, 3, 512, 512])
        with torch.no_grad():
            x = self.patch_embed(x)
            # print(x.shape)
            #x.shape=torch.Size([512, 32, 32, 768])
        if self.num_slice > 1:
            x = self.slice_embed(x.permute(3, 1, 2, 0).unsqueeze(0))
            x = x.permute(0, 2, 3, 4, 1)
        else:
            x = x.permute(1, 2, 0, 3).unsqueeze(0)
            print(x.shape)
           #x.shape=torch.Size([1, 32, 32, 32, 768])

        if self.pos_embed is not None:
            pos_embed = F.avg_pool2d(self.pos_embed.permute(0,3,1,2), kernel_size=2).permute(0,2,3,1).unsqueeze(3)
            pos_embed = pos_embed + (self.depth_embed.unsqueeze(1).unsqueeze(1))
            x = x + pos_embed
        
       

        idx = 0
        feature_list = []
        for idx, blk in enumerate(self.blocks_0):
        # for blk in self.blocks[:6]:
            x = blk(x)
            idx += 1
            if idx % 3 == 0 and idx != 12:
                feature_list.append(self.neck_3d[idx//3-1](x.permute(0, 4, 1, 2, 3)))
        
        # 将数据转移到第二个GPU
        x = x.to(self.device_1)
        # 处理后6个blocks (在device_1上)
        for idx, blk in enumerate(self.blocks_1):
        # for blk in self.blocks[6:12]:
            x = blk(x)
            if idx % 3 == 0 and idx != 12:
                feature_list.append(self.neck_3d[idx//3+2](x.permute(0, 4, 1, 2, 3)))

        x = self.neck_3d[-1](x.permute(0, 4, 1, 2, 3))
        x.to(self.device_0)

        return x, feature_list


class Block_3d(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            res_size = None,#用于窗口注意力的残差块大小。
            shift = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_3d(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=(window_size, window_size, window_size),
            res_size=(res_size, res_size, res_size),
        )
        self.shift_size = shift
        if self.shift_size > 0:
            H, W, D = 32, 32, 32
            img_mask = torch.zeros((1, H, W, D, 1))
            h_slices = (slice(0, -window_size),#从索引 0 开始切割，到倒数第 window_size 个位置结束（不包括该位置）。
                        slice(-window_size, -self.shift_size),#从倒数第 window_size 个元素开始，切割到倒数第 shift_size 个元素。这是用来实现窗口的移动（shift）。
                        slice(-self.shift_size, None))#从倒数第 shift_size 个元素开始，一直到序列的结尾（None 表示直到结束）。
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            d_slices = (slice(0, -window_size),
                        slice(-window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    for d in d_slices:
                        img_mask[:, h, w, d, :] = cnt
                        cnt += 1
                        #遍历 h_slices、w_slices 和 d_slices，为 img_mask 赋值，每个不同的 (h, w, d) 组合分配一个唯一的 cnt 值（从 0 递增）。这将 img_mask 划分为 3×3×3=27 个不同的区域，每个区域有不同的整数值。
            mask_windows = window_partition(img_mask, window_size)[0]#window_partition(img_mask, window_size) 将 img_mask 划分成 window_size 大小的小块。
            mask_windows = mask_windows.view(-1, window_size * window_size * window_size)#mask_windows.view(-1, window_size * window_size * window_size) 把 mask_windows 变成二维矩阵，每行是一个窗口。
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0,
                                                                                         float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)


        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size
        self.adapter = Adapter(input_dim=dim, mid_dim=dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W, D = x.shape[1], x.shape[2], x.shape[3]
            if self.shift_size > 0:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1,2,3))
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x, mask=self.attn_mask)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W, D))
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1,2,3))
        
        # print("x的：",x.shape)
        # print("shortcut的shape：",shortcut.shape)
        


        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class Attention_3d(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
            res_size = None
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # print(input_size)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)#self.qkv: 用于将输入特征映射到查询（Q）、键（K）和值（V）。它的输出维度是 dim * 3，因为查询、键和值是分开的。
        #self.proj: 用于输出的线性变换，将 V 的加权和映射回原始的特征空间。

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * res_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * res_size[1] - 1, head_dim))
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * res_size[2] - 1, head_dim))
            self.lr = nn.Parameter(torch.tensor(1.))
            #相对位置编码是通过相对位置偏差来调整 Q 和 K 之间的关系。每个维度（height, width, depth）都有对应的相对位置编码。
            # rel_pos_h, rel_pos_w, rel_pos_d 分别代表高度、宽度和深度的相对位置编码。
            # lr 是一个学习率因子，用于调整相对位置编码的影响。

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, H, W, D, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W * D, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        #首先，使用 self.qkv 将输入 x 映射到查询、键和值，结果的形状是 (B, H * W * D, 3, num_heads, head_dim)。
        #permute(2, 0, 3, 1, 4) 重新排列维度，使得 q, k, v 分别具有 (B, num_heads, H * W * D, head_dim) 形状。
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        #q, k, v = qkv.reshape(3, B * self.num_heads, H * W * D, -1).unbind(0)
        q_sub = q.reshape(B * self.num_heads, H * W * D, -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        #q_sub 是将 q 展平后的结果，形状为 (B * num_heads, H * W * D, head_dim)。
        #attn 是计算注意力的核心公式，q * scale @ k.transpose(-2, -1)，也就是通过缩放后的 Q 和 K 进行点积。

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q_sub, self.rel_pos_h, self.rel_pos_w, self.rel_pos_d, (H, W, D), (H, W, D), self.lr)
            attn = attn.reshape(B, self.num_heads, H * W * D, -1)
            #如果启用了相对位置编码，调用 add_decomposed_rel_pos 函数将相对位置信息添加到计算得到的注意力矩阵中。
        if mask is None:
            attn = attn.softmax(dim=-1)
        else:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, H*W*D, H*W*D) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, H*W*D, H*W*D)
            attn = attn.softmax(dim=-1)
            #如果没有掩码，则直接对 attn 进行 softmax 操作，得到标准的注意力权重。
            #如果有掩码，将掩码加到 attn 上（通常用于遮挡一些区域的计算）。
        x = (attn @ v).view(B, self.num_heads, H, W, D, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, H, W, D, -1)
        x = self.proj(x)
        #attn @ v: 使用计算得到的注意力权重和 V 进行加权求和。
        #view() 和 permute(): 对输出进行重排，使其恢复到原始输入的形状（去掉多头维度，并将 C 维放到最后）。
        #self.proj(x): 最后通过一个线性层将输出映射回原始的特征维度。

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    这段代码实现了一个 window_partition 函数，用于将一个输入张量（通常是一个 3D 图像或特征图）分割成大小固定的非重叠窗口。如果输入的高度、宽度或深度不完全是窗口大小的倍数，则会对输入进行填充，使得它的维度可以被窗口大小整除。
    """
    B, H, W, D, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    pad_d = (window_size - D % window_size) % window_size
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
    Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, Dp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Hp, Wp, Dp)


def window_unpartition(
        windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int, int], hw: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    这段代码实现了一个 window_unpartition 函数，它的作用是将先前分割成窗口的张量恢复（即反向操作），从窗口恢复到原来的形状，并且去掉填充部分。如果输入的张量在分割时进行了填充，这个函数会移除填充后的部分，恢复到原始尺寸。
    """
    Hp, Wp, Dp = pad_hw
    H, W, D = hw
    B = windows.shape[0] // (Hp * Wp * Dp // window_size // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, Dp // window_size, window_size, window_size, window_size,
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Hp, Wp, Dp, -1)

    if Hp > H or Wp > W or Dp > D:
        x = x[:, :H, :W, :D, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).
    Returns:
        Extracted positional embeddings according to relative positions.
    
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        rel_pos_d: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
        lr,
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    add_decomposed_rel_pos 函数的作用是为给定的注意力图（attention map）添加相对位置编码（relative positional encoding）。这种方法在许多视觉任务中应用，以加强模型对空间位置的感知能力。此函数采用分解的相对位置编码，这些编码可以通过沿着不同的轴（如高度、宽度和深度）对查询和键进行编码来捕获空间关系。
    """
    q_h, q_w, q_d = q_size
    k_h, k_w, k_d = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    Rd = get_rel_pos(q_d, k_d, rel_pos_d)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, q_d, dim)
    rel_h = torch.einsum("bhwdc,hkc->bhwdk", r_q, Rh)
    rel_w = torch.einsum("bhwdc,wkc->bhwdk", r_q, Rw)
    rel_d = torch.einsum("bhwdc,dkc->bhwdk", r_q, Rd)

    attn = (
            attn.view(B, q_h, q_w, q_d, k_h, k_w, k_d) +
            lr * rel_h[:, :, :, :, :, None, None] +
            lr * rel_w[:, :, :, :, None, :, None] +
            lr * rel_d[:, :, :, :, None, None, :]
    ).view(B, q_h * q_w * q_d, k_h * k_w * k_d)

    return attn

# device_0 = torch.device('cuda:0')
# device_1 = torch.device('cuda:1')
# image_encoder=ImageEncoderViT_3d_v2(
#         device_0=device_0,
#         device_1=device_1,
#         depth=12,
#         embed_dim=768,
#         img_size=1024,
#         mlp_ratio=4,
#         norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#         num_heads=12,
#         patch_size=16,
#         qkv_bias=True,
#         use_rel_pos=True,
#         global_attn_indexes=[2, 5, 8, 11],
#         window_size=14,
#         cubic_window_size=8,
#         out_chans=256,
#         num_slice = 16)
# x = torch.rand(512, 3, 512, 512) 
# output,out=image_encoder(x)
# for i, tensor in enumerate(out):
#     print(f"Shape of element {i}:", tensor.shape)
# print(output.shape)