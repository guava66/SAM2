# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Optional, List, Tuple, Union,Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.modeling.common import LayerNorm2d, MLPBlock
from segment_anything.modeling.image_encoder import Attention, PatchEmbed, window_partition, window_unpartition


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


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W,D, C) -> (B, C, H, W,D)
    x = x.permute(0, 4, 1, 2,3)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3,4, 1)
    if norm:
        x = norm(x)

    return x




class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        device_0,
        device_1,
        img_size: int = 1024,
        patch_size: int = 16,
        patch_depth: int=32,
        in_chans: int = 3,
        embed_dim: int = 768,
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        out_chans: int = 256,
        mlp_ratio: float = 4.0,
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
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (1,2,7,2),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            5,
            7,
            9,
        ),
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec
        self.img_size = img_size
        self.device_0 = device_0
        self.device_1 = device_1
        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

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
        

        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        cur_stage = 1
         # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Split blocks between GPUs
        self.blocks_0 = nn.ModuleList()
        self.blocks_1 = nn.ModuleList()
        split_point = depth // 2  # Split blocks equally between GPUs

        self.neck=FPN3D().to(device_0)
        

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = Block_3d(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                input_size=(img_size // patch_size, img_size // patch_size, img_size // patch_size),
                res_size=window_size if i not in global_attn_indexes else img_size // patch_size,
                shift=window_size // 2 if i % 2 == 0 else 0
            )
            
            if i < split_point:
                self.blocks_0.append(block.to(device_0))
            else:
                self.blocks_1.append(block.to(device_1))
            

            embed_dim = dim_out

        
        
        self.channel_list = ([self.blocks_0[i].dim_out if i < len(self.blocks_0) else self.blocks_1[i - len(self.blocks_0)].dim_out 
        for i in self.stage_ends[::-1]]  # reverse the list of stage ends]
        if return_interm_layers
        else [self.blocks_1[-1].dim_out]
)


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
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

        # Process first half of blocks on device_0
        outputs = []
        for i, blk in enumerate(self.blocks_0):
            # print("window.shape:",self.blocks_0[i].window_size)
            x = blk(x)
            if (i in self.stage_ends) and self.return_interm_layers:
                feats = x.permute(0, 4, 1, 2, 3)
                outputs.append(feats)

        # Move to device_1 for second half of processing
        x = x.to(self.device_1)
        
        # Process second half of blocks on device_1
        for i, blk in enumerate(self.blocks_1):
            # print("window.shape:",self.blocks_1[i].window_size)
            x = blk(x)
            if ((i + len(self.blocks_0)) in self.stage_ends) and self.return_interm_layers:
                feats = x.permute(0, 4, 1, 2, 3)
                outputs.append(feats.to(self.device_0))

        # Final output
        if not self.return_interm_layers:
            x = x.permute(0, 4, 1, 2, 3)
            outputs.append(x)
        
        outputs=self.neck(outputs)
        x=outputs[-1].to(self.device_0)
            
        
        return x,outputs
    

class Block_3d(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            dim_out:int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            q_stride: Tuple[int, int] = None,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int, int]] = None,
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
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool3d(
                kernel_size=(q_stride[0], q_stride[0], q_stride[0]), stride=(q_stride[0], q_stride[0],q_stride[0]), ceil_mode=False
            )

        self.attn = Attention_3d(
            dim,
            dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size, window_size),
            res_size=(res_size, res_size, res_size),
            q_pool=self.pool,

        )
        self.dim = dim
        self.dim_out = dim_out
        # self.shift_size = shift
        # if self.shift_size > 0:
        #     H, W, D = 32, 32, 32
        #     img_mask = torch.zeros((1, H, W, D, 1))
        #     h_slices = (slice(0, -window_size),#从索引 0 开始切割，到倒数第 window_size 个位置结束（不包括该位置）。
        #                 slice(-window_size, -self.shift_size),#从倒数第 window_size 个元素开始，切割到倒数第 shift_size 个元素。这是用来实现窗口的移动（shift）。
        #                 slice(-self.shift_size, None))#从倒数第 shift_size 个元素开始，一直到序列的结尾（None 表示直到结束）。
        #     w_slices = (slice(0, -window_size),
        #                 slice(-window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     d_slices = (slice(0, -window_size),
        #                 slice(-window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     cnt = 0
        #     for h in h_slices:
        #         for w in w_slices:
        #             for d in d_slices:
        #                 img_mask[:, h, w, d, :] = cnt
        #                 cnt += 1
        #                 #遍历 h_slices、w_slices 和 d_slices，为 img_mask 赋值，每个不同的 (h, w, d) 组合分配一个唯一的 cnt 值（从 0 递增）。这将 img_mask 划分为 3×3×3=27 个不同的区域，每个区域有不同的整数值。
        #     mask_windows = window_partition(img_mask, window_size)[0]#window_partition(img_mask, window_size) 将 img_mask 划分成 window_size 大小的小块。
        #     mask_windows = mask_windows.view(-1, window_size * window_size * window_size)#mask_windows.view(-1, window_size * window_size * window_size) 把 mask_windows 变成二维矩阵，每行是一个窗口。
        #     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0,
        #                                                                                  float(0.0))
        # else:
        #     attn_mask = None
        # self.register_buffer("attn_mask", attn_mask)


        self.norm2 = norm_layer(dim_out)
        self.mlp = MLPBlock(embedding_dim=dim_out, mlp_dim=int(dim_out * mlp_ratio), act=act_layer)

        self.window_size = window_size
        self.adapter = Adapter(input_dim=dim, mid_dim=dim // 2)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        shortcut = x
        # print("shortcut's shape:",shortcut.shape)
        x = self.norm1(x)
        window_size=self.window_size
        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)
            # print("after pool,shortcut's shape:",shortcut.shape)
        # Window partition
        if self.window_size > 0:
            H, W, D = x.shape[1], x.shape[2], x.shape[3]
            # if self.shift_size > 0:
            #     x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1,2,3))
            x, pad_hw = window_partition(x, self.window_size)
        #x = self.attn(x, mask=self.attn_mask)
        # print("before attn,x's shape:",x.shape)
        x = self.attn(x)
        # print("after attn,x.shape:",x.shape)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W,D = shortcut.shape[1:4]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_d = (window_size - D % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w,D+pad_d)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W, D))
            # print("after unpartition，x.shape:",x.shape)
        # if self.shift_size > 0:
        #     x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1,2,3))
        B,H,W,D,C=x.shape[0],x.shape[1],x.shape[2],x.shape[3],x.shape[4]
        if self.q_stride:
            x=x.reshape(1,H,W,D,B*C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        # print("最后，x.shape:",x.shape)
        return x


class Attention_3d(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            dim_out:int,
            num_heads: int = 8,
            # qkv_bias: bool = True,
            # use_rel_pos: bool = False,
            # rel_pos_zero_init: bool = True,
            # input_size: Optional[Tuple[int, int]] = None,
            q_pool: nn.Module = None,
            # res_size = None
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
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        # print(input_size)

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)#self.qkv: 用于将输入特征映射到查询（Q）、键（K）和值（V）。它的输出维度是 dim * 3，因为查询、键和值是分开的。
        #self.proj: 用于输出的线性变换，将 V 的加权和映射回原始的特征空间。

        # self.use_rel_pos = use_rel_pos
        # if self.use_rel_pos:
        #     assert (
        #             input_size is not None
        #     ), "Input size must be provided if using relative positional encoding."
        #     # initialize relative positional embeddings
        #     self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
        #     self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
        #     self.rel_pos_d = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
        #     self.lr = nn.Parameter(torch.tensor(1.))
        #     #相对位置编码是通过相对位置偏差来调整 Q 和 K 之间的关系。每个维度（height, width, depth）都有对应的相对位置编码。
        #     # rel_pos_h, rel_pos_w, rel_pos_d 分别代表高度、宽度和深度的相对位置编码。
        #     # lr 是一个学习率因子，用于调整相对位置编码的影响。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W * D, 3, self.num_heads, -1)
        # print("qkv:",qkv.shape)
        #首先，使用 self.qkv 将输入 x 映射到查询、键和值，结果的形状是 (B, H * W * D, 3, num_heads, head_dim)。
        #permute(2, 0, 3, 1, 4) 重新排列维度，使得 q, k, v 分别具有 (B, num_heads, H * W * D, head_dim) 形状。
        # q, k, v with shape (B * nHead, H * W, C)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        q, k, v = torch.unbind(qkv, 2)
        #q, k, v = qkv.reshape(3, B * self.num_heads, H * W * D, -1).unbind(0)
        # print(q.shape)

        if self.q_pool:
            q = do_pool(q.reshape(B, H, W,D ,-1), self.q_pool)
            H, W,D = q.shape[1:4]  # downsampled shape
            q = q.reshape(B, H * W*D, self.num_heads, -1)
            # print("池化了")
            # q_sub = q.transpose(1,2).reshape(B * self.num_heads, H * W * D, -1)
        # else:
            # q_sub = q.reshape(B * self.num_heads, H * W * D, -1)

        # # print("attn之前,q's shape:",q.shape)
        # attn = (q * self.scale) @ k.transpose(-2, -1)
        # # print("attn之后,q's shape:",q.shape)
        # #q_sub 是将 q 展平后的结果，形状为 (B * num_heads, H * W * D, head_dim)。
        # #attn 是计算注意力的核心公式，q * scale @ k.transpose(-2, -1)，也就是通过缩放后的 Q 和 K 进行点积。

        # if self.use_rel_pos:
        #     attn = add_decomposed_rel_pos(attn, q_sub, self.rel_pos_h, self.rel_pos_w, self.rel_pos_d, (H, W, D), (H, W, D), self.lr)
        #     attn = attn.reshape(B, self.num_heads, H * W * D, -1)
        #     #如果启用了相对位置编码，调用 add_decomposed_rel_pos 函数将相对位置信息添加到计算得到的注意力矩阵中。
        # if mask is None:
        #     attn = attn.softmax(dim=-1)
        # else:
        #     nW = mask.shape[0]
        #     attn = attn.view(B // nW, nW, self.num_heads, H*W*D, H*W*D) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, H*W*D, H*W*D)
        #     attn = attn.softmax(dim=-1)
        #     #如果没有掩码，则直接对 attn 进行 softmax 操作，得到标准的注意力权重。
        #     #如果有掩码，将掩码加到 attn 上（通常用于遮挡一些区域的计算）。
        # x = (attn @ v).view(B, self.num_heads, H, W, D, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, H, W, D, -1)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        x = x.transpose(1, 2).reshape(B, H, W, D, -1)
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



class FPN3D(nn.Module):
    def __init__(self, out_channels=256):
        super(FPN3D, self).__init__()
        
        # 输入特征图的通道数
        in_channels = [768, 1536, 3072, 6144]
        
        # 横向连接（lateral connections）- 1x1x1 卷积降维
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels[i], out_channels, 1, bias=False),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU()
            ) for i in range(4)
        ])
        
        
        
    def forward(self, inputs):
        # inputs是一个list，包含4个不同尺度的特征图
        
        # 横向连接
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # 自顶向下的路径
        features = [laterals[-1]]  # 从最高层开始
        for i in range(len(laterals)-2, -1, -1):
            # 上采样
            top_down = nn.functional.interpolate(
                features[-1],
                size=laterals[i].shape[2:],
                mode='trilinear',
                align_corners=False
            )
            # 特征融合
            features.append(laterals[i] + top_down)
            
        # 反转列表，使其从底层到顶层
        features = features[::-1]
        # 获取目标形状
        target_shape = features[0].shape[2:]

                # 确保所有特征图的形状与第一个特征图的形状一致
        features = [
            F.interpolate(feature, size=target_shape, mode='trilinear', align_corners=False)
            for feature in features
        ]
        

            
        return features
    

# device_0 = torch.device('cuda:0')
# device_1 = torch.device('cuda:1')
# image_encoder=Hiera(
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
# x = torch.rand(512, 3, 512, 512).to(device_0)
# out=image_encoder(x)
# # 打印每个张量的形状
# for i, tensor in enumerate(out):
#     tensor.to(device_0)
#     print(f"Shape of element {i}:", tensor.shape)
# fpn = FPN3D()
# fpn.to(device_0)

# # 前向传播
# outputs = fpn(out)

# # 验证输出形状
# for i, feat in enumerate(outputs):
#     print(f"Output feature {i} shape: {feat.shape}")


