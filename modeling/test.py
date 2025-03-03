# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP

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


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, D, H, W, C) -> (B, C, D, H, W)
    x = x.permute(0, 4, 1, 2, 3)
    x = pool(x)
    # (B, C, D', H', W') -> (B, D', H', W', C)
    x = x.permute(0, 2, 3, 4, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, _ = x.shape
        # qkv with shape (B, D * H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, D * H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, D * H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, D, H, W, -1), self.q_pool)
            D, H, W = q.shape[1:4]  # downsampled shape
            q = q.reshape(B, D * H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, D*H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, D, H, W, -1)

        x = self.proj(x)

        return x

class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        reduction_factor: int = 4,  # Adapter 的降维因子
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool3d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        # Adapter for 3D data
        self.adapter = Adapter(dim_out, reduction_factor)

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, D, H, W, C
        x = self.norm1(x)
        x = self.adapter(x) + x  # Insert adapter

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition for 3D
        window_size = self.window_size
        if window_size > 0:
            D, H, W = x.shape[1:4]
            x, pad_dhw = window_partition_3d(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            D, H, W = shortcut.shape[1:4]

            pad_d = (window_size - D % window_size) % window_size
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_dhw = (D + pad_d, H + pad_h, W + pad_w)

        # Reverse window partition for 3D
        if self.window_size > 0:
            x = window_unpartition_3d(x, window_size, pad_dhw, (D, H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    Adapted for 3D inputs.
    """

    def __init__(
        self,
        device_0,
        device_1,
        img_size: int = 1024,
        patch_size: int = 16,
        patch_depth: int=32,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        in_chans: int = 3,
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int, int] = (2, 2, 2),  # downsample stride between stages for 3D
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int, int] = (14, 14, 14),  # 3D bkg size
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True,  # return feats from every stage
        reduction_factor: int = 4,  # Adapter's reduction factor
        num_slice = 1
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

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
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            ).to(device_0)
            self.depth_embed = nn.Parameter(
                torch.ones(1, patch_depth, embed_dim)
            ).to(device_0)
        #patch_depth 表示图像在深度方向（例如 3D 图像中的 z 轴）上划分的 patch 数量。即深度方向上有多少个 patch。
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding for 3D
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0], self.window_spec[0])
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                reduction_factor=reduction_factor,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, dhw: Tuple[int, int, int]) -> torch.Tensor:
        d, h, w = dhw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(d, h, w), mode="trilinear")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        # x: (B, D, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:4])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x.permute(0, 4, 1, 2, 3)  # B, C, D, H, W
                outputs.append(feats)

        return outputs