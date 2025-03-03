import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections.abc import Mapping
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandFlipd,
    RandAffined,
    RandZoomd,
    RandRotated,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    MapTransform,
    RandScaleIntensityd,
    RandSpatialCropd,
)


class BinarizeLabeld(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            threshold: float = 0.5,
            allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.as_tensor(d[key])

            dtype = d[key].dtype
            d[key] = (d[key] > self.threshold).to(dtype)
        return d
    
class NpzBaseVolumeDataset(Dataset):
    def __init__(
            self,
            npz_paths,
            augmentation,
            split="train",
            rand_crop_spatial_size=(96, 96, 96),
            convert_to_sam=True,
            do_test_crop=True,
            do_val_crop=True,
            do_nnunet_intensity_aug=True,
            

    ):
        super().__init__()
        if isinstance(npz_paths, tuple):
            self.image_paths = list(npz_paths[0])  # 如果是元组且包含一个列表
        else:
            self.image_paths = npz_paths
        # print(f"Type of image_paths: {type(self.image_paths)}")
        # print(f"Length of image_paths: {len(self.image_paths)}")
        # print(f"First element of image_paths: {self.image_paths[0]}")
        self.label_meta=npz_paths,    # 假设每个npz包含image和label
        self.aug = augmentation
        self.split = split
        self.rand_crop_spatial_size = rand_crop_spatial_size
        self.convert_to_sam = convert_to_sam
        self.do_test_crop = do_test_crop
        self.do_nnunet_intensity_aug = do_nnunet_intensity_aug
        self.do_val_crop = do_val_crop
        self.intensity_range = (
            self.target_spacing
        ) = (
            self.global_mean
        ) = self.global_std = self.spatial_index = self.do_dummy_2D = self.target_class = None

        self._set_dataset_stat()
        self.transforms = self.get_transforms()

    def _set_dataset_stat(self):
        pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        npz_path = self.image_paths[idx]  # 直接获取npz路径
        # print(f"Loading npz file from path: {npz_path}")  # 打印 npz_path
        # print(f"Type of npz_path: {type(npz_path)}")  # 打印 npz_path 的类型


        # 加载npz文件
        with np.load(npz_path) as npz_data:
            img = npz_data['imgs'].astype(np.float32)  # 假设包含image和label
            seg = npz_data['gts'].astype(np.float32)
            # img_spacing = npz_data.get('spacing', [1.0, 1.0, 1.0])  # 获取spacing信息
            # 获取 spacing 信息
            img_spacing = npz_data["spacing"]  # 这里是通过 SimpleITK 获取的 spacing 信息，假设在 .npz 中存储了 "spacing"

        # 调整维度顺序（根据实际存储格式）
        if self.spatial_index is not None:
            img = img.transpose(self.spatial_index)
            seg = seg.transpose(self.spatial_index)
        
        # 如果图像或标签中有 NaN 值，填充为零
        img[np.isnan(img)] = 0
        seg[np.isnan(seg)] = 0

        seg = (seg == self.target_class).astype(np.float32)
        lower_bound, upper_bound = np.percentile(
                img[img > 0], 0.5
            ), np.percentile(img[img > 0], 99.5)
        intensity_range=[lower_bound, upper_bound]

        # 进行空间大小的调整（与原来一致）
        if (np.max(img_spacing) / np.min(img_spacing) > 8) or (np.max(self.target_spacing / np.min(self.target_spacing) > 8)):
            # Resize 2D
            img_tensor = F.interpolate(
                input=torch.tensor(img[:, None, :, :]),
                scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                mode="bilinear",
            )

            if self.split != "test":
                seg_tensor = F.interpolate(
                    input=torch.tensor(seg[:, None, :, :]),
                    scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                    mode="bilinear",
                )

            img = (
                F.interpolate(
                    input=img_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                    scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                    mode="trilinear",
                )
                .squeeze(0)
                .numpy()
            )

            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=seg_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                        scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )
        else:
            img = (
                F.interpolate(
                    input=torch.tensor(img[None, None, :, :, :]),
                    scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(3)]),
                    mode="trilinear",
                )
                .squeeze(0)
                .numpy()
            )
            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=torch.tensor(seg[None, None, :, :, :]),
                        scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(3)]),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )

        # 应用数据增强
        if (self.aug and self.split == "train") or ((self.do_val_crop  and self.split=='val')):
            trans_dict = self.transforms({"image": img, "label": seg})[0]
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        else:
            trans_dict = self.transforms({"image": img, "label": seg})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        seg_aug = seg_aug.squeeze()

        # 重复图像通道
        img_aug = img_aug.repeat(3, 1, 1, 1)

        return img_aug, seg_aug, img_spacing  # 返回 img_spacing

    def get_transforms(self):
        transforms = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.intensity_range[0],
                a_max=self.intensity_range[1],
                b_min=self.intensity_range[0],
                b_max=self.intensity_range[1],
                clip=True,
            ),
        ]

        # 如果是训练数据，则进行数据增强
        if self.split == "train":
            transforms.extend(
                [
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=20,#RandShiftIntensityd：随机对图像亮度进行平移（偏移 20）。
                        prob=0.5,#50% 概率进行此变换。
                        
                    ),
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                        select_fn=lambda x: x > self.intensity_range[0],#CropForegroundd：裁剪前景（去除周围无效区域）。
                                                                        # source_key="image"：基于图像本身裁剪。
                                                                        # select_fn=lambda x: x > self.intensity_range[0]：裁剪掉低于 intensity_range[0] 的部分。
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                ]
            )
            if self.do_dummy_2D:
                transforms.extend(
                    [
                       RandRotated(
                            keys=["image", "label"],
                            prob=0.3,
                            range_x=30 / 180 * np.pi,
                            keep_size=False,
                                ),
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.3,
                            min_zoom=[1, 0.9, 0.9],
                            max_zoom=[1, 1.1, 1.1],
                            mode=["trilinear", "trilinear"],
                        ),
                        #RandRotated：随机 绕 X 轴 旋转 30°（30 / 180 * π）。
                        #RandZoomd：随机缩放 0.9x ~ 1.1x。
                    ]
                )
            else:
                transforms.extend(
                    [
                        # RandRotated(
                        #     keys=["image", "label"],
                        #     prob=0.3,
                        #     range_x=30 / 180 * np.pi,
                        #     range_y=30 / 180 * np.pi,
                        #     range_z=30 / 180 * np.pi,
                        #     keep_size=False,
                        # ),
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.8,
                            min_zoom=0.85,
                            max_zoom=1.25,
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )

            transforms.extend(
                [
                    BinarizeLabeld(keys=["label"]),#BinarizeLabeld：将 label 二值化（0 或 1）。
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                    ),#patialPadd：填充 图像，使其尺寸略大（1.2x）
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                        label_key="label",
                        pos=2,
                        neg=1,
                        num_samples=1,
                    ),#随机裁剪（基于正负样本）。
                        #pos=2，neg=1 → 裁剪窗口更偏向正样本区域。
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.rand_crop_spatial_size,
                        random_size=False,#固定尺寸，不会出现不同大小的 ROI
                    ),#随机裁剪固定大小 roi_size。
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                    # RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
                    # RandShiftIntensityd(
                    #     keys=["image"],
                    #     offsets=0.10,
                    #     prob=0.2,
                    # ),
                    # RandGaussianNoised(keys=["image"], prob=0.1),
                    # RandGaussianSmoothd(
                    #     keys=["image"],
                    #     prob=0.2,
                    #     sigma_x=(0.5, 1),
                    #     sigma_y=(0.5, 1),
                    #     sigma_z=(0.5, 1),
                    # ),
                    # AddChanneld(keys=["image", "label"]),
                    # RandShiftIntensityd(keys=["image"], offsets=10),
                    # RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),]
                ]
            )
        elif (not self.do_val_crop) and (self.split == "val"):
            transforms.extend(
                [
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif  (self.do_val_crop)  and (self.split == "val"):
            transforms.extend(
                [
                    # CropForegroundd(
                    #     keys=["image", "label"],
                    #     source_key="image",
                    #     select_fn=lambda x: x > self.intensity_range[0],
                    # ),
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[i for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=self.rand_crop_spatial_size,
                        label_key="label",
                        pos=1,
                        neg=0,
                        num_samples=1,
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif self.split == "test":
            transforms.extend(
                [
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        else:
            raise NotImplementedError
        


        transforms = Compose(transforms)
        return transforms
