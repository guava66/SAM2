import pickle
import os, sys
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from .base_dataset import BaseVolumeDataset


# class KiTSVolumeDataset(BaseVolumeDataset):
#     def _set_dataset_stat(self):
#         # org load shape: d, h, w
#         self.intensity_range = (-54, 247)
#         self.target_spacing = (1, 1, 1)
#         self.global_mean = 59.53867
#         self.global_std = 55.457336
#         self.spatial_index = [0, 1, 2]  # index used to convert to DHW
#         self.do_dummy_2D = False
#         self.target_class = 2


# class LiTSVolumeDataset(BaseVolumeDataset):
#     def _set_dataset_stat(self):
#         # org load shape: d, h, w
#         self.intensity_range = (-48, 163)
#         self.target_spacing = (1, 1, 1)
#         self.global_mean = 60.057533
#         self.global_std = 40.198017
#         self.spatial_index = [2, 1, 0]  # index used to convert to DHW
#         self.do_dummy_2D = False
#         self.target_class = 2


# class PancreasVolumeDataset(BaseVolumeDataset):
#     def _set_dataset_stat(self):
#         # org load shape: d, h, w
#         self.intensity_range = (-39, 204)
#         self.target_spacing = (1, 1, 1)
#         self.global_mean = 68.45214
#         self.global_std = 63.422806
#         self.spatial_index = [2, 1, 0]  # index used to convert to DHW
#         self.do_dummy_2D = True
#         self.target_class = 2


class BrainVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (0, 225)
        self.global_mean = 44.9
        self.global_std = 22.6
        self.do_dummy_2D = True
        self.target_class = 1
        self.spatial_index = [2,1,0]

DATASET_DICT = {
    # "kits": KiTSVolumeDataset,
    # "lits": LiTSVolumeDataset,
    # "pancreas": PancreasVolumeDataset,
    # "colon": ColonVolumeDataset,
    "brain":BrainVolumeDataset,
}

def load_data_volume(
    *,
    data,
    img_path,
    seg_path,
    batch_size,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop = True,
    do_nnunet_intensity_aug=False,
    num_worker=4,
):
    if not img_path:
        raise ValueError("unspecified data directory")

    img_files = [os.path.join(img_path, f) for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    seg_files = [os.path.join(img_path, f) for f in os.listdir(seg_path) if os.path.isfile(os.path.join(seg_path, f))]

    # 排序
    img_files.sort()
    seg_files.sort()

    dataset = DATASET_DICT[data](
        img_files,
        seg_files,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_val_crop=do_val_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
        )
    return loader