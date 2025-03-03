import pickle
import os, sys
from torch.utils.data import DataLoader, Dataset,random_split
import torch
import numpy as np

import torch.nn.functional as F
# from .base_dataset import BaseVolumeDataset
sys.path.append('/data/pyhData/3DSAM-adapter-main/3DSAM-adapter/dataset')
from base_npz_dataset import NpzBaseVolumeDataset



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


# class ColonVolumeDataset(BaseVolumeDataset):
#     def _set_dataset_stat(self):
#         # org load shape: d, h, w
#         self.intensity_range = (-57, 175)
#         self.target_spacing = (1, 1, 1)
#         self.global_mean = 65.175035
#         self.global_std = 32.651197
#         self.spatial_index = [2, 1, 0]  # index used to convert to DHW
#         self.do_dummy_2D = True
#         self.target_class = 1

class BrainVolumeDataset(NpzBaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (0, 225)
        self.global_mean = 44.9
        self.global_std = 22.6
        self.target_spacing = (1, 1, 1)
        self.do_dummy_2D = True
        self.target_class = 1
        self.spatial_index = [0,1,2]

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
    path_prefix,
    batch_size,
    train_ratio=1,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(256,256,256),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop = True,
    do_nnunet_intensity_aug=False,
    num_worker=4,
):
    if not path_prefix:
        raise ValueError("unspecified data directory")

    # img_files = [os.path.join(path_prefix, d[i][0].strip("/")) for i in list(d.keys())]
    # seg_files = [os.path.join(path_prefix, d[i][1].strip("/")) for i in list(d.keys())]
    npz_paths = []
    for apath in path_prefix:
        npz_paths.extend([os.path.join(apath, f) for f in os.listdir(apath) if os.path.isfile(os.path.join(apath, f))])
    print(len(npz_paths))
    # 根据 train_ratio 划分训练集和测试集
    total_size = len(npz_paths)
    print(total_size)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    # 随机划分训练集和验证集/测试集
    train_paths, val_paths = random_split(npz_paths, [train_size, val_size])
    train_paths = [npz_paths[i] for i in train_paths.indices]
    val_paths = [npz_paths[i] for i in val_paths.indices]

    # 根据 split 参数决定使用训练集还是测试集
    if split == "train":
        selected_paths = train_paths
    elif split == "val":
        selected_paths = val_paths
    elif split=="test":
        selected_paths =train_paths

    else:
        raise ValueError("Invalid split! Choose either 'train' or 'val'or'test'.")

    # 这里假设 DATASET_DICT[data] 是已经定义好的数据集类
    dataset = DATASET_DICT[data](
        npz_paths=selected_paths,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_val_crop=do_val_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
    )

    # 根据 deterministic 参数决定 DataLoader 的配置
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
        )
    
    #return loader
    return dataset

    
