import numpy as np
import os

def calculate_global_mean_std_and_intensity_range(npz_paths):
    all_image_data = []
    global_min = float('inf')  # 初始化全局最小值
    global_max = float('-inf')  # 初始化全局最大值

    # 遍历所有 npz 文件
    for npz_path in npz_paths:
        data = np.load(npz_path)
        image_data = data['imgs']  # 假设每个 npz 文件包含一个 'image' 键
        all_image_data.append(image_data.flatten())
        
        # 计算当前图像数据的最小值和最大值
        current_min = np.min(image_data)
        current_max = np.max(image_data)
        
        # 更新全局最小值和最大值
        if current_min < global_min:
            global_min = current_min
        if current_max > global_max:
            global_max = current_max

    # 将所有图像数据合并到一个大数组中
    all_image_data = np.concatenate(all_image_data, axis=0)  # 按样本维度合并

    # 计算全局均值和全局标准差
    global_mean = np.mean(all_image_data)  # 对 (N, C, H, W) 中的 (N, H, W) 求均值
    global_std = np.std(all_image_data)  # 对 (N, C, H, W) 中的 (N, H, W) 求标准差

    return global_mean, global_std, global_min, global_max

# 示例：指定多个 npz 文件的路径
npz_dir = '/data/pyhData/MedSAM-MedSAM2/data/npz_train/BraTS-PED'  # 替换为你的npz文件目录
npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]

# 计算全局均值、标准差和强度范围
global_mean, global_std, global_min, global_max = calculate_global_mean_std_and_intensity_range(npz_files)
print("Global mean:", global_mean)
print("Global std:", global_std)
print("Intensity range: min =", global_min, ", max =", global_max)
