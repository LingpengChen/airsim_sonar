import cv2
import numpy as np
import os
import glob

def load_depth_from_npy(filepath):
    """
    从.npy文件加载深度图像数据
    
    参数:
        filepath: .npy文件的路径
    
    返回:
        加载的深度图像数据
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    # 加载数据
    depth_data = np.load(filepath)
    print(f"已从 {filepath} 加载深度数据，形状: {depth_data.shape}")
    
    return depth_data

def visualize_depth(depth_data, max_depth=100.0):
    """
    将深度数据可视化为彩色图像
    
    参数:
        depth_data: 深度图像数据
        max_depth: 深度最大值，用于归一化，默认100米
    
    返回:
        彩色深度图
    """
    # 归一化深度图像以便于可视化
    normalized = np.clip(depth_data, 0, max_depth) / max_depth
    normalized = (normalized * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    return depth_colormap

# 获取目录中所有的.npy文件
depth_data_dir = "depth_data"
npy_files = glob.glob(os.path.join(depth_data_dir, "*.npy"))

for current_file in npy_files:
    filename = os.path.basename(current_file)
    depth_data = load_depth_from_npy(current_file)
    depth_colormap = visualize_depth(depth_data)
    cv2.imshow("Depth Image", depth_colormap)
    cv2.waitKey(100)