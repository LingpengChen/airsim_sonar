import cv2
import numpy as np
import os, re
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

intrisic = np.array([[554.25624847,   0.,         320.,        ],
                    [  0.,         554.25624847, 240.        ],
                    [  0.,           0.,           1.        ]])

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


def generate_sonar_view(depth_data, fx=554.25, cx=320, max_depth=10.0, gamma=0.7):
    """
    从透视深度图生成声纳视图，保持原始图像的宽度，应用余弦校正
    
    参数:
        depth_data: 深度图像数据 (H, W) 的numpy数组，值代表沿视线的距离
        fx: 相机内参中的水平焦距
        cx: 相机内参中的水平主点坐标，如果为None则使用图像中心
        max_depth: 最大深度值截断，默认为10米
        gamma: 伽马校正系数
        
    返回:
        声纳视图图像
    """
    height, half_width = 640, 320

    
    # 创建声纳图（初始为全0），保持原始宽度
    sonar_image = np.zeros((height, 2*half_width), dtype=np.float32)

    v_indices, u_indices = np.where((depth_data > 0) & (depth_data <= max_depth))
    depths = depth_data[v_indices, u_indices]
    
    # 计算归一化的方向向量的x分量 tan(alpha )
    dx = (u_indices - cx) / fx
    
    # 计算视线方向与Z轴夹角的余弦值
    cos_alpha = 1.0 / np.sqrt(dx**2 + 1.0)
    sin_alpha = dx / np.sqrt(dx**2 + 1.0)
    
    
    u_sonar = half_width + (height * depths * sin_alpha / max_depth).astype(int)
    v_sonar = (height  * (1 -  (depths * cos_alpha) / max_depth) ).astype(int)
    
    # 在声纳图像中标记点（累积密度）
    # 使用深度值作为权重，更远的点权重更小
    weights = 1.0 / (1.0 + depths)  # 简单的距离衰减公式
    
    for u, v, w in zip(u_sonar, v_sonar, weights):
        sonar_image[v][u] += w
        
    sonar_image = np.clip(sonar_image, 0, 1)
    
    # 图像归一化
    # if np.max(sonar_image) > 0:
    #     sonar_image = sonar_image / np.max(sonar_image)
    
    # # 应用高斯模糊使点更明显
    # sonar_image = cv2.GaussianBlur(sonar_image, (5, 5), 0)
    
    # # 应用伽马校正增强对比度
    # sonar_image = np.power(sonar_image, gamma)
    
    sonar_color = cv2.applyColorMap(
            (200 * sonar_image).astype(np.uint8), 
            cv2.COLORMAP_HOT
        )
    
    return sonar_image, sonar_color

        
        

if __name__ == "__main__":
    # 设置参数
    depth_data_dir = "/home/clp/catkin_ws/src/airsim_sonar/scripts/depth_data"  # 包含深度数据的目录
    save_dir = "sonar_results"    # 保存结果的目录
    
    
    """
    处理目录中的所有深度数据文件并生成声纳图
    
    参数:
        depth_data_dir: 包含深度数据文件的目录
        save_dir: 保存声纳图的目录，如果为None则不保存
        max_depth: 最大深度值
        resolution_factor: 分辨率因子
    """
    # 获取目录中所有的.npy文件
    npy_files = glob.glob(os.path.join(depth_data_dir, "*.npy"))
    npy_files = sorted(npy_files, key=lambda x: os.path.basename(x).split('_')[1] + os.path.basename(x).split('_')[2]  + os.path.basename(x).split('_')[3])
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for current_file in npy_files:
        # current_file = npy_files[0]
        filename = os.path.basename(current_file)
        print(f"处理文件: {filename}")
        
        # 加载深度数据
        depth_data = load_depth_from_npy(current_file)
        # depth_colormap = visualize_depth(depth_data)
        # cv2.imshow("depth", depth_colormap)
        # cv2.waitKey(0)
            
        # 可视化深度图和声纳图
        
        # 将声纳图转换为可视化图像
        sonar_image, sonar_color = generate_sonar_view(depth_data)
        # sonar_color = cv2.applyColorMap(
        #     (255 * sonar_image).astype(np.uint8), 
        #     cv2.COLORMAP_HOT
        # )
        # sonar_color = cv2.cvtColor(sonar_color, cv2.COLOR_BGR2RGB)
        cv2.imshow("sonar_color", sonar_color)
        cv2.waitKey(500)
        