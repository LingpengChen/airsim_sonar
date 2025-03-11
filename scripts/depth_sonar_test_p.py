import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_sonar_view(depth_data, fx=533, cx=None, max_depth=10.0, gamma=0.7):
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
    height, width =  depth_data.shape[1],  depth_data.shape[1]
    
    # 设置默认相机主点为图像中心
    if cx is None:
        cx = width / 2
    
    # 创建声纳图（初始为全0），保持原始宽度
    sonar_image = np.zeros_like(depth_data, dtype=np.float32)

    v_indices, u_indices = np.where((depth_data > 0) & (depth_data <= max_depth))
    depths = depth_data[v_indices, u_indices]
    
    # 计算归一化的方向向量的x分量 tan(alpha )
    dx = (u_indices - cx) / fx
    
    # 计算视线方向与Z轴夹角的余弦值
    # 注意：视线方向为 (dx, dz) 其中 dz = 1.0
    cos_alpha = 1.0 / np.sqrt(dx**2 + 1.0)
    
    # 计算实际水平距离（应用余弦校正）
    # 修改为: 实际距离 = 视线距离 * cos(alpha)
    r_actual = depths * cos_alpha
    
    
    # 计算声纳图中的垂直位置
    v_sonar = (height * (1 - r_actual / max_depth)).astype(int)
    
    # 计算声纳图中的水平位置
    # 需要适当缩放确保坐标在合理范围内
    scale_factor = width / (4 * max(np.max(np.abs(dx * r_actual)), 1))  # 动态调整缩放因子
    u_sonar = (cx + scale_factor * dx * r_actual).astype(int)
    
    # 确保坐标在有效范围内
    valid_coords = (v_sonar >= 0) & (v_sonar < height) & (u_sonar >= 0) & (u_sonar < width)
    u_sonar = u_sonar[valid_coords]
    v_sonar = v_sonar[valid_coords]
    r_actual = r_actual[valid_coords]  # 同样过滤深度值，用于后续计算
    
    # 在声纳图像中标记点（累积密度）
    # 使用深度值作为权重，更远的点权重更小
    weights = 1.0 / (1.0 + r_actual)  # 简单的距离衰减公式
    
    for u, v, w in zip(u_sonar, v_sonar, weights):
        sonar_image[v, u] += w
    
    # 图像归一化
    if np.max(sonar_image) > 0:
        sonar_image = sonar_image / np.max(sonar_image)
    
    # 应用高斯模糊使点更明显
    sonar_image = cv2.GaussianBlur(sonar_image, (5, 5), 0)
    
    # 应用伽马校正增强对比度
    sonar_image = np.power(sonar_image, gamma)
    
    return sonar_image

def create_synthetic_depth_map(width=640, height=480, fx=500, cx=None, scene_type="flat_wall"):
    """
    创建合成深度图进行测试
    
    参数:
        width, height: 深度图尺寸
        fx: 相机水平焦距
        cx: 相机主点x坐标，默认为图像中心
        scene_type: 场景类型，可选 "flat_wall", "corridor", "objects"
        
    返回:
        合成的深度图
    """
    if cx is None:
        cx = width / 2
    cy = height / 2
    
    # 创建网格坐标
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 计算归一化坐标
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fx
    
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    if scene_type == "flat_wall":
        # 模拟一面平坦的墙，距离为5米
        wall_distance = 5.0
        # 考虑透视投影：depth = Z / cos(alpha)，其中alpha是视线与Z轴的夹角
        # cos(alpha) = 1 / sqrt(1 + x_norm^2 + y_norm^2)
        cos_angle = 1.0 / np.sqrt(1 + x_norm**2 + y_norm**2)
        depth_map = wall_distance / cos_angle
        
    elif scene_type == "corridor":
        # 模拟走廊，有左右两面墙和前面一面墙
        corridor_width = 3.0  # 走廊宽度（米）
        corridor_length = 8.0  # 走廊长度（米）
        
        # 计算3D点坐标（假设相机在走廊中心，距离后墙4米）
        cam_to_back = 0.0  # 相机到后墙的距离
        cam_to_front = corridor_length  # 相机到前墙的距离
        half_width = corridor_width / 2
        
        # X坐标（水平方向）
        X = x_norm * (cam_to_front - cam_to_back)
        
        # 前方墙的深度
        front_wall = np.ones((height, width)) * cam_to_front
        # 左侧墙 (X < -half_width)
        left_wall_mask = X < -half_width
        left_wall_depth = -half_width / x_norm
        left_wall_depth[left_wall_depth < 0] = 0  # 避免负值
        # 右侧墙 (X > half_width)
        right_wall_mask = X > half_width
        right_wall_depth = half_width / x_norm
        right_wall_depth[right_wall_depth < 0] = 0  # 避免负值
        
        # 组合所有墙
        depth_map = front_wall
        depth_map[left_wall_mask] = np.minimum(front_wall[left_wall_mask], left_wall_depth[left_wall_mask])
        depth_map[right_wall_mask] = np.minimum(front_wall[right_wall_mask], right_wall_depth[right_wall_mask])
        
    elif scene_type == "objects":
        # 模拟场景中的多个物体
        # 背景墙
        background_dist = 8.0
        cos_angle = 1.0 / np.sqrt(1 + x_norm**2 + y_norm**2)
        depth_map = background_dist / cos_angle
        
        # 添加几个球体
        objects = [
            {"x": -1.5, "y": 0, "z": 3.0, "radius": 0.5},  # 左边的物体
            {"x": 0, "y": 0, "z": 4.0, "radius": 0.7},     # 中间的物体
            {"x": 2.0, "y": 0, "z": 5.0, "radius": 1.0},   # 右边的物体
        ]
        
        # 对于每个点，计算到相机的距离
        for obj in objects:
            # 计算视线与物体的相交
            # 简化：使用距离场近似
            obj_x = obj["x"]
            obj_z = obj["z"]
            obj_radius = obj["radius"]
            
            # 计算视线方向上对应的点
            ray_dirs_x = x_norm
            ray_dirs_z = np.ones_like(x_norm)
            ray_length = np.sqrt(ray_dirs_x**2 + ray_dirs_z**2)
            ray_dirs_x = ray_dirs_x / ray_length
            ray_dirs_z = ray_dirs_z / ray_length
            
            # 计算t值使得射线位于与物体相同的XZ平面上
            t_values = obj_z / ray_dirs_z
            
            # 计算射线在该平面上的x坐标
            intersect_x = t_values * ray_dirs_x
            
            # 计算到物体中心的距离
            dist_to_center = np.abs(intersect_x - obj_x)
            
            # 如果距离小于半径，则射线与物体相交
            intersect_mask = (dist_to_center < obj_radius) & (t_values > 0)
            depth_map[intersect_mask] = np.minimum(depth_map[intersect_mask], t_values[intersect_mask])
    
    # 为了模拟真实传感器数据，限制最大深度并添加一些随机噪声
    depth_map = np.clip(depth_map, 0.1, 10.0)
    # 添加噪声
    # noise = np.random.normal(0, 0.02, depth_map.shape) * depth_map
    # depth_map += noise
    
    return depth_map

def depth_to_point_cloud(depth_map, fx, fy=None, cx=None, cy=None, downsample=10):
    """
    将深度图转换为点云，考虑深度值是沿视线方向的距离
    
    参数:
        depth_map: 深度图数据，表示沿视线的距离
        fx, fy: 相机焦距
        cx, cy: 相机主点
        downsample: 下采样因子，以减少点数
        
    返回:
        点云坐标 (N, 3)，包含X, Y, Z
    """
    height, width = depth_map.shape
    
    if fy is None:
        fy = fx
    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2
    
    # 创建网格坐标 (下采样以减少点数)
    v, u = np.mgrid[0:height:downsample, 0:width:downsample]
    v = v.flatten()
    u = u.flatten()
    
    # 获取深度值（沿视线的距离）
    ray_distance = depth_map[v, u]
    
    # 过滤无效深度
    valid = ray_distance > 0
    ray_distance = ray_distance[valid]
    v = v[valid]
    u = u[valid]
    
    # 计算归一化的方向向量
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    
    # 计算视线方向与Z轴夹角的余弦值
    # 视线方向向量为 (dx, dy, 1)，其长度为 sqrt(dx^2 + dy^2 + 1)
    # 余弦值为 1/sqrt(dx^2 + dy^2 + 1)
    cos_angle = 1.0 / np.sqrt(dx**2 + dy**2 + 1.0)
    
    # 计算实际Z值（考虑余弦校正）
    z = ray_distance * cos_angle
    
    # 通过Z值反投影计算X和Y值
    x = dx * z
    y = dy * z
    
    # 返回点云坐标
    return np.vstack((x, y, z)).T

def plot_point_cloud(depth_map, fx, fy=None, cx=None, cy=None, downsample=10, title="3D Point Cloud"):
    """
    绘制深度图对应的3D点云
    
    参数:
        depth_map: 深度图数据
        fx, fy: 相机焦距
        cx, cy: 相机主点
        downsample: 下采样因子
        title: 图表标题
    
    返回:
        点云图像的figure对象
    """
    # 将深度图转换为点云
    points = depth_to_point_cloud(depth_map, fx, fy, cx, cy, downsample)
    
    # 创建3D图像
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云（使用深度值进行着色）
    # ax = plt.gca()
    sc = ax.scatter(points[:, 0], points[:, 2], -points[:, 1], 
                c=points[:, 2], cmap='jet', s=2, alpha=0.5)
    
    # 设置轴标签和标题
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_zlabel('-Y (meters)')
    ax.set_title(title)
    
    # 添加颜色条
    plt.colorbar(sc, ax=ax, label='Depth (meters)')
    
    # 设置视角
    ax.view_init(elev=20, azim=-60)
    
    # 调整坐标轴范围以便更好地观察
    max_range = np.max([
        np.max(np.abs(points[:, 0])), 
        np.max(np.abs(points[:, 1])), 
        np.max(np.abs(points[:, 2]))
    ])
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([0, np.max(points[:, 2])])
    ax.set_zlim([-max_range, max_range])
    
    return fig

def main():
    # 设置参数
    width, height = 640, 480
    fx = 500  # 相机焦距
    max_depth = 20.0  # 最大深度（米）
    
    # 创建合成深度图，测试不同的场景类型
    # scenes = ["flat_wall"]
    scenes = ["flat_wall", "corridor", "objects"]
    
    
    # 为每个场景创建图像
    for i, scene_type in enumerate(scenes):
        # 创建合成深度图
        depth_map = create_synthetic_depth_map(width, height, fx, scene_type=scene_type)
        
        # 生成声纳视图
        sonar_view = generate_sonar_view(depth_map)
        
        
        # 创建一个大的图形，包含深度图、声纳图和3D点云
        plt.figure(figsize=(18, 6))
        
        # 1 将深度图可视化为彩色图
        ax1 = plt.subplot(1, 3, 1)
        depth_color = cv2.applyColorMap(
            (255 * np.clip(depth_map / max_depth, 0, 1)).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        plt.imshow(depth_color)
        plt.title(f"Depth Map - {scene_type}")
        plt.axis('off')
        
        # 2 将声纳视图可视化为彩色图
        ax2 = plt.subplot(1, 3, 2)
        sonar_color = cv2.applyColorMap(
            (255 * sonar_view).astype(np.uint8), 
            cv2.COLORMAP_HOT
        )
        sonar_color = cv2.cvtColor(sonar_color, cv2.COLOR_BGR2RGB)
        plt.imshow(sonar_color)
        plt.title(f"Sonar View - {scene_type}")
        plt.axis('off')
        
        # 3 绘制3D点云
        plt.subplot(1, 3, 3, projection='3d')
        # 为了提高性能，对点云进行下采样
        points = depth_to_point_cloud(depth_map, fx, downsample=10)
        # 按深度值着色
        ax = plt.gca()
        sc = ax.scatter(points[:, 0], points[:, 2], -points[:, 1], 
                    c=points[:, 2], cmap='jet', s=2, alpha=0.5)
        plt.colorbar(sc, label='Depth (meters)')
        plt.title(f"3D Point Cloud - {scene_type}")
        plt.xlabel('X (meters)')
        plt.ylabel('Z (meters)')
        plt.gca().set_zlabel('-Y (meters)')
        
        # 设置3D视角
        plt.gca().view_init(elev=20, azim=-60)
        
        plt.tight_layout()
        plt.show(f'sonar_view_{scene_type}.png')
        plt.close()
        
        # # 额外创建一个单独的、更大的3D点云视图用于详细观察
        # fig_3d = plot_point_cloud(depth_map, fx, downsample=8, 
        #                           title=f"3D Point Cloud - {scene_type} (Detailed View)")
        # fig_3d.show(f'point_cloud_{scene_type}.png')
        # plt.close(fig_3d)
    
    # # 创建汇总图像
    # fig_summary = plt.figure(figsize=(15, 10))
    
    # for i, scene_type in enumerate(scenes):
    #     # 加载保存的图像
    #     img = plt.imread(f'sonar_view_{scene_type}.png')
    #     plt.subplot(3, 1, i+1)
    #     plt.imshow(img)
    #     plt.axis('off')
    
    # plt.tight_layout()
    # plt.show('sonar_view_results_summary.png')
    # plt.close(fig_summary)
    
    print("测试完成，结果已保存为各个场景的图像文件")

if __name__ == "__main__":
    main()