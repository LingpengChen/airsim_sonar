import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CGSIM:
    """计算机图形学仿真器，用于创建3D环境、生成深度图和点云"""
    
    def __init__(self, width=640, height=480, fx=500, fy=None, cx=None, cy=None, max_depth=10.0):
        """
        初始化仿真器
        
        参数:
            width, height: 图像分辨率
            fx, fy: 相机焦距
            cx, cy: 相机主点
            max_depth: 最大深度值（米）
        """
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy if fy is not None else fx
        self.cx = cx if cx is not None else width / 2
        self.cy = cy if cy is not None else height / 2
        self.max_depth = max_depth
        
        # 3D环境数据结构
        self.objects = []
        self.walls = []
        
        # 当前深度图
        self.depth_data = None
        
    def create_3d_env(self, env_type="simple"):
        """
        创建3D环境，包括墙面和物体
        
        参数:
            env_type: 环境类型，可选 "simple", "corridor", "room"
        
        返回:
            self，用于链式调用
        """
        # 清空当前环境
        self.objects = []
        self.walls = []
        
        if env_type == "simple":
            # 添加一面墙
            self.walls.append({
                "type": "plane",
                "normal": [0, 0, 1],  # 法向量指向相机
                "distance": 5.0,      # 距离相机5米
                "width": 10.0,        # 墙的宽度
                "height": 8.0         # 墙的高度
            })
            
            # 添加一个球体
            self.objects.append({
                "type": "sphere",
                "center": [0, 0, 3.0],  # 中心位置
                "radius": 0.7           # 半径（米）
            })
            
        elif env_type == "corridor":
            # 前墙
            self.walls.append({
                "type": "plane",
                "normal": [0, 0, 1],
                "distance": 8.0,
                "width": 3.0,
                "height": 3.0
            })
            
            # 左墙
            self.walls.append({
                "type": "plane",
                "normal": [1, 0, 0],
                "distance": 1.5,
                "width": 8.0,
                "height": 3.0
            })
            
            # 右墙
            self.walls.append({
                "type": "plane",
                "normal": [-1, 0, 0],
                "distance": 1.5,
                "width": 8.0,
                "height": 3.0
            })
            
        elif env_type == "room":
            # 四面墙、地板和天花板
            # 前墙
            self.walls.append({
                "type": "plane",
                "normal": [0, 0, 1],
                "distance": 6.0,
                "width": 5.0,
                "height": 3.0
            })
            
            # 后墙
            self.walls.append({
                "type": "plane",
                "normal": [0, 0, -1],
                "distance": 2.0,
                "width": 5.0,
                "height": 3.0
            })
            
            # 左墙
            self.walls.append({
                "type": "plane",
                "normal": [1, 0, 0],
                "distance": 2.5,
                "width": 8.0,
                "height": 3.0
            })
            
            # 右墙
            self.walls.append({
                "type": "plane",
                "normal": [-1, 0, 0],
                "distance": 2.5,
                "width": 8.0,
                "height": 3.0
            })
            
            # 地板
            self.walls.append({
                "type": "plane",
                "normal": [0, 1, 0],
                "distance": 1.5,
                "width": 5.0,
                "height": 8.0
            })
            
            # 天花板
            self.walls.append({
                "type": "plane",
                "normal": [0, -1, 0],
                "distance": 1.5,
                "width": 5.0,
                "height": 8.0
            })
            
            # 添加几个物体
            self.objects.append({
                "type": "sphere",
                "center": [-1.0, -0.5, 3.0],
                "radius": 0.5
            })
            
            self.objects.append({
                "type": "sphere",
                "center": [1.5, 0, 4.0],
                "radius": 0.7
            })
            
            self.objects.append({
                "type": "box",
                "center": [0, -1.0, 5.0],
                "dimensions": [1.0, 1.0, 1.0]
            })
        
        return self
    
    def visualize_3d_env(self, figsize=(10, 8), point_size=50):
        """
        可视化3D环境
        
        参数:
            figsize: 图像大小
            point_size: 点的大小
            
        返回:
            matplotlib figure对象
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置相机位置为原点
        ax.scatter([0], [0], [0], color='red', s=100, label='Camera')
        
        # 可视化墙面（简化为采样点）
        for wall in self.walls:
            if wall["type"] == "plane":
                normal = np.array(wall["normal"])
                distance = wall["distance"]
                width = wall["width"]
                height = wall["height"]
                
                # 创建墙面的基向量
                if np.abs(normal[1]) > 0.9:  # 如果法向量接近y轴
                    u_vec = np.array([1, 0, 0])
                else:
                    u_vec = np.array([0, 1, 0])
                
                v_vec = np.cross(normal, u_vec)
                v_vec = v_vec / np.linalg.norm(v_vec)
                u_vec = np.cross(v_vec, normal)
                u_vec = u_vec / np.linalg.norm(u_vec)
                
                # 计算墙面中心
                center = normal * distance
                
                # 生成墙面上的网格点
                u_samples = np.linspace(-width/2, width/2, 20)
                v_samples = np.linspace(-height/2, height/2, 20)
                
                for u in u_samples:
                    for v in v_samples:
                        point = center + u * u_vec + v * v_vec
                        ax.scatter(point[0], point[1], point[2], color='blue', s=point_size/10, alpha=0.3)
        
        # 可视化物体
        for obj in self.objects:
            if obj["type"] == "sphere":
                center = obj["center"]
                radius = obj["radius"]
                
                # 为球体生成采样点
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 10)
                
                for i in range(len(u)):
                    for j in range(len(v)):
                        x = center[0] + radius * np.sin(v[j]) * np.cos(u[i])
                        y = center[1] + radius * np.sin(v[j]) * np.sin(u[i])
                        z = center[2] + radius * np.cos(v[j])
                        ax.scatter(x, y, z, color='green', s=point_size/5, alpha=0.5)
            
            elif obj["type"] == "box":
                center = np.array(obj["center"])
                dims = np.array(obj["dimensions"]) / 2
                
                # 生成立方体的8个顶点
                corners = np.array([
                    [dims[0], dims[1], dims[2]],
                    [dims[0], dims[1], -dims[2]],
                    [dims[0], -dims[1], dims[2]],
                    [dims[0], -dims[1], -dims[2]],
                    [-dims[0], dims[1], dims[2]],
                    [-dims[0], dims[1], -dims[2]],
                    [-dims[0], -dims[1], dims[2]],
                    [-dims[0], -dims[1], -dims[2]]
                ])
                
                # 平移到中心位置
                corners = corners + center
                
                # 绘制顶点
                for corner in corners:
                    ax.scatter(corner[0], corner[1], corner[2], color='orange', s=point_size)
        
        # 设置坐标轴标签和标题
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D Environment Visualization')
        
        # 添加图例
        ax.legend()
        
        # 调整坐标轴范围
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([0, 10])
        
        # 设置合适的视角
        ax.view_init(elev=20, azim=-40)
        
        plt.tight_layout()
        return fig
    
    def generate_depth_data(self):
        """
        根据3D环境生成深度图数据
        
        返回:
            depth_data: 深度图数据，shape为(height, width)
        """
        # 初始化深度图，全部设为最大深度
        depth_data = np.ones((self.height, self.width)) * self.max_depth
        
        # 创建图像平面上的坐标网格
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        
        # 计算归一化的方向向量
        x_norm = (x - self.cx) / self.fx
        y_norm = (y - self.cy) / self.fy
        
        # 对每个像素计算视线与场景的交点
        for i in range(self.height):
            for j in range(self.width):
                # 视线方向向量
                direction = np.array([x_norm[i, j], y_norm[i, j], 1.0])
                direction = direction / np.linalg.norm(direction)
                
                min_distance = self.max_depth
                
                # 检查与墙面的交点
                for wall in self.walls:
                    if wall["type"] == "plane":
                        normal = np.array(wall["normal"])
                        wall_distance = wall["distance"]
                        
                        # 计算视线与平面的交点
                        dn = np.dot(direction, normal)
                        if abs(dn) > 1e-6:  # 非平行
                            t = wall_distance / dn
                            
                            if t > 0 and t < min_distance:
                                # 计算交点
                                intersection = t * direction
                                
                                # 检查交点是否在墙面范围内
                                if wall["width"] >= 0 and wall["height"] >= 0:
                                    # 创建墙面的基向量
                                    if np.abs(normal[1]) > 0.9:  # 如果法向量接近y轴
                                        u_vec = np.array([1, 0, 0])
                                    else:
                                        u_vec = np.array([0, 1, 0])
                                    
                                    v_vec = np.cross(normal, u_vec)
                                    v_vec = v_vec / np.linalg.norm(v_vec)
                                    u_vec = np.cross(v_vec, normal)
                                    u_vec = u_vec / np.linalg.norm(u_vec)
                                    
                                    # 计算交点在墙面坐标系中的位置
                                    center = normal * wall_distance
                                    rel_pos = intersection - center
                                    u_coord = np.dot(rel_pos, u_vec)
                                    v_coord = np.dot(rel_pos, v_vec)
                                    
                                    if (abs(u_coord) <= wall["width"]/2 and 
                                        abs(v_coord) <= wall["height"]/2):
                                        min_distance = t
                
                # 检查与物体的交点
                for obj in self.objects:
                    if obj["type"] == "sphere":
                        center = np.array(obj["center"])
                        radius = obj["radius"]
                        
                        # 计算视线与球体的交点
                        # 解方程: |origin + t*direction - center|^2 = radius^2
                        oc = -center  # 相机在原点
                        a = 1.0  # |direction|^2 = 1，因为direction已归一化
                        b = 2.0 * np.dot(direction, oc)
                        c = np.dot(oc, oc) - radius * radius
                        
                        discriminant = b*b - 4*a*c
                        
                        if discriminant >= 0:
                            # 有交点
                            t = (-b - np.sqrt(discriminant)) / (2.0 * a)
                            
                            if t > 0 and t < min_distance:
                                min_distance = t
                    
                    elif obj["type"] == "box":
                        center = np.array(obj["center"])
                        dims = np.array(obj["dimensions"]) / 2
                        
                        # 简化计算：检查射线与AABB（轴对齐包围盒）的交点
                        # 计算每个轴上的入射点和出射点
                        t_min = -np.inf
                        t_max = np.inf
                        
                        for a in range(3):  # x, y, z轴
                            if abs(direction[a]) < 1e-6:  # 接近平行
                                if -center[a] - dims[a] > 0 or -center[a] + dims[a] < 0:
                                    # 射线在盒子外且平行于盒子面
                                    continue
                            
                            inv_dir = 1.0 / direction[a]
                            t1 = (-center[a] - dims[a]) * inv_dir
                            t2 = (-center[a] + dims[a]) * inv_dir
                            
                            if t1 > t2:
                                t1, t2 = t2, t1
                            
                            t_min = max(t_min, t1)
                            t_max = min(t_max, t2)
                            
                            if t_min > t_max:
                                break
                        
                        if t_min <= t_max and t_min > 0 and t_min < min_distance:
                            min_distance = t_min
                
                # 将最近交点的距离存入深度图
                depth_data[i, j] = min_distance
        
        # 添加一些随机噪声模拟真实传感器
        noise = np.random.normal(0, 0.01, depth_data.shape) * depth_data
        depth_data += noise
        
        # 保存当前深度图
        self.depth_data = depth_data
        
        return depth_data
    
    def visualize_depth_data(self, depth_data=None, figsize=(10, 8)):
        """
        可视化深度图为彩色图像
        
        参数:
            depth_data: 深度图数据，如果为None则使用当前深度图
            figsize: 图像大小
            
        返回:
            matplotlib figure对象
        """
        if depth_data is None:
            if self.depth_data is None:
                raise ValueError("No depth data available. Call generate_depth_data() first.")
            depth_data = self.depth_data
        
        fig = plt.figure(figsize=figsize)
        
        # 将深度图归一化并转换为彩色图
        normalized_depth = np.clip(depth_data / self.max_depth, 0, 1)
        depth_color = cv2.applyColorMap(
            (255 * normalized_depth).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        
        plt.imshow(depth_color)
        plt.title('Perspective Depth Image')
        plt.colorbar(label='Depth (normalized)')
        plt.axis('off')
        
        plt.tight_layout()
        return fig
    
    def depth_to_point_cloud(self, depth_data=None, downsample=10):
        """
        将深度图转换为3D点云，考虑深度值是沿视线方向的距离
        
        参数:
            depth_data: 深度图数据，如果为None则使用当前深度图
            downsample: 下采样因子，以减少点数
            
        返回:
            点云坐标 (N, 3)，包含X, Y, Z
        """
        if depth_data is None:
            if self.depth_data is None:
                raise ValueError("No depth data available. Call generate_depth_data() first.")
            depth_data = self.depth_data
        
        # 创建网格坐标 (下采样以减少点数)
        v, u = np.mgrid[0:self.height:downsample, 0:self.width:downsample]
        v = v.flatten()
        u = u.flatten()
        
        # 获取深度值（沿视线的距离）
        ray_distance = depth_data[v, u]
        
        # 过滤无效或最大深度值
        valid = (ray_distance > 0) & (ray_distance < self.max_depth * 0.99)
        ray_distance = ray_distance[valid]
        v = v[valid]
        u = u[valid]
        
        # 计算归一化的方向向量
        dx = (u - self.cx) / self.fx
        dy = (v - self.cy) / self.fy
        
        # 计算视线方向与Z轴夹角的余弦值
        cos_angle = 1.0 / np.sqrt(dx**2 + dy**2 + 1.0)
        
        # 计算实际Z值（考虑余弦校正）
        z = ray_distance * cos_angle
        
        # 通过Z值反投影计算X和Y值
        x = dx * z
        y = dy * z
        
        # 返回点云坐标
        return np.vstack((x, y, z)).T
    
    def visualize_point_cloud(self, depth_data=None, downsample=10, figsize=(10, 8)):
        """
        可视化深度图对应的3D点云
        
        参数:
            depth_data: 深度图数据，如果为None则使用当前深度图
            downsample: 下采样因子
            figsize: 图像大小
            
        返回:
            matplotlib figure对象
        """
        # 获取点云数据
        points = self.depth_to_point_cloud(depth_data, downsample)
        
        # 创建3D图像
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制点云（使用深度值进行着色）
        sc = ax.scatter(points[:, 0], points[:, 2], -points[:, 1], 
                       c=points[:, 2], cmap='jet', s=2, alpha=0.5)
        
        # 设置轴标签和标题
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Z (meters)')
        ax.set_zlabel('-Y (meters)')
        ax.set_title('3D Point Cloud from Depth Data')
        
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
        
        plt.tight_layout()
        return fig


class SonarSIM:
    """声纳模拟器，用于将深度图转换为声纳视图"""
    
    def __init__(self, fx=500, cx=None, max_depth=10.0, gamma=0.7):
        """
        初始化声纳模拟器
        
        参数:
            fx: 相机水平焦距
            cx: 相机主点x坐标，默认为图像宽度的一半
            max_depth: 最大深度值（米）
            gamma: 伽马校正系数
        """
        self.fx = fx
        self.cx = cx  # 会在generate_sonar_view中设置
        self.max_depth = max_depth
        self.gamma = gamma
    
    def generate_sonar_view(self, depth_data, fx=None, cx=None, max_depth=None, gamma=None):
        """
        从透视深度图生成声纳视图，保持原始图像的宽度，应用余弦校正
        
        参数:
            depth_data: 深度图像数据 (H, W) 的numpy数组，值代表沿视线的距离
            fx: 相机内参中的水平焦距，如果为None则使用初始化值
            cx: 相机内参中的水平主点坐标，如果为None则使用图像中心
            max_depth: 最大深度值截断，如果为None则使用初始化值
            gamma: 伽马校正系数，如果为None则使用初始化值
            
        返回:
            声纳视图图像
        """
        height, width = depth_data.shape
        
        # 使用传入的参数，如果没有则使用初始化时的参数
        fx = fx if fx is not None else self.fx
        cx = cx if cx is not None else (width / 2 if self.cx is None else self.cx)
        max_depth = max_depth if max_depth is not None else self.max_depth
        gamma = gamma if gamma is not None else self.gamma
        
        # 创建声纳图（初始为全0），保持原始宽度
        sonar_image = np.zeros_like(depth_data, dtype=np.float32)
        
        # 截断深度值
        valid_depth = np.clip(depth_data, 0, max_depth)
        
        # 获取有效深度点的坐标和值
        v_indices, u_indices = np.where(valid_depth > 0)
        depths = valid_depth[v_indices, u_indices]
        
        # 计算归一化的方向向量的x分量
        dx = (u_indices - cx) / fx
        
        # 计算视线方向与Z轴夹角的余弦值
        # 注意：视线方向为 (dx, dz) 其中 dz = 1.0
        cos_alpha = 1.0 / np.sqrt(dx**2 + 1.0)
        
        # 计算实际水平距离（应用余弦校正）
        # 实际距离 = 视线距离 / cos(alpha)
        # r_actual = depths / cos_alpha
        r_actual = depths 
        
        # 限制最大距离
        r_actual = np.clip(r_actual, 0, max_depth)
        
        # 计算声纳图中的位置
        u_sonar = u_indices
        v_sonar = (height * (1 - r_actual / max_depth)).astype(int)
        
        # 确保坐标在有效范围内
        valid_coords = (v_sonar >= 0) & (v_sonar < height)
        u_sonar = u_sonar[valid_coords]
        v_sonar = v_sonar[valid_coords]
        
        # 在声纳图像中标记点（累积密度）
        for u, v in zip(u_sonar, v_sonar):
            sonar_image[v, u] += 1.0
        
        # 图像归一化
        if np.max(sonar_image) > 0:
            sonar_image = sonar_image / np.max(sonar_image)
        
        # 应用高斯模糊使点更明显
        sonar_image = cv2.GaussianBlur(sonar_image, (3, 3), 0)
        
        # 应用伽马校正增强对比度
        sonar_image = np.power(sonar_image, gamma)
        
        return sonar_image
    
    def visualize_sonar_view(self, sonar_image, figsize=(10, 8)):
        """
        可视化声纳视图为彩色图像
        
        参数:
            sonar_image: 声纳图像数据
            figsize: 图像大小
            
        返回:
            matplotlib figure对象
        """
        fig = plt.figure(figsize=figsize)
        
        # 将声纳图可视化为彩色图
        sonar_color = cv2.applyColorMap(
            (255 * sonar_image).astype(np.uint8),
            cv2.COLORMAP_HOT
        )
        sonar_color = cv2.cvtColor(sonar_color, cv2.COLOR_BGR2RGB)
        
        plt.imshow(sonar_color)
        plt.title('Sonar View')
        plt.axis('off')
        
        plt.tight_layout()
        return fig


def main():
    """测试程序，展示完整工作流程"""
    
    # 1. 初始化3D图形仿真器
    cgsim = CGSIM(width=640, height=480, fx=500, max_depth=10.0)
    
    # 2. 创建不同类型的3D环境
    # env_types = ["simple", "corridor", "room"]
    env_types = ["simple"]
    
    
    for env_type in env_types:
        print(f"\n正在处理环境类型: {env_type}")
        
        # 创建环境
        cgsim.create_3d_env(env_type)
        
        # 可视化3D环境
        fig_env = cgsim.visualize_3d_env()
        print(f"已显示3D环境")
        plt.show()
        
        # 生成深度图
        depth_data = cgsim.generate_depth_data()
        
        # 可视化深度图
        fig_depth = cgsim.visualize_depth_data()
        print(f"已显示深度图像")
        plt.show()
        
        # 可视化点云
        fig_pc = cgsim.visualize_point_cloud(downsample=8)
        print(f"已显示点云图像")
        plt.show()
        
        # 3. 初始化声纳模拟器
        sonar_sim = SonarSIM(fx=500, max_depth=10.0, gamma=0.7)
        
        # 4. 生成声纳视图
        sonar_image = sonar_sim.generate_sonar_view(depth_data)
        
        # 5. 可视化声纳视图
        fig_sonar = sonar_sim.visualize_sonar_view(sonar_image)
        print(f"已显示声纳视图")
        plt.show()
        
        # 6. 创建组合图像（深度图、点云和声纳图）
        plt.figure(figsize=(18, 6))
        
        # 深度图
        plt.subplot(1, 3, 1)
        depth_color = cv2.applyColorMap(
            (255 * np.clip(depth_data / cgsim.max_depth, 0, 1)).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        plt.imshow(depth_color)
        plt.title(f"Depth Map - {env_type}")
        plt.axis('off')
        
        # 声纳图
        plt.subplot(1, 3, 2)
        sonar_color = cv2.applyColorMap(
            (255 * sonar_image).astype(np.uint8),
            cv2.COLORMAP_HOT
        )
        sonar_color = cv2.cvtColor(sonar_color, cv2.COLOR_BGR2RGB)
        plt.imshow(sonar_color)
        plt.title(f"Sonar View - {env_type}")
        plt.axis('off')
        
        # 3D点云
        ax = plt.subplot(1, 3, 3, projection='3d')
        points = cgsim.depth_to_point_cloud(downsample=10)
        sc = ax.scatter(points[:, 0], points[:, 2], -points[:, 1],
                      c=points[:, 2], cmap='jet', s=2, alpha=0.5)
        plt.colorbar(sc, label='Depth (meters)')
        plt.title(f"3D Point Cloud - {env_type}")
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Z (meters)')
        ax.set_zlabel('-Y (meters)')
        ax.view_init(elev=20, azim=-60)
        
        plt.tight_layout()
        plt.show()
        print(f"已显示组合视图")
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    main()