import sys, os
sys.path.append(r"/home/clp/workspace/PythonClient/multirotor")
import setup_path

from process_depth import generate_sonar_view

import airsim
import cv2
import numpy as np
import time
import math

from datetime import datetime

def calculate_intrinsics(fov_deg, width=640, height=480):
    fov_rad = float(fov_deg) * math.pi / 180
    fx = (width / 2.0) / math.tan(fov_rad / 2)
    fy = (height / 2.0) / math.tan(fov_rad / 2)
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy

# 连接到AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# success = client.simSetSegmentationObjectID("[\w]*", 0, True)
# print(success)
success = client.simSetSegmentationObjectID("SM_URock[\w]*", 2, True)
print(success)

success = client.simSetSegmentationObjectID("SM_KI-[\w]*", 0, True)
print(success)

success = client.simSetSegmentationObjectID("SM_Sub[\w]*", 1, True)
print(success)


success = client.simSetSegmentationObjectID("Landscape1", 4, True)
print(success)



def save_depth_to_npy(depth_data, save_dir="depth_data"):
    """
    将深度图像数据保存为.npy文件
    
    参数:
        depth_data: 深度图像数据，numpy数组
        save_dir: 保存目录，默认为'depth_data'
    
    返回:
        保存的文件路径
    """
    # 创建保存目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 生成文件名（使用时间戳确保唯一性）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"depth_{timestamp}.npy"
    filepath = os.path.join(save_dir, filename)
    
    # 保存数据
    np.save(filepath, depth_data)
    print(f"深度数据已保存至: {filepath}")
    
    return filepath


# 设置图像请求
image_requests = [
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),     # RGB图像
    airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False),  # 分割图像
    airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False),  # 深度图像
]

# 创建显示窗口
cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Segmentation Image", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
# cv2.namedWindow("sonar_image", cv2.WINDOW_NORMAL)




try:
    while True:
        # 获取图像
        responses = client.simGetImages(image_requests)
          
        # 处理RGB图像
        rgb_response = responses[0]
        rgb_image = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        rgb_image = rgb_image.reshape(rgb_response.height, rgb_response.width, 3)
        cv2.imshow("RGB Image", rgb_image)
        
        # 处理RGB图像
        seg_response = responses[1]
        seg_image = np.frombuffer(seg_response.image_data_uint8, dtype=np.uint8)
        seg_image = seg_image.reshape(seg_response.height, seg_response.width, 3)
        cv2.imshow("Segmentation Image", seg_image)
        
        
        # 处理深度图像
        depth_response = responses[2]
        depth_perspective_image = np.array(depth_response.image_data_float, dtype=np.float32)
        depth_perspective_image = depth_perspective_image.reshape(depth_response.height, depth_response.width)
        # filepath = save_depth_to_npy(depth_perspective_image)
        _, sonar_image = generate_sonar_view(depth_perspective_image)
        cv2.imshow("sonar_image", sonar_image)
        
        # normalized = np.clip(depth_perspective_image, 0, 100) / 100.0
        # normalized = (normalized * 255).astype(np.uint8)
        # depth_colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        # cv2.imshow("Depth Image", depth_colormap)
        
        
        
        
        # 显示图像并检查是否退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 控制帧率
        time.sleep(0.1)

except KeyboardInterrupt:
    pass

finally:
    # 清理窗口
    cv2.destroyAllWindows()
    print("程序已退出")