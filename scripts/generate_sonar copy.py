#!/usr/bin/env python3

import sys, os
sys.path.append(r"/home/clp/workspace/PythonClient/multirotor")
import setup_path

import math, json
import airsim
import numpy as np
import cv2
from datetime import datetime
import time
import rospy
import rosbag
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import tf.transformations as tf_trans


import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class AirSimImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 创建窗口
        cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Segmentation Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
        
        # 订阅三种图像类型
        self.rgb_sub = rospy.Subscriber("/airsim_node/SimpleFlight/front_center/Scene", Image, self.rgb_callback)
        self.seg_sub = rospy.Subscriber("/airsim_node/SimpleFlight/front_center/Segmentation", Image, self.seg_callback)
        self.depth_sub = rospy.Subscriber("/airsim_node/SimpleFlight/front_center/DepthPerspective", Image, self.depth_callback)
        
        rospy.loginfo("AirSim Image Subscriber initialized")

    def rgb_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imshow("RGB Image", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def seg_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imshow("Segmentation Image", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def depth_callback(self, data):
        try:
            # 深度图像通常需要特殊处理以便于可视化
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            
            # 归一化深度图像以便于可视化
            depth_array = np.array(cv_image, dtype=np.float32)
            # 将深度限制在0-100米范围内以便更好地可视化
            normalized_depth = np.clip(depth_array, 0, 100) / 100.0
            normalized_depth = (normalized_depth * 255).astype(np.uint8)
            cv2.imshow("Depth Image", normalized_depth)
            # depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
            # cv2.imshow("Depth Image", depth_colormap)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

def main():
    rospy.init_node('airsim_image_subscriber', anonymous=True)
    AirSimImageSubscriber()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()