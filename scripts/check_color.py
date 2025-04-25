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


# 连接到AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

import pprint
# 获取所有对象的分割ID

success = client.simSetSegmentationObjectID("[\w]*", 0, True)
print(success)

success = client.simSetSegmentationObjectID("SM_URockB[\w]*", 21, True)
print(success)

id = client.simGetSegmentationObjectID("[\w]*")
print(id)
# # 使用pprint美化输出
# print("分割对象颜色映射:")
# pprint.pprint(objects_info)