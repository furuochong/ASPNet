# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 21:50:31 2022

@author: f6325
"""

import numpy as np
import os
import open3d as o3d
path = "E:/paper/SSC/SSC_points/dataset/points"
files = os.listdir(path)

number = 0
max_number = 0
min_number = 129600

for file in files:
    
    name = os.path.join(path, file)
    pcd=o3d.io.read_point_cloud(name,format='ply')
    points=np.asarray(pcd.points)
    number+=points.shape[0]
    
    if points.shape[0]<min_number:
        min_number = points.shape[0]
    
    if points.shape[0]>max_number:
        max_number = points.shape[0]

number/=1449

print(max_number)
print(min_number)

print(number)