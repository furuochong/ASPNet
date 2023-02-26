# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 05:47:56 2023

@author: f6325
"""

import numpy as np
import open3d as o3d

a = np.load("D:/matlab/NYUCAD_npz/NYUCAD_npz/NYU0007_0000_voxels.npz")['target_lr']
a = np.reshape(a, (60,36,60))

color_table = [[0, 0, 0],    # 0 empty, free space
                     [214,  38, 40],    # 1 ceiling
                     [43, 160, 4],      # 2 floor
                     [158, 216, 229],   # 3 wall
                     [114, 158, 206],   # 4 window
                     [204, 204, 91],    # 5 chair  new: 180, 220, 90
                     [255, 186, 119],   # 6 bed
                     [147, 102, 188],   # 7 sofa
                     [30, 119, 181],    # 8 table
                     [188, 188, 33],    # 9 tvs
                     [255, 127, 12],    # 10 furn
                     [196, 175, 214],   # 11 objects
                     ]

points = []
colors = []
for i in range(60):
    for j in range(36):
        for k in range(60):
            if a[i,j,k]!=255 and a[i,j,k]!=0:
                points.append(np.array([[i,j,k]]))
                colors.append(np.array([color_table[a[i,j,k]]])/255)

points = np.concatenate(points, 0)
colors = np.concatenate(colors, 0)



pcd=o3d.open3d.geometry.PointCloud()
pcd.points= o3d.open3d.utility.Vector3dVector(points)
pcd.colors= o3d.open3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("D:/matlab/NYUCAD_npz/tsdf/0001.ply", pcd)
            
