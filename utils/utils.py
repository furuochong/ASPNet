# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:34:54 2022

@author: f6325
"""

import numpy as np
import open3d as o3d
import os
import torch

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


path = "E:/dataset/NYU/TSDF/"
files = os.listdir(path)

number=0

for file in files:
    print(file)
    img = torch.from_numpy(np.ones((480,640)))
    img = img.unsqueeze(0).unsqueeze(0).cuda()
    mapping = torch.from_numpy(np.load("E:/dataset/NYU/Mapping/"+file)['arr_0']).unsqueeze(0).long().cuda()
    tsdf = torch.from_numpy(np.load("E:/dataset/NYU/TSDF/"+file)['arr_0']).view(60,36,60)
    gt = torch.from_numpy(np.load("E:/dataset/NYU/Label/"+file)['arr_0']).view(60,36,60)
    label_weight = torch.from_numpy(np.load("E:/dataset/NYU/TSDF/"+file)['arr_1']).view(60,36,60)

    b, c, h, w = img.shape
    img = img.view(b, c, h * w).permute(0, 2, 1)  # b x h*w x c

    zerosVec = torch.zeros(b, 1, c).cuda()  # for voxels that could not be projected from the depth map, we assign them zero vector
    segVec = torch.cat((img, zerosVec), 1)

    segres = [torch.index_select(segVec[i], 0, mapping[i]) for i in range(b)]
    segres = torch.stack(segres).permute(0, 2, 1).contiguous().view(b, c, 60, 36, 60)  # B, (channel), 60, 36, 60


    occupied = torch.squeeze(segres)
    
    
    seg = torch.from_numpy(np.load("E:/paper/SSC/SSC_points/dataset/seg_12/"+file.split('.')[0]+'.npy')).unsqueeze(0).cuda()
    _,seg = torch.max(seg, 1)
    seg = seg.unsqueeze(0)
    
    b, c, h, w = seg.shape
    seg = seg.view(b, c, h * w).permute(0, 2, 1)
    
    zerosVec = torch.zeros(b, 1, c).cuda()  # for voxels that could not be projected from the depth map, we assign them zero vector
    segVec = torch.cat((seg, zerosVec), 1)
    
    segres = [torch.index_select(segVec[i], 0, mapping[i]) for i in range(b)]
    segres = torch.stack(segres).permute(0, 2, 1).contiguous().view(b, c, 60, 36, 60).squeeze()  # B, (channel), 60, 36, 60
    
    seg_class = segres


    gt_mask = torch.zeros((60,36,60)).long()
    
    for i in range(60):
        for j in range(36):
            for k in range(60):
                if k==59 and occupied[i,j,k]==0:
                    print("hello")
                elif occupied[i,j,k]==1 and gt[i,j,k]!=0:
                    
                    break

  
    points = []
    colors = []
    points_imp = []
    colors_imp = []
    
       
    """
    for i in range(60):
        for j in range(36):
            k_min = 60
            for k in range(60):    
                class_n = gt[i,j,59-k] 
                class_s = int(seg_class[i,j,59-k])
                if occupied[i,j,59-k]==1 and class_n!=255:
                    k_min = 59-k
                    points.append(np.array([[i,j,59-k]]))
                    colors.append(np.array([color_table[class_s]])/255)
            if k_min<=59:
                class_n = gt[i,j,k_min] 
                class_s = int(seg_class[i,j,k_min])
                points_imp.append(np.array([[i,j,k_min]]))
                colors_imp.append(np.array([color_table[class_s]])/255)
                
                for k in range(k_min+1, 60):
                    class_n = gt[i,j,k] 
                    class_s = int(seg_class[i,j,k])
                    if class_n!=255 and label_weight[i,j,k]==1:
                        points_imp.append(np.array([[i,j,k]]))
                        colors_imp.append(np.array([color_table[class_s]])/255)
                        
            if k_min==60:
                for k in range(60):
                    class_n = gt[i,j,k] 
                    if class_n!=255 and label_weight[i,j,k]==1:
                        points_imp.append(np.array([[i,j,k]]))
                        colors_imp.append(np.array([color_table[0]])/255)
            

                    

                
                    

    points_imp = np.concatenate(points_imp, 0)
    colors_imp = np.concatenate(colors_imp, 0)
    
    points = np.concatenate(points, 0)
    colors = np.concatenate(colors, 0)
    

    pcd_imp=o3d.open3d.geometry.PointCloud()
    pcd_imp.points= o3d.open3d.utility.Vector3dVector(points_imp)
    pcd_imp.colors= o3d.open3d.utility.Vector3dVector(colors_imp)
    o3d.io.write_point_cloud("E:/paper/SSC/SSC_points/dataset/points/"+file.split('.')[0]+'.ply', pcd_imp)


    pcd=o3d.open3d.geometry.PointCloud()
    pcd.points= o3d.open3d.utility.Vector3dVector(points)
    pcd.colors= o3d.open3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud("E:/paper/SSC/SSC_points/dataset/surface/"+file.split('.')[0]+'.ply', pcd)
    """








