# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:34:54 2022

@author: f6325
"""

import numpy as np
import open3d as o3d
import os
import torch
from torch_scatter import scatter_max
from torch.nn import functional as F
import torch.nn as nn

class Project2Dto3D(nn.Module):
    def __init__(self, w=240, h=144, d=240):
        super(Project2Dto3D, self).__init__()
        self.w = w
        self.h = h
        self.d = d

    def forward(self, x2d, idx):
        # bs, c, img_h, img_w = x2d.shape
        bs, c, _, _ = x2d.shape
        src = x2d.view(bs, c, -1)
        idx = idx.view(bs, 1, -1)
        index = idx.expand(-1, c, -1)  # expand to c channels

        x3d = x2d.new_zeros((bs, c, self.w*self.h*self.d))
        x3d, _ = scatter_max(src, index, out=x3d)  # dim_size=240*144*240,

        x3d = x3d.view(bs, c, self.w, self.h, self.d)  # (BS, c, vW, vH, vD)
        x3d = x3d.permute(0, 1, 4, 3, 2)     # (BS, c, vW, vH, vD)--> (BS, c, vD, vH, vW)
        return x3d

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

project_layer = Project2Dto3D(240,144,240)
maxpool_layer = nn.MaxPool3d(kernel_size=(4,4,4), stride=(4,4,4))

for file in files:
    print(file)
    mapping = torch.from_numpy(np.load("E:/dataset/NYU/Mapping/"+file)['arr_0']).unsqueeze(0).long().cuda()
    tsdf = torch.from_numpy(np.load("E:/dataset/NYU/TSDF/"+file)['arr_0']).view(60,36,60)
    gt = torch.from_numpy(np.load("E:/dataset/NYU/Label/"+file)['arr_0']).view(60,36,60).cuda()
    label_weight = torch.from_numpy(np.load("E:/dataset/NYU/TSDF/"+file)['arr_1']).view(60,36,60).cuda()
    position = torch.from_numpy(np.load("E:/paper/SSC/SSC_points/dataset/NYUCAD/NYUCAD_npz/NYU"+file.split('.')[0]+"_0000_voxels.npz")['position']).cuda().long()
   
    points = []
    colors = []
    
    
    seg = torch.from_numpy(np.load("E:/paper/SSC/SSC_points/dataset/NYUCAD/seg_12/"+file.split('.')[0]+'.npy')).unsqueeze(0).cuda()
    _,seg = torch.max(seg, 1)
    seg = seg.unsqueeze(0)
    
    occupied = project_layer(seg, position).float().permute(0,1,4,3,2)
    seg_class = maxpool_layer(occupied).squeeze()
    
    

    points = torch.nonzero(torch.logical_and(label_weight, gt!=255))
    class_s  = seg_class[points[:,0:1],points[:,1:2],points[:,2:3]]
    
    colors = []
    for i in range(class_s.size(0)):
        class_s1 = int(class_s[i])
        colors.append(np.array([color_table[class_s1]])/255)
    
    colors = np.concatenate(colors, 0)
    points = points.cpu().numpy()

    pcd=o3d.open3d.geometry.PointCloud()
    pcd.points= o3d.open3d.utility.Vector3dVector(points)
    pcd.colors= o3d.open3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud("E:/paper/SSC/SSC_points/dataset/NYUCAD/Points/"+file.split('.')[0]+'.ply', pcd)

