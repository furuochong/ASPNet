# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:02:58 2022

@author: f6325
"""

import numpy as np
import torch
import open3d as o3d
import math
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


def pc_normalize(pc):
    height = pc[:,1:2]-18 #[-18,+18]
    height = height/18
    
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  
    pc = pc / m

    return pc, centroid, height

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def pc_utils(points, tsdf):
    tsdf = np.expand_dims(tsdf[points[:,0],points[:,1],points[:,2]], axis=1)
    points, centroid, height = pc_normalize(points)
    points = np.concatenate((points,tsdf,height), axis=1)
    return points


def points_handle(points, colors, tsdf, gt, seg_soft, mapping, position, mode):
    tsdf = np.reshape(tsdf, (60,36,60))
    gt = np.reshape(gt, (60,36,60))


    seg_soft = seg_soft.unsqueeze(0).cuda()
    mapping = mapping.unsqueeze(0).long().cuda()

    """
    b, c, h, w = seg_soft.shape
    seg_soft = seg_soft.view(b, c, h * w).permute(0, 2, 1)  # b x h*w x c

    zerosVec = torch.zeros(b, 1, c).cuda()  # for voxels that could not be projected from the depth map, we assign them 255 vector
    segVec = torch.cat((seg_soft, zerosVec), 1)

    segres = [torch.index_select(segVec[i], 0, mapping[i]) for i in range(b)]
    segres = torch.stack(segres).permute(0, 2, 1).contiguous().view(b,c,60, 36, 60)  # B, (channel), 60, 36, 60
    segres = segres.cpu().numpy()
    segres = np.squeeze(segres)
    """
    
    position = position.cuda().long()
    project_layer = Project2Dto3D(240,144,240)
    maxpool_layer = nn.MaxPool3d(kernel_size=(4,4,4), stride=(4,4,4))
    segres = project_layer(seg_soft, position).float().permute(0,1,4,3,2)
    segres = maxpool_layer(segres).squeeze()
    segres = segres.cpu().numpy()
    segres = np.squeeze(segres)
    
    colors = np.sum(colors, axis=1)
    
    
    index = np.argwhere(colors == 0).flatten()
    index_surface = np.argwhere(colors != 0).flatten()
    
    
    points_inside = points[index]
    points_surface = points[index_surface]
    

    inside_number = points_inside.shape[0]
    surface_number = points_surface.shape[0]

    if mode =='Train':
        inside_number = 8192
        surface_number = 2048
        
        if points_inside.shape[0]>8192:
            choices = np.random.choice(points_inside.shape[0], 8192, replace=False)
            points_inside = points_inside[choices]


        else:
            choices = np.random.choice(points_inside.shape[0], 8192, replace=True)
            points_inside = points_inside[choices]
       
            
        if points_surface.shape[0]>2048:
            choices = np.random.choice(points_surface.shape[0], 2048, replace=False)
            points_surface = points_surface[choices]
            
        else:
            choices = np.random.choice(points_surface.shape[0], 2048, replace=True)
            points_surface = points_surface[choices]


    
    
    tsdf_inside = np.expand_dims(tsdf[points_inside[:,0],points_inside[:,1],points_inside[:,2]], axis=1)
    tsdf_surface = np.expand_dims(tsdf[points_surface[:,0],points_surface[:,1],points_surface[:,2]], axis=1)
    

    gt_inside = np.expand_dims(gt[points_inside[:,0],points_inside[:,1],points_inside[:,2]], axis=1)
    gt_surface = np.expand_dims(gt[points_surface[:,0],points_surface[:,1],points_surface[:,2]], axis=1)
    
    seg_soft_inside = segres[:,points_inside[:,0],points_inside[:,1],points_inside[:,2]].T
    seg_soft_surface = segres[:,points_surface[:,0],points_surface[:,1],points_surface[:,2]].T

    points = np.concatenate((points_surface, points_inside), axis=0) 
    
    points_2 = points
    
    tsdf = np.concatenate((tsdf_surface, tsdf_inside), axis=0) 
    gt = np.concatenate((gt_surface, gt_inside), axis=0) 
    seg_soft = np.concatenate((seg_soft_surface, seg_soft_inside), axis=0) 
      
    points_input, centroid, height = pc_normalize(points)
    height_surface = height[:surface_number,:]

    points_input = np.concatenate((points_input, tsdf, height, seg_soft), axis=1)
    points_surface = np.concatenate((points_surface, tsdf_surface, height_surface, seg_soft_surface), axis=1)
    

    
    return  points_input,points_surface,gt, points_2






