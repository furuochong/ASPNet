# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:26:23 2022

@author: f6325
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 21:27:44 2022

@author: f6325
"""

import torch
import numpy as np
import os
import cv2
import io
from io import BytesIO
import open3d as o3d
from utils.normalize import points_handle, rotz
import random


class NYUv2(torch.utils.data.Dataset):
    def __init__(self, input_path, mode):
        super(NYUv2, self).__init__()
        self.input_path = input_path
        self.mode = mode
        self.file_names = self.get_file_names(self.input_path, self.mode)
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])
        self.point_path = "E:/paper/SSC/SSC_points/dataset/NYUCAD/Points"
        self.seg_soft_path = "E:/paper/SSC/SSC_points/dataset/NYUCAD/seg_12"

    def get_file_names(self, input_path, mode):
        if mode == 'Train':
            source_file = os.path.join(input_path, 'train.txt')
        else:
            source_file = os.path.join(input_path, 'test.txt')
        file_names = []
        with open(source_file) as f:
            files = f.readlines()
        
        for file in files:
            file = file.strip()
            file_names.append(file)
        
        return file_names
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        idx = self.file_names[index]
        gt_path = os.path.join(self.input_path, 'Label')
        gt_path = os.path.join(gt_path, idx+'.npz')
        tsdf_path = os.path.join(self.input_path, 'TSDF')
        label_weight_path = os.path.join(tsdf_path, idx+'.npz')
        tsdf_path = os.path.join(tsdf_path, idx+'.npz')
        point_path = os.path.join(self.point_path, idx+'.ply')
        rgb_path = os.path.join(self.input_path, 'RGB')
        rgb_path = os.path.join(rgb_path, idx+'.png')
        seg_soft_path = os.path.join(self.seg_soft_path, idx+'.npy')
        map_path = os.path.join(self.input_path, 'Mapping')
        map_path = os.path.join(map_path, idx+'.npz')
        label_weight = np.load(label_weight_path)['arr_1'].astype(np.float32)
        
        
        pcd=o3d.io.read_point_cloud(point_path,format='ply')
        points=np.asarray(pcd.points).astype(np.int64)
        colors=np.asarray(pcd.colors).astype(np.float32)
        """
        tsdf: [129600, ] gt:[129600, ] seg: [480,640]
        """
        tsdf = np.load("E:/paper/SSC/SSC_points/dataset/NYUCAD/TSDF/"+idx+'.npy').astype(np.float32)
        gt = np.load(gt_path)['arr_0'].astype(np.int64)
        gt_voxel = np.load(gt_path)['arr_0'].astype(np.int64)
        seg_soft = np.load(seg_soft_path).astype(np.float32)
        seg_soft = torch.from_numpy(np.ascontiguousarray(seg_soft)).float()

        
        depth_mapping_3d = np.load(map_path)['arr_0'].astype(np.int64)
        depth_mapping_3d = torch.from_numpy(np.ascontiguousarray(depth_mapping_3d)).long().cuda()
        """
        points: [10240,3] points: [10240,5]
        """
        
        position = np.load("E:/paper/SSC/SSC_points/dataset/NYUCAD/NYUCAD_npz/NYU"+idx+"_0000_voxels.npz")['position']
        position = torch.from_numpy(position)

        points_input, points_surface, gt, points_2 = points_handle(points, colors, tsdf, gt, seg_soft, depth_mapping_3d, position,self.mode)
        

        """
        Load data
        """
        
    
        

        points_input = torch.from_numpy(np.ascontiguousarray(points_input)).float()
        points_surface = torch.from_numpy(np.ascontiguousarray(points_surface)).float()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
        gt_voxel = torch.from_numpy(np.ascontiguousarray(gt_voxel)).long()
        label_weight = torch.from_numpy(np.ascontiguousarray(label_weight)).float()

        
        

        output_dict = dict(gt = gt, points_input = points_input, points_surface = points_surface, points = points_2, gt_voxel = gt_voxel, \
                           label_weight = label_weight, name = idx)


        
        return output_dict




        

if __name__ == '__main__':
       path = 'E:/dataset/NYU'
       dataset = NYUv2(path, 'Train')
       dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
       for i, data in enumerate(dataloader):
          gt = data['gt']
          points_surface = data['points_surface']
          points_input = data['points_input']
          print(points_surface.size())
          print(points_input.size())


 
