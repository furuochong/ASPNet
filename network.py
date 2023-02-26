# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 03:20:08 2022

@author: f6325
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

ROOT_DIR = "../SSC_points"
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from models.transformer_layers import TransformerBlock, TransformerBlock_surface, TransformerBlock_masked, Adaptive_FP
import pointnet2_utils

class Network(nn.Module):
    """
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    
    def __init__(self, final_class = 12, input_feature_dim=14):
        super().__init__()
        self.semantic_class = final_class
        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        self.sa1_surface = PointnetSAModuleVotes(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )
        """
        feature extraction for surface point
        """

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )
        
        self.sa2_surface = PointnetSAModuleVotes(
                npoint=256,
                radius=0.4,
                nsample=32,
                mlp=[128, 256, 256, 128],
                use_xyz=True,
                normalize_xyz=True
            )
        
        """
        feature extraction for surface point
        """
        
        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        """
        feature extraction for surface point
        """
        
        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )
        
            
        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp3 = PointnetFPModule(mlp=[128+256,256,256])
        self.fp4 = PointnetFPModule(mlp=[14+256,256,256])
        

        self.dim_layer = nn.Conv1d(256, self.semantic_class, 1, bias=True)
        
        # __init__(self, d_points, d_model, k) -> None:
        """
        inputs: xyz      features
                [b,n,3]  [b,n,c]
        """
        # 对于空间中的每一个点，找到与其空间距离最相近的个点， 以及特征空间下距离最近的点

        self.atten1_surface = TransformerBlock_surface(128, 64, 32)
        self.atten1 = TransformerBlock(128, 64, 32)
        self.atten2 = TransformerBlock(256, 64, 16)
        self.atten3 = TransformerBlock(256, 64, 16)
        self.atten4 = TransformerBlock(256, 64, 16)
        self.atten5 = TransformerBlock_masked(256, 64, 16)
        self.atten6 = TransformerBlock_masked(256, 64, 16)# masked atten
        self.atten7 = TransformerBlock_masked(256, 64, 32)



        
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features
    def forward(self, pointcloud, points_surface, gt , end_points):
        
        """
        pointcloud: [b, 1536, 17]
        pointcloud_surface: [b, 1536+8192, 17]
        """
        gt = gt.float().permute(0,2,1).contiguous()
        
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        xyz_surface, features_surface = self._break_up_pc(points_surface)
        
        
        xyz_surface, features_surface, _ = self.sa1_surface(xyz_surface, features_surface)
        end_points['sa1_xyz_surface'] = xyz_surface #[b,2048,3]
        end_points['sa1_features_surface'] = features_surface #[b,128,2048]
        xyz_surface, features_surface, _ = self.sa2_surface(xyz_surface, features_surface)
        end_points['sa2_xyz_surface'] = xyz_surface #[b,2048,3]
        end_points['sa2_features_surface'] = features_surface #[b,128,2048]
        
        
        
        end_points['sa0_xyz'] = xyz 
        end_points['sa0_features'] = features 
        

        #xyz : [b,n,3] features:[b,3,n]
        xyz, features, fps_inds = self.sa1(xyz, features)
        
        features_surface = features_surface.permute(0,2,1).contiguous()
        features = features.permute(0,2,1).contiguous()
        features,_ = self.atten1_surface(xyz, features, xyz_surface, features_surface)
        features = features.permute(0,2,1).contiguous()
        features_surface = features_surface.permute(0,2,1).contiguous()
        
        features = features.permute(0,2,1).contiguous()
        features,_ = self.atten1(xyz, features)
        features = features.permute(0,2,1).contiguous()
        
    
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz #[b,2048,3]
        end_points['sa1_features'] = features #[b,128,2048]
        
        gt_sa1 = pointnet2_utils.gather_operation(
            gt, fps_inds
        ).transpose(1, 2).contiguous()
        end_points['sa1_gts'] = gt_sa1
        
        
        
        
        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023     
        features = features.permute(0,2,1).contiguous()
        features,_ = self.atten2(xyz, features)
        features = features.permute(0,2,1).contiguous()
        
        end_points['sa2_inds'] = fps_inds #[b,512]
        end_points['sa2_xyz'] = xyz #[b,512,3]
        end_points['sa2_features'] = features #[b,256,512]
        
        gt_sa2 = pointnet2_utils.gather_operation(
            gt, fps_inds
        ).transpose(1, 2).contiguous()
        end_points['sa2_gts'] = gt_sa2
  
        
                
        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        features = features.permute(0,2,1).contiguous()
        features,_ = self.atten3(xyz, features)
        features = features.permute(0,2,1).contiguous()


        end_points['sa3_inds'] = fps_inds #[b,512]
        end_points['sa3_xyz'] = xyz #[b,512,3]
        end_points['sa3_features'] = features #[b,256,512]
        
        gt_sa3 = pointnet2_utils.gather_operation(
            gt, fps_inds
        ).transpose(1, 2).contiguous()
        end_points['sa3_gts'] = gt_sa3


    
        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        features = features.permute(0,2,1).contiguous()
        features,_ = self.atten4(xyz, features)
        features = features.permute(0,2,1).contiguous()

        
        end_points['sa4_inds'] = fps_inds #[b,256]
        end_points['sa4_xyz'] = xyz #[b,256,3]
        end_points['sa4_features'] = features #[b,256,256]

        
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features']) #[b,256,512]
        end_points['fp1_features'] = features
        
        features = features.permute(0,2,1).contiguous()
        features,gt_mask, class_map = self.atten5(end_points['sa3_xyz'], features, end_points['sa3_gts'])
        features = features.permute(0,2,1).contiguous()
        
        end_points['fp1_class_map'] = class_map #[b,512]
        end_points['fp1_gt_mask'] = gt_mask #[b,512,3]
        
        
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)  #[b,256,1024]
        end_points['fp2_features'] = features
        
        features = features.permute(0,2,1).contiguous()
        features,gt_mask, class_map = self.atten6(end_points['sa2_xyz'], features, end_points['sa2_gts'])
        features = features.permute(0,2,1).contiguous()
       
        end_points['fp2_class_map'] = class_map #[b,512]
        end_points['fp2_gt_mask'] = gt_mask #[b,512,3]

 
        features = self.fp3(end_points['sa1_xyz'], end_points['sa2_xyz'], end_points['sa1_features'], features)  #[b,256,2048]
        end_points['fp3_features'] = features
        
        features = features.permute(0,2,1).contiguous()
        features,gt_mask, class_map = self.atten7(end_points['sa1_xyz'], features, end_points['sa1_gts'])
        features = features.permute(0,2,1).contiguous()
        end_points['fp3_class_map'] = class_map #[b,512]
        end_points['fp3_gt_mask'] = gt_mask #[b,512,3]

        features = self.fp4(end_points['sa0_xyz'], end_points['sa1_xyz'], end_points['sa0_features'], features)  #[b,256,1536+8192]
        features = self.dim_layer(features)


        
        return features, end_points



 

 
