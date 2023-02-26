# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 23:48:24 2022

@author: f6325
"""


import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import open3d as o3d

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset import NYUv2   
from network import Network
from utils.metrics import MIoU, CompletionIoU
import tqdm
from utils import normalize
import torch.nn.functional as F
from thop import profile


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




seed = 12345
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

path = 'E:/dataset/NYU'
dataset_test = NYUv2(path, 'Test')


dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 1, shuffle = False)

model = Network()
model = torch.load("./checkpoints/surface_attention_spt_3.pth")
model.cuda()


model.eval()
miou = MIoU(num_classes=12, ignore_class=0)
ciou = CompletionIoU()

macss = 0

for idx, data in enumerate(dataloader_test):

    points_input = data['points_input'].cuda()
    points_surface = data['points_surface'].cuda()
    gt = data['gt'].cuda()
    points = data['points'].cuda().squeeze().detach().cpu().numpy()

    label = data['gt_voxel']
    label_weight = data['label_weight']
    name = data['name'] 
    

    end_points = {}
    output,_ = model(points_input, points_surface, gt.clone(), end_points)
    print(output.size())
    


    output = output.squeeze().detach().cpu().numpy()
    result = np.zeros((12,60,36,60))
    result[:,points[:,0],points[:,1],points[:,2]] = output
    result = torch.from_numpy(result)
    
    miou.update(result.unsqueeze(0).cuda(), label.cuda() , label_weight.cuda())
    ciou.update(result.unsqueeze(0).cuda(), label.cuda() , label_weight.cuda())
    


epoch_miou_test = miou.compute()
epoch_per_class_iou_test = miou.per_class_iou()
comp_iou, precision, recall = ciou.compute()

print("prec rec. IoU  MIou")
print("{:4.1f} {:4.1f} {:4.1f} {:4.1f}".format(100 * precision, 100 * recall, 100 * comp_iou,
                                               miou.compute() * 100))

print(epoch_miou_test)
print(epoch_per_class_iou_test)



