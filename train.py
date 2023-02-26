# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 03:20:02 2022

@author: f6325
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset import NYUv2   
from torch.utils.tensorboard import SummaryWriter
from network import Network
from utils.metrics import MIoU
import tqdm
from utils import normalize
import torch.nn.functional as F


def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)  # exclude 255
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))

    return np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                          minlength=n_cl ** 2).reshape(n_cl,
                                                       n_cl), correct, labeled

seed = 12345
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

path = 'E:/dataset/NYU'
dataset = NYUv2(path, 'Train')
dataset_test = NYUv2(path, 'Test')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 1, shuffle = False)

writer = SummaryWriter()
model = Network()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

weights = [1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2]
class_weights = torch.FloatTensor(weights).cuda()
citation =  nn.CrossEntropyLoss(weight=class_weights)
citation_2 = nn.BCELoss()

max_iou = 0
model.cuda()



for epoch in range(0, 800):

    sum_loss = 0
    sum_loss_test = 0
    m_miou_test = MIoU(num_classes=12, ignore_class=0)    
    max_miou = 0
    model.train()

    for idx, data in enumerate(dataloader):
        optimizer.zero_grad()
 
        points_input = data['points_input'].cuda()
        points_surface = data['points_surface'].cuda()
        gt = data['gt'].cuda()


        end_points = {}
        output,end_results = model(points_input, points_surface, gt.clone(), end_points)
        loss_2 = citation_2(end_points['fp3_class_map'].float(), end_points['fp3_gt_mask'].float())
        loss_3 = citation_2(end_points['fp2_class_map'].float(), end_points['fp2_gt_mask'].float())
        loss_4 = citation_2(end_points['fp1_class_map'].float(), end_points['fp1_gt_mask'].float())
        loss = citation(output, gt.squeeze())
        loss = loss+loss_2+loss_3+loss_4
        loss.backward()
        optimizer.step()
        

        sum_loss+=loss.item()
        
        print("epoch: "+str(epoch)+"  train_loss:"+str(loss.item()))
        

    
    model.eval()
    for idx, data in enumerate(dataloader_test):

        points_input = data['points_input'].cuda()
        points_surface = data['points_surface'].cuda()
        gt = data['gt'].cuda()
        gt_voxel = data['gt_voxel'].cuda()
        points = data['points'].cuda().squeeze()
        label_weight = data['label_weight'].cuda()
                
        
        end_points = {}
        output,_ = model(points_input, points_surface, gt.clone(), end_points)
        m_miou_test.update(output, gt)


    epoch_miou_test = m_miou_test.compute()
    epoch_per_class_iou_test = m_miou_test.per_class_iou()
            

    max_iou = max(max_iou,float(epoch_miou_test))
    

    if  max_iou==float(epoch_miou_test):
        torch.save(model, "./checkpoints/surface_attention_spt_3.pth")
    
    
    filename = './test.txt'
    with open(filename,'a') as file_object:
        file_object.write("MIOU in DA_paper test: ")
        file_object.write(str(epoch_miou_test))
        file_object.write("\n")
        file_object.write(str(epoch_per_class_iou_test))
        file_object.write("\n")

    
    
    
    

    
    


    
    
        
    

    
