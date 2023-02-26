# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:08:27 2023

@author: f6325
"""

import os
import numpy as np

path = "E:/paper/SSC/SSC_points/dataset/NYUCAD/TSDF/"
label_path = "E:/dataset/NYU/Label/"
weight_path = "E:/paper/SSC/SSC_points/dataset/NYUCAD/Weight"

files = os.listdir(path)
for file in files:
    tsdf = np.reshape(np.load(os.path.join(path,file.split('.')[0]+'.npy')),(60,36,60))
    gt = np.reshape(np.load(os.path.join(label_path, file.split('.')[0]+'.npz'))['arr_0'],(60,36,60))
    
    gt_mask = np.logical_and(gt!=255, gt!=0)
    tsdf_mask = np.logical_and(tsdf<0, gt!=255)
    mask = np.logical_or(gt_mask, tsdf_mask)
    
    weight = np.ones((60,36,60))
    weight*=mask
    np.save(os.path.join(weight_path, file.split('.')[0]), weight)