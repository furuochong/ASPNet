# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:53:28 2022

@author: f6325
"""

import numpy as np
import os
import cv2
colorMap = np.array([[0, 0, 0],    # 0 empty, free space
                     [40,  38, 214],    # 1 ceiling
                     [4, 160, 43],      # 2 floor
                     [229, 216, 158],   # 3 wall
                     [206, 158, 114],   # 4 window
                     [91, 204, 204],    # 5 chair  new: 180, 220, 90
                     [119, 186, 255],   # 6 bed
                     [188, 102, 147],   # 7 sofa
                     [181, 119, 30],    # 8 table
                     [33, 188, 188],    # 9 tvs
                     [12, 127, 255],    # 10 furn
                     [214, 175, 196],   # 11 objects
                     [153, 153, 153],     # 12 Accessible area, or label==255, ignore
                     ]).astype(np.int32)


corrrespond_table = [6,11,1,5,2,10,11,11,9,7,8,3,4]

path = "E:/paper/SSC/Segmentation/nyu_v2/nyu_v2/result/nyu13_deeplabv3plus_resnext101_shape/seg_12"
path_label = "E:/paper/SSC/Segmentation/nyu_v2/nyu_v2/result/nyu13_deeplabv3plus_resnext101_shape/seg_label"
files = os.listdir(path)


max_number = 0
min_number = 100

for file in files:
    label_name = os.path.join(path_label, file)
    name = os.path.join(path, file)
    seg_label = np.load(label_name)
    seg = np.squeeze(np.load(name))   
    seg_temp = np.zeros((12,480,640))
    
    for i in range(seg_temp.shape[1]):
        for j in range(seg_temp.shape[2]):
            label_value = int(seg_label[i][j])

            for k in range(13):
                seg_temp[corrrespond_table[k]][i][j]+=seg[k][i][j]
                
    np.save(name, seg_temp)