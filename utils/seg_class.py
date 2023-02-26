# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 00:32:32 2022

@author: f6325
"""

"""
NYUv2-13
void bed books ceiling chair floor furniture objects picture sofa table tv wall window
      0   1     2       3     4       5        6       7       8    9   10  11   12 
NYUv2-SSC
void ceil floor wall window chair bed table tvs sofa furn object
  0   1     2    3    4       5    6    7    8    9   10   11

0-->6 1-->11 2-->1 3-->5 4-->2 5-->10 6--> 11 7--> 11 8--> 9 9-->7 10-->8 11-->3 12-->4
      
      
今天任务：数据增强， 类别对应


问题：
train: 795 mIoU: 0.9
test:  654 mIoU  0.656

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


path = "E:/paper/SSC/Segmentation/nyu_v2/nyu_v2/result/nyu13_deeplabv3plus_resnext101_shape/seg"
path_label = "E:/paper/SSC/Segmentation/nyu_v2/nyu_v2/result/nyu13_deeplabv3plus_resnext101_shape/seg_label"
files = os.listdir(path)


max_number = 0
min_number = 100


for file in files:
    label_name = os.path.join(path_label, file)
    name = os.path.join(path, file)
    seg_label = np.load(label_name)
    seg = np.load(name)
    seg_temp = np.zeros((480,640))
    

    for i in range(seg_temp.shape[0]):
        for j in range(seg_temp.shape[1]):
            label_value = int(seg_label[i][j])
            value =  int(seg[i][j]) 

            
            if label_value==255:
                class_num=0
                seg_temp[i][j] = class_num
            else:
                class_num = corrrespond_table[int(value)]
                seg_temp[i][j] = class_num

    
    np.save(name, seg_temp)

        
    