# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:06:08 2022

@author: f6325
"""


import numpy as np
import open3d as o3d
import os




source_file = "E:/dataset/NYU/train.txt"
file_names = []
with open(source_file) as f:
    files = f.readlines()
        
    for file in files:
        file = file.strip()
        file_names.append(file)
        
path = "E:/paper/SSC/Segmentation/nyu_v2/nyu_v2/result/nyu13_deeplabv3plus_resnext101_shape/seg"
files = os.listdir(path)

for i in range(795):
    name_new = os.path.join(path, file_names[i]+'.npy')
    name_old = os.path.join(path, str(i+1)+'.npy')
    
    print(name_new)
    print(name_old)
    
    os.rename(name_old, name_new)