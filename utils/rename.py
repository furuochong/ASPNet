# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 20:03:12 2022

@author: f6325
"""

import numpy as np
import os
path = "E:/paper/SSC/Segmentation/nyu_v2/nyu_v2/result/nyu13_deeplabv3plus_resnext101_shape/seg_12"

files = os.listdir(path)
for file in files:
    number = int(file.split('.')[0])
    aaa = ""
    for i in range(4-len(str(number))):
        aaa+='0'
    aaa+=str(number)
    old_name = os.path.join(path, file)
    new_name = os.path.join(path, aaa+'.npy')

    os.rename(old_name, new_name)