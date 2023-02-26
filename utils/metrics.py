# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:23:28 2022

@author: f6325
"""

import torch
from torch import Tensor
import numpy as np

"""
先看懂
"""

class MIoU:
    def __init__(self, num_classes=12, ignore_class=None):

        if ignore_class is None:
            self.intersection = np.zeros(num_classes)
            self.union = np.zeros(num_classes)+ 1e-10
        else:
            self.intersection = np.zeros(num_classes-1)
            self.union = np.zeros(num_classes-1)+ 1e-10


        self.num_classes = num_classes
        self.ignore_class = ignore_class

    def update(self, pred: Tensor, target: Tensor, weights: Tensor = None, vol: Tensor = None):

        pred = pred.view(pred.size(0), pred.size(1), -1)  # N,C,H,W => N,C,H*W
        pred = pred.transpose(1, 2)  # N,C,H*W => N,H*W,C
        _, pred = torch.max(pred, 2) # N,H*W,C => N,H*W

        target = target.view(target.size(0), -1)  # N,H,W => N,H*W
        
        if self.ignore_class is None:
            idx = target < self.num_classes
        else:
            idx = (target < self.num_classes) & (target != self.ignore_class)

        if weights is not None:
            weights = weights.view(target.size(0), -1)  # N,H,W => N,H*W
            idx = idx & (weights != 0.0)

        if vol is not None:
            vol = vol.view(target.size(0), -1)  # N,H,W => N,H*W
            idx = idx & (torch.logical_or(torch.abs(vol)<1,vol==-1.0))

        _target = target[idx]
        _pred = pred[idx]
        
        c = -1
        for i in range(self.num_classes):
            if i != self.ignore_class:
                c += 1
                inter = torch.sum(_pred[_pred==i] == _target[_pred==i])
                union = _pred[_pred==i].size(0) + _target[_target==i].size(0) - inter
                self.intersection[c] += inter
                self.union[c] += union

    def compute(self):
        iou = self.intersection/self.union
        return np.mean(iou)

    def per_class_iou(self):
        iou = self.intersection/self.union
        return iou
    
    
class CompletionIoU:
    def __init__(self):

        self.intersection = 0.0
        self.union = 1e-10
        self.tp = 0.0
        self.fp = 0.0
        self.tn = 0.0
        self.fn = 0.0

    def update(self, pred: Tensor, target: Tensor, weights: Tensor = None, vol: Tensor = None):

        pred = pred.view(pred.size(0), pred.size(1), -1)  # N,C,H,W => N,C,H*W
        pred = pred.transpose(1, 2)  # N,C,H*W => N,H*W,C
        _, pred = torch.max(pred, 2) # N,H*W,C => N,H*W

        target = target.view(target.size(0), -1)  # N,H,W => N,H*W

        if weights is not None:
            weights = weights.view(target.size(0), -1)  # N,H,W => N,H*W
            idx = torch.logical_and(weights != 0.0, weights != 0.0)  # Only occluded
            idx = (idx & (target != 255)) #Only occluded

        if vol is not None:
            vol = vol.view(target.size(0), -1)  # N,H,W => N,H*W
            idx = torch.logical_and(vol < 0, vol >= -1.0)
        
        

        pred = pred[idx] #>0
        target = target[idx]

        inter = torch.sum(torch.logical_and(pred>0, target>0))
        union = torch.sum(torch.logical_or(pred>0,target>0))

        tp = inter
        fp = torch.sum(torch.logical_and(target == 0, pred>0))
        tn = torch.sum(torch.logical_and(target == 0, pred==0))
        fn = torch.sum(torch.logical_and(target> 0, pred==0))

        self.intersection += inter
        self.union += union
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn



    def compute(self):
        comp_iou = self.intersection/self.union

        if (self.tp + self.fp) > 0:
            precision = self.tp / (self.tp + self.fp)
        else:
            precision = 0

        if (self.tp + self.fn) > 0:
            recall = self.tp / (self.tp + self.fn)
        else:
            recall = 0

        return comp_iou, precision, recall


