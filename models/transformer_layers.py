# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:32:58 2022

@author: f6325
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
        
        
    pytorch none 增加维度
    src [b,n,c] src[:,:,none] [b,n,1,3]
    dst [b,n,c] dst[:,none]   [b,1,n,3]
    """

    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def square_distance_cos(src, dst):
    src = src / torch.norm(src, dim = -1, keepdim=True)
    dst = dst / torch.norm(dst, dim = -1, keepdim=True)
    
    dst = dst.permute(0,2,1).contiguous()
    
    #src [b,n,128] dst [b,n,128]
    
    return torch.matmul(src, dst)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
        
        
    torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor
    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    
        
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz) #[b,n,n]
        knn_idx = dists.argsort()[:, :, :self.k]  #[b,n,k] 离得最近的前k个

        knn_xyz = index_points(xyz, knn_idx) #[b, n, k, 3]

        pre = features # [b,n,f]
        x = self.fc1(features) #[b, n, d_model]
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
              
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)       
        # attn [b, n, k, d_model]
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        
        # res  [b, n, c] back to the origin position
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre

        
        return res, attn

class TransformerBlock_surface(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        """
        surface transformer block:
            inputs: xyz, features [b, n1, 3] [b, n1, c]
                    xyz_surface, xyz_features [b, n2, 3] [b, n2, c]
                    
        """
        
        
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc1_surface = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        

        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

        
        
    def forward(self, xyz, features, xyz_surface, features_surface):
        
        dists = square_distance(xyz, xyz_surface) #位置上最相近的点

        knn_idx = dists.argsort()[:, :, :self.k]  #[b,n1,k] 离得最近的前k个

        
        knn_xyz_surface = index_points(xyz_surface, knn_idx) #[b, n1, k, 3], 每个点距离最近的k个表面点的坐标


        pre = features
        q = self.fc1(features) #[b, n1, d_model]
        
        x_surface = self.fc1_surface(features_surface)# [b,n2, d_model]
        
        
        w_ks = self.w_ks(x_surface)
        w_vs = self.w_vs(x_surface)
        
        
        k = index_points(w_ks, knn_idx) #[b, n1, k, d_model]
        v = index_points(w_vs, knn_idx)
        
        
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz_surface) #[b, n, k, d_model
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)     #[b, n, k, d_model]
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # [b, n, k, d_model]
        res = torch.einsum('bmnf,bmnf->bmf', attn, v+pos_enc) #[b, n, d_model]
        res = pre+ self.fc2(res)
        
        
        
        return res, attn
        

class TransformerBlock_masked(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features, gt):
        

        dists = square_distance(xyz, xyz) #[b,n,n]
        knn_idx = dists.argsort()[:, :, :self.k]  #[b,n,k] 离得最近的前k个
        
        knn_xyz = index_points(xyz, knn_idx) #[b, n, k, 3]
        

        pre = features # [b,n,f]
        x = self.fc1(features) #[b, n, d_model]
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
              
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc) 

        
        class_map = self.fc_score(q[:, :, None] - k + pos_enc).squeeze(-1) #[b,n,k]
        gt_index = index_points(gt, knn_idx).squeeze()  # [b,n,k]
        

        
        gt_template = gt.repeat(1,1,self.k)
        gt_mask = (gt_index==gt_template).long()

        attn_diff = attn
        attn = attn*(class_map.unsqueeze(-1))
        

        # attn [b, n, k, d_model]
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        
        # res  [b, n, c] back to the origin position
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)

        
        
        res = self.fc2(res)+pre
        
        return res, gt_mask, class_map

class Adaptive_FP(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.k = k
        self.fc1 = nn.Linear(d_points, d_model)
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, xyz_fp, features, features_fp):
        
        # xyz 512 xyz_fp 256
        
        dists = square_distance(xyz_fp, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  #[b,n,k] 离得最近的前k个
        knn_xyz = index_points(xyz, knn_idx) #[b, n, k, 3]
        
        features = features.permute(0,2,1).contiguous()
        features_fp = features_fp.permute(0,2,1).contiguous()
        
        x = self.fc1(features)
        print(x.size())
        
        """
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)
        """

        
        return features
    
    
if __name__ == '__main__':
    layer = TransformerBlock_masked(256,64,16).cuda()
    points = torch.rand((1,2048,3)).cuda()
    features = torch.rand((1,2048,256)).cuda()
    
    res = layer(points, features)
    
    
