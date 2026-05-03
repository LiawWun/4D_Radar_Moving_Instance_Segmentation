import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import pointnet2_utils as pointutils

class PointNetSetAbstraction(nn.Module):

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all = False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = in_channel+3

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        if group_all:
            self.queryandgroup = pointutils.GroupAll()
        else:
            self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()

        fps_idx = pointutils.furthest_point_sample(xyz_t, self.npoint)  # [B, N]
        new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, C, N]
        new_points = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points) # [B, 3+C, N, S]
        
        # new_xyz: sampled points position data, [B, C, N]
        # new_points: sampled points data, [B, 3+D, N, S]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0]

        return new_xyz, new_points

class PointNetSetUpConv(nn.Module):

    def __init__(self, nsample, radius, f1_channel, f2_channel, mlp, mlp2, knn = True):
        super(PointNetSetUpConv, self).__init__()
        self.nsample = nsample
        self.radius = radius
        self.knn = knn
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = f2_channel+3
        for out_channel in mlp:
            self.mlp1_convs.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
        if len(mlp) != 0:
            last_channel = mlp[-1] + f1_channel
        else:
            last_channel = last_channel + f1_channel
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm1d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
            Feature propagation from xyz2 (less points) to xyz1 (more points)

        Inputs:
            xyz1: (batch_size, 3, npoint1)
            xyz2: (batch_size, 3, npoint2)
            feat1: (batch_size, channel1, npoint1) features for xyz1 points (earlier layers, more points)
            feat2: (batch_size, channel1, npoint2) features for xyz2 points
        Output:
            feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B,C,N = pos1.shape

        _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)           
        
        pos2_grouped = pointutils.grouping_operation(pos2, idx)
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)    # [B,3,N1,S]

        feat2_grouped = pointutils.grouping_operation(feature2, idx)
        feat_new = torch.cat([feat2_grouped, pos_diff], dim = 1)   # [B,C1+3,N1,S]
        for conv in self.mlp1_convs:
            feat_new = conv(feat_new)
        # max pooling
        feat_new = feat_new.max(-1)[0]   # [B,mlp1[-1],N1]
        # concatenate feature in early layer
        if feature1 is not None:
            feat_new = torch.cat([feat_new, feature1], dim=1)
        # feat_new = feat_new.view(B,-1,N,1)
        for conv in self.mlp2_convs:
            feat_new = conv(feat_new)
        
        return feat_new

class PointNetFeaturePropogation(nn.Module):

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropogation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, C, N = pos1.shape
        
        dists,idx = pointutils.three_nn(pos1_t,pos2_t)
        dists[dists < 1e-10] = 1e-10
        weight = 1.0 / dists
        weight = weight / torch.sum(weight, -1,keepdim = True)   # [B,N,3]
        interpolated_feat = torch.sum(pointutils.grouping_operation(feature2, idx) * weight.view(B, 1, N, 3), dim = -1) # [B,C,N,3]

        if feature1 is not None:
            feat_new = torch.cat([interpolated_feat, feature1], 1)
        else:
            feat_new = interpolated_feat
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feat_new = F.relu(bn(conv(feat_new)))
        return feat_new


if __name__ == "__main__":

    # ======== 測試資料 ========
    device = torch.device('cuda:0')


    B, N1, N2, C = 2, 256, 512, 16
    pos1 = torch.randn(B, 3, N1)
    pos2 = torch.randn(B, 3, N2)
    feat1 = torch.randn(B, C, N1)
    feat2 = torch.randn(B, C, N2)

    pos1 = pos1.to(device)
    pos2 = pos2.to(device)
    feat1 = feat1.to(device)
    feat2 = feat2.to(device)