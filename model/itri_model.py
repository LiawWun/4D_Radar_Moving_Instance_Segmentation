import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.flownet_util import *
from util.flow_embedding_util import *
from util.transformer_util import *

class FlowNet3D_GlobalAttention(nn.Module):

    def __init__(self):

        super(FlowNet3D_GlobalAttention,self).__init__()

        ### SPATIAL TEMPORAL ENCODER ###
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=1.0, nsample=16, in_channel=2, mlp=[16, 16, 32], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=2.0, nsample=8, in_channel=32, mlp=[32, 32, 64], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=128, radius=4.0, nsample=4, in_channel=64, mlp=[64,64,128], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=64,  radius=8.0, nsample=2, in_channel=128, mlp=[128,128,256], group_all=False)

        self.sa2_transformer_block = PointTransformerBlock(in_channel = 64, out_channel=64)
        self.sa3_transformer_block = PointTransformerBlock(in_channel = 128, out_channel=128)
        self.sa4_transformer_block = PointTransformerBlock(in_channel = 256, out_channel=256)

        self.fe_layer = FlowEmbeddingCrossAttention(nsample = 32, in_channel = 64, mlp = [64, 64, 64], attn_dim = 128)

        self.su1 = PointNetSetUpConv(nsample=8, radius=2.4, f1_channel = 128, f2_channel = 256, mlp=[], mlp2=[128, 128])
        self.su2 = PointNetSetUpConv(nsample=8, radius=1.2, f1_channel = 64 + 64, f2_channel = 128, mlp=[64, 64, 128], mlp2=[128])
        self.su3 = PointNetSetUpConv(nsample=8, radius=0.6, f1_channel = 32, f2_channel = 128, mlp=[64, 64, 128], mlp2=[128])
        self.fp = PointNetFeaturePropogation(in_channel = 128+2, mlp = [128, 128])

        self.su_dropout1 = nn.Dropout(p=0.3)
        self.su_dropout2 = nn.Dropout(p=0.3)
        self.su_dropout3 = nn.Dropout(p=0.3)

        ### HEAD ###
        self.semantic_classifier = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(64, 2, kernel_size=1, bias=True),
            nn.LogSoftmax(dim = 1)
        )

        self.offset_regressor = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 3, kernel_size=1, bias=False)
        )
    
    def forward(self, xyzs, points):
        
        ### CURRENT FRAME ###
        pc1 = xyzs[:, 0, :, :].contiguous()
        feat1 = points[:, 0, :, :].contiguous()

        pc2 = xyzs[:, 1, :, :].contiguous()
        feat2 = points[:, 1, :, :].contiguous()

        l1_pc1, l1_feat1 = self.sa1(pc1, feat1)
        l2_pc1, l2_feat1 = self.sa2(l1_pc1, l1_feat1)
        l2_pc1, l2_feat1 = self.sa2_transformer_block(l2_pc1, l2_feat1)

        l1_pc2, l1_feat2 = self.sa1(pc2, feat2)
        l2_pc2, l2_feat2 = self.sa2(l1_pc2, l1_feat2)
        l2_pc2, l2_feat2 = self.sa2_transformer_block(l2_pc2, l2_feat2)

        _, flow_feat = self.fe_layer(l2_pc1, l2_pc2, l2_feat1, l2_feat2)

        ## BACKBONE ###
        l3_pc1, l3_feat1 = self.sa3(l2_pc1, flow_feat)
        l3_pc1, l3_feat1 = self.sa3_transformer_block(l3_pc1, l3_feat1)

        l4_pc1, l4_feat1 = self.sa4(l3_pc1, l3_feat1)
        l4_pc1, l4_feat1 = self.sa4_transformer_block(l4_pc1, l4_feat1)

        ### DECODER ###
        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feat1, l4_feat1)
        l3_fnew1 = self.su_dropout1(l3_fnew1)

        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feat1, flow_feat], dim=1), l3_fnew1)
        l2_fnew1 = self.su_dropout2(l2_fnew1)

        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feat1, l2_fnew1)
        l1_fnew1 = self.su_dropout3(l1_fnew1)

        l0_fnew1 = self.fp(pc1, l1_pc1, feat1, l1_fnew1) 

        ### HEAD ###
        semantic_out = self.semantic_classifier(l0_fnew1)
        offset_out  = self.offset_regressor(l0_fnew1)

        semantic_out = semantic_out.permute(0, 2, 1)
        offset_out = offset_out.permute(0, 2, 1) 

        return semantic_out, offset_out
    
def initialize_weights(m):
    
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def count_parameters(model: nn.Module):
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
        
        
if __name__ == "__main__":

    from dataset.vod_dataset import VOD_Dataset

    B = 8
    frame_num = 2
    vod_data = VOD_Dataset(mode="valid", num_frame=frame_num)

    coord_list = []
    feat_list = []
    semantic_label_list = [] 

    for i in range(B):
        coord_3ds, feat_3ds, _, _ = vod_data.__getitem__(705 + i)
        coord_list.append(torch.tensor(coord_3ds))  # [T, 3, N]
        feat_list.append(torch.tensor(feat_3ds))    # [T, D, N]

    # Stack into batch: [B, T, 3, N] and [B, T, D, N]
    coord_3ds = torch.stack(coord_list, dim=0)
    feat_3ds = torch.stack(feat_list, dim=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    coord_3ds = coord_3ds.to(device)
    feat_3ds = feat_3ds.to(device)

    model = FlowNet3D_GlobalAttention().to(device)
    count_parameters(model)
    semantic_out, offset_out = model(coord_3ds, feat_3ds)
    print(semantic_out.shape)
    print(offset_out.shape)

    print(model)

    


