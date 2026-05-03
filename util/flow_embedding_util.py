import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import pointnet2_utils as pointutils

class FlowEmbedding(nn.Module):
    
    def __init__(self, radius, nsample, in_channel, mlp, pooling='max', corr_func='concat', knn = True):
        super(FlowEmbedding, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.pooling = pooling
        self.corr_func = corr_func
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if corr_func == 'concat':
            last_channel = in_channel*2+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
        Input:
            xyz1: (batch_size, 3, npoint)
            xyz2: (batch_size, 3, npoint)
            feat1: (batch_size, channel, npoint)
            feat2: (batch_size, channel, npoint)
        Output:
            xyz1: (batch_size, 3, npoint)
            feat1_new: (batch_size, mlp[-1], npoint)
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, N, C = pos1_t.shape
        _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)
        
        pos2_grouped = pointutils.grouping_operation(pos2, idx) # [B, 3, N, S]
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)    # [B, 3, N, S]
        
        feat2_grouped = pointutils.grouping_operation(feature2, idx)    # [B, C, N, S]
        if self.corr_func=='concat':
            feat_diff = torch.cat([feat2_grouped, feature1.view(B, -1, N, 1).repeat(1, 1, 1, self.nsample)], dim = 1)
        
        feat1_new = torch.cat([pos_diff, feat_diff], dim = 1)  # [B, 2*C+3,N,S]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feat1_new = F.relu(bn(conv(feat1_new)))

        feat1_new = torch.max(feat1_new, -1)[0]  # [B, mlp[-1], npoint]

        return pos1, feat1_new

class FlowEmbeddingCrossAttention(nn.Module):
    """
    A module that combines Flow Embedding with single-head Cross-Attention.
    It uses a combined max and mean pooling strategy to create a rich,
    motion-aware feature vector to be used as the attention query.
    """
    def __init__(self, nsample, in_channel, mlp, attn_dim=128):
        super(FlowEmbeddingCrossAttention, self).__init__()

        # --- Parameters ---
        self.nsample = nsample
        self.attn_dim = attn_dim

        # --- FlowEmbedding MLP ---
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        # Input to the MLP is: pos_diff (3) + feat2 (in_channel) + feat1 (in_channel)
        last_channel = in_channel * 2 + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # --- Single-Head Cross-Attention Projections ---
        # The input dimension is doubled (mlp[-1] * 2) from combined pooling.
        pooled_dim = mlp[-1] * 2
        self.q_linear = nn.Linear(pooled_dim, attn_dim)
        self.k_linear = nn.Linear(in_channel, attn_dim)
        self.v_linear = nn.Linear(in_channel, attn_dim)

        # Projection layer to map attention output to the residual dimension.
        self.attn_proj = nn.Linear(attn_dim, pooled_dim)

        # --- Residual Connection & Normalization ---
        self.layernorm = nn.LayerNorm(pooled_dim)
        self.final_proj = nn.Linear(pooled_dim, mlp[-1])

    def forward(self, pos1, pos2, feature1, feature2):

        B, N = pos1.shape[0], pos1.shape[2]
        _, M = pos2.shape[0], pos2.shape[2]

        # === 1. FlowEmbedding Feature Extraction ===
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()

        _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)
        pos_diff = pointutils.grouping_operation(pos2, idx) - pos1.view(B, -1, N, 1)
        feat2_grouped = pointutils.grouping_operation(feature2, idx)

        feat_corr = torch.cat([
            feat2_grouped,
            feature1.view(B, -1, N, 1).repeat(1, 1, 1, self.nsample)
        ], dim=1)

        feat_embedding = torch.cat([pos_diff, feat_corr], dim=1)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            feat_embedding = F.relu(bn(conv(feat_embedding)))

        # === 2. Combined Max & Mean Pooling ===
        feat_max = torch.max(feat_embedding, -1)[0]
        feat_mean = torch.mean(feat_embedding, dim=-1)
        feat_pooled = torch.cat([feat_max, feat_mean], dim=1) # Shape: [B, mlp[-1] * 2, N]
        feat_out = feat_pooled.permute(0, 2, 1)               # Shape: [B, N, mlp[-1] * 2]

        # === 3. Single-Head Cross-Attention ===
        Q = self.q_linear(feat_out)                  # Shape: [B, N, attn_dim]
        K = self.k_linear(feature2.permute(0, 2, 1)) # Shape: [B, M, attn_dim]
        V = self.v_linear(feature2.permute(0, 2, 1)) # Shape: [B, M, attn_dim]

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attn_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # Shape: [B, N, M]
        attn_out = torch.matmul(attn_weights, V)          # Shape: [B, N, attn_dim]

        # Project attention output to match the residual dimension
        attn_out_projected = self.attn_proj(attn_out)     # Shape: [B, N, mlp[-1] * 2]

        # === 4. Residual Connection & Final Output ===
        final_feat = feat_out + attn_out_projected
        final_feat = self.layernorm(final_feat)
        final_feat = self.final_proj(final_feat)
        final_feat = final_feat.permute(0, 2, 1).contiguous()

        return pos1, final_feat
    
class FlowEmbeddingMultiHeadCrossAttention(nn.Module):

    def __init__(self, nsample, in_channel, mlp, attn_dim=256, num_heads=4):
        super(FlowEmbeddingMultiHeadCrossAttention, self).__init__()

        # --- Parameters ---
        self.nsample = nsample
        assert attn_dim % num_heads == 0, f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})"
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        # --- FlowEmbedding MLP ---
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        # Input to the MLP is: pos_diff (3) + feat2 (in_channel) + feat1 (in_channel)
        last_channel = in_channel * 2 + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # --- Multi-Head Cross-Attention Projections ---
        # The input dimension is doubled (mlp[-1] * 2) because we concatenate max and mean pooling results.
        pooled_dim = mlp[-1] * 2
        self.q_linear = nn.Linear(pooled_dim, attn_dim)
        self.k_linear = nn.Linear(in_channel, attn_dim)
        self.v_linear = nn.Linear(in_channel, attn_dim)

        # Final projection layer to map the concatenated head outputs back to the residual dimension.
        self.fc_out = nn.Linear(attn_dim, pooled_dim)

        # --- Residual Connection & Normalization ---
        self.layernorm = nn.LayerNorm(pooled_dim)
        self.final_proj = nn.Linear(pooled_dim, mlp[-1])

    def forward(self, pos1, pos2, feature1, feature2):

        B, N = pos1.shape[0], pos1.shape[2]
        _, M = pos2.shape[0], pos2.shape[2]

        # === 1. FlowEmbedding Feature Extraction ===
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()

        _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)
        pos_diff = pointutils.grouping_operation(pos2, idx) - pos1.view(B, -1, N, 1)
        feat2_grouped = pointutils.grouping_operation(feature2, idx)

        feat_corr = torch.cat([
            feat2_grouped,
            feature1.view(B, -1, N, 1).repeat(1, 1, 1, self.nsample)
        ], dim=1)
        feat_embedding = torch.cat([pos_diff, feat_corr], dim=1)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            feat_embedding = F.relu(bn(conv(feat_embedding)))

        # === 2. Combined Max & Mean Pooling ===
        feat_max = torch.max(feat_embedding, -1)[0]   # [B, mlp[-1], N]
        feat_mean = torch.mean(feat_embedding, dim=-1) # [B, mlp[-1], N]

        # Concatenate along the channel dimension for a richer feature vector.
        feat_pooled = torch.cat([feat_max, feat_mean], dim=1) # Shape: [B, mlp[-1] * 2, N]
        feat_out = feat_pooled.permute(0, 2, 1)               # Shape: [B, N, mlp[-1] * 2]

        # === 3. Multi-Head Cross-Attention ===
        Q = self.q_linear(feat_out)                  # Shape: [B, N, attn_dim]
        K = self.k_linear(feature2.permute(0, 2, 1)) # Shape: [B, M, attn_dim]
        V = self.v_linear(feature2.permute(0, 2, 1)) # Shape: [B, M, attn_dim]

        # Reshape for multi-head processing
        Q = Q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, N, head_dim]
        K = K.view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, M, head_dim]
        V = V.view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, M, head_dim]

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # [B, num_heads, N, M]
        attn_out = torch.matmul(attn_weights, V)          # [B, num_heads, N, head_dim]

        # Concatenate heads and project back
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, N, self.attn_dim)
        attn_out_projected = self.fc_out(attn_out) # Shape: [B, N, mlp[-1] * 2]

        # === 4. Residual Connection & Final Output ===
        final_feat = feat_out + attn_out_projected
        final_feat = self.layernorm(final_feat)
        final_feat = self.final_proj(final_feat)
        final_feat = final_feat.permute(0, 2, 1).contiguous()

        return pos1, final_feat
