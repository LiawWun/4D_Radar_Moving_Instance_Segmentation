import torch
import torch.nn as nn
import torch.nn.functional as F

class GAMMA_MLP(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(GAMMA_MLP, self).__init__()
        mid_layer = (input_dim + output_dim) // 2
        self.gamma_mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=mid_layer),
            nn.ReLU(),
            nn.Linear(in_features=mid_layer, out_features=output_dim)
        )

    def forward(self, feature_in):
        return self.gamma_mlp(feature_in)
    
class PositionEncoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PositionEncoder, self).__init__()
        self.hidden_dim = (input_dim + output_dim) // 2
        
        self.position_embedding = nn.Sequential(
            nn.BatchNorm1d(input_dim), 
            nn.Linear(in_features=input_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=output_dim),
        )
        
    def forward(self, position):

        B, N, _, C = position.shape
        position = position.reshape(B * N * N, C)
        normalized_pos = self.position_embedding(position)
        
        return normalized_pos.reshape(B, N, N, -1)

class PointTransformerLayer(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(PointTransformerLayer, self).__init__()
        mid_layer = (in_channel + out_channel) // 2
        self.q = nn.Linear(in_channel, mid_layer)
        self.k = nn.Linear(in_channel, mid_layer)
        self.v = nn.Linear(in_channel, out_channel)
        
        self.position_encoding = PositionEncoder(input_dim=3, output_dim=mid_layer)
        self.gamma_mlp = GAMMA_MLP(input_dim=mid_layer, output_dim=out_channel)
        self.rho = nn.Softmax(dim=2)
        
    def forward(self, xyz, points):
        xyz_t = xyz
        points_t = points

        delta_p = xyz_t.unsqueeze(2) - xyz_t.unsqueeze(1)
        
        position_encoding = self.position_encoding(delta_p)

        q = self.q(points_t).unsqueeze(2)
        k = self.k(points_t).unsqueeze(1)
        attention_input = q - k + position_encoding
        gamma = self.gamma_mlp(attention_input)
        attention_weights = self.rho(gamma)
        v = self.v(points_t).unsqueeze(1)
        y = torch.sum(v * attention_weights, dim=2)
        return y

class PointTransformerBlock(nn.Module):

    def __init__ (self, in_channel, out_channel):
        super(PointTransformerBlock, self).__init__()
        self.fc1 = nn.Linear(in_channel, out_channel)
        self.point_transformer_layer = PointTransformerLayer(out_channel, out_channel)
        self.fc2 = nn.Linear(out_channel, out_channel)
        
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)

        self.activation = nn.GELU()
        
        if in_channel != out_channel:
            self.skip_connection = nn.Linear(in_channel, out_channel)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, xyz, point):

        xyz_t = xyz.permute(0, 2, 1).contiguous()
        points_t = point.permute(0, 2, 1).contiguous()

        residual = self.skip_connection(points_t)

        points_t = self.fc1(points_t)
        points_t = self.bn1(points_t.permute(0, 2, 1)).permute(0, 2, 1)
        points_t = self.activation(points_t)
        
        point_out = self.point_transformer_layer(xyz_t, points_t)
        
        point_out = self.bn2(point_out.permute(0, 2, 1)).permute(0, 2, 1)
        point_out = self.fc2(point_out)
        
        point_out += residual
        point_out = self.activation(point_out)
        
        point_out = point_out.permute(0, 2, 1).contiguous()

        return xyz, point_out


if __name__ == "__main__":
    B = 16   # batch size
    C = 3   # xyz dimensions
    D = 64  # feature dimensions
    N = 256  # number of points

    xyz = torch.randn(B, C, N) * 100
    points = torch.randn(B, D, N)

    model = PointTransformerBlock(in_channel=64, out_channel=64)
    out_xyz, out_points = model(xyz, points)

    print("Input xyz shape:", xyz.shape)
    print("Input points shape:", points.shape)
    print("Output xyz shape:", out_xyz.shape)
    print("Output points shape:", out_points.shape)