import torch
import torch.nn as nn
import torch.nn.functional as F
from kan_layer import KANLinear

# KAN版本的非线性自适应融合
class KANNonlinearFusion(nn.Module):
    def __init__(self, input_dim, grid=5, order=3):
        super().__init__()
        self.fc1 = KANLinear(2 * input_dim, input_dim, grid_size=grid, spline_order=order)
        self.fc2 = KANLinear(input_dim, input_dim, grid_size=grid, spline_order=order)
        self.act = nn.SiLU()

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1)
        gate = torch.sigmoid(self.fc2(self.act(self.fc1(combined))))
        return gate * x1 + (1 - gate) * x2

# KAN版本的全局上下文融合
class KANGlobalFusion(nn.Module):
    def __init__(self, input_dim, grid=5, order=3):
        super().__init__()
        self.fc1 = KANLinear(input_dim, input_dim, grid_size=grid, spline_order=order)
        self.fc2 = KANLinear(input_dim, 1, grid_size=grid, spline_order=order)
        self.act = nn.SiLU()

    def forward(self, x):
        global_context = torch.mean(x, dim=0, keepdim=True)
        gate = torch.sigmoid(self.fc2(self.act(self.fc1(global_context))))
        return gate * x + (1 - gate) * global_context
