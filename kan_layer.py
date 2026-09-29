import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=nn.SiLU,
            grid_eps=1e-6,
            grid_range=[-1, 1],
            bias=True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.bias = bias
        self.grid_eps = grid_eps
        self.n_basis = grid_size + spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, self.n_basis)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.register_buffer("spline_scaler", torch.ones(out_features, in_features))

        if self.bias:
            self.base_bias = nn.Parameter(torch.zeros(out_features))

        self.base_activation = base_activation()
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.scale_noise = scale_noise
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        noise = (
                torch.randn_like(self.spline_weight)
                * self.scale_noise
                / self.grid_size
        )
        self.spline_weight.data.copy_(noise)
        if hasattr(self, "spline_scaler"):
            nn.init.ones_(self.spline_scaler)

    def b_splines(self, x):
        N, D = x.shape
        x = x.unsqueeze(-1)  # [N, D, 1]
        grid = self.grid  # [G]
        G = len(grid)

        # 0阶基：向量化计算，消除Python循环
        left = grid[:-1]  # [G-1]
        right = grid[1:]  # [G-1]
        bases = (x >= left) & (x < right)  # [N, D, G-1]
        bases = bases.float()

        # k阶递推：阶数循环次数很少（默认3次），保留循环
        for k in range(1, self.spline_order + 1):
            n_intervals = G - 1 - k
            bases_left = bases[..., :-1]  # [N, D, n_intervals]
            bases_right = bases[..., 1:]  # [N, D, n_intervals]

            grid_left = grid[:-k-1]  # [n_intervals]
            grid_right_left = grid[k:-1]  # [n_intervals]
            grid_right_right = grid[k+1:]  # [n_intervals]
            grid_left_right = grid[1:-k]  # [n_intervals]

            denom_left = grid_right_left - grid_left + self.grid_eps
            term_left = (x - grid_left) / denom_left * bases_left

            denom_right = grid_right_right - grid_left_right + self.grid_eps
            term_right = (grid_right_right - x) / denom_right * bases_right

            bases = term_left + term_right

        return bases

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base = self.base_activation(x) @ self.base_weight.T * self.scale_base
        spline_bases = self.b_splines(x)
        scaled_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline = torch.einsum("ndi,odi->no", spline_bases, scaled_weight)
        spline = spline * self.scale_spline

        output = base + spline
        if self.bias:
            output += self.base_bias
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output
