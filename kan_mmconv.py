import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from kan_layer import KANLinear
from gpu_utils import ensure_sparse_coalesced

class AdaptiveMMConv(nn.Module):
    def __init__(self, in_features, out_features, max_moment=6, use_center_moment=False,
                 kan_grid=5, kan_order=3, residual=True, lightweight=False, use_kan=True):
        super().__init__()
        self.max_moment = max_moment
        self.use_center_moment = use_center_moment
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.lightweight = lightweight

        if use_kan:
            self.weight = KANLinear(in_features, out_features, kan_grid, kan_order, bias=False)
        else:
            self.weight = nn.Linear(in_features, out_features, bias=False)
        self.moment_importance = Parameter(torch.ones(max_moment) / max_moment)

        use_linear_aux = lightweight or (not use_kan)
        if use_linear_aux:
            self.threshold_net = nn.Sequential(
                nn.Linear(in_features, 32),
                nn.SiLU(),
                nn.Linear(32, 16),
                nn.SiLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            self.moment_transforms = nn.ModuleList([
                nn.Linear(in_features, out_features) for _ in range(max_moment)
            ])
            self.enhanced_attention = nn.Sequential(
                nn.Linear(in_features * 2, 128),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(128, out_features),
                nn.Tanh()
            )
        else:
            self.threshold_net = nn.Sequential(
                KANLinear(in_features=in_features, out_features=32, grid_size=kan_grid, spline_order=kan_order),
                nn.SiLU(),
                KANLinear(32, 16, kan_grid, kan_order),
                nn.SiLU(),
                KANLinear(16, 1, kan_grid, kan_order),
                nn.Sigmoid()
            )
            self.moment_transforms = nn.ModuleList([
                KANLinear(in_features, out_features, kan_grid, kan_order) for _ in range(max_moment)
            ])
            self.enhanced_attention = nn.Sequential(
                KANLinear(in_features * 2, 128, kan_grid, kan_order),
                nn.SiLU(),
                nn.Dropout(0.1),
                KANLinear(128, out_features, kan_grid, kan_order),
                nn.Tanh()
            )

        # 残差投影
        if residual and in_features != out_features:
            self.res_proj = KANLinear(in_features, out_features, kan_grid, kan_order) if use_kan \
                else nn.Linear(in_features, out_features)
        else:
            self.res_proj = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.moment_importance, 0.5, 1.5)

    def stable_moment_calculation(self, x, adj_t, moment_order):
        eps = 1e-8
        # 支持稀疏矩阵乘法
        if adj_t.is_sparse:
            matmul = torch.sparse.mm
        else:
            matmul = torch.matmul

        if moment_order == 1:
            return matmul(adj_t, x)
        elif moment_order == 2:
            if self.use_center_moment:
                mu = matmul(adj_t, x)
                x_centered = x - mu
                variance = matmul(adj_t, x_centered.pow(2))
                return torch.sqrt(variance + eps)
            else:
                sigma = matmul(adj_t, x.pow(2))
                sigma = torch.clamp(sigma, min=eps)
                return sigma.sqrt()
        else:
            gamma = matmul(adj_t, x.pow(moment_order))
            gamma_abs = torch.abs(gamma)
            gamma_sign = torch.sign(gamma)
            gamma_abs = torch.clamp(gamma_abs, min=eps)
            log_gamma = torch.log(gamma_abs)
            stable_gamma = torch.exp(log_gamma / moment_order)
            if moment_order % 2 == 1:
                stable_gamma = stable_gamma * gamma_sign
            return stable_gamma

    def adaptive_moment_selection(self, x, adj_t, training=True):
        batch_size, node_dim = x.shape
        threshold_input = x.mean(dim=0, keepdim=True)
        dynamic_threshold = self.threshold_net(threshold_input).squeeze()
        if dynamic_threshold.dim() == 0:
            threshold_value = dynamic_threshold
        else:
            threshold_value = dynamic_threshold.view(-1).mean()

        if training:
            moment_weights = F.gumbel_softmax(
                self.moment_importance, tau=0.5, hard=False
            )
        else:
            moment_weights = F.softmax(self.moment_importance, dim=0)

        tau_thr = 0.1
        gate = torch.sigmoid((moment_weights - threshold_value) / tau_thr)
        top_k = min(3, self.max_moment)
        _, top_indices = torch.topk(moment_weights, top_k)
        top_mask = torch.zeros_like(moment_weights)
        top_mask[top_indices] = 1.0

        final_weights = torch.maximum(gate, top_mask) * moment_weights
        select_mask = ((gate > 0.5) | (top_mask > 0)).detach()
        selected_indices = torch.where(select_mask > 0)[0]

        selected_moments = []
        moment_contributions = []
        for idx in selected_indices:
            order = int(idx.item()) + 1
            weight = final_weights[idx]
            moment_val = self.stable_moment_calculation(x, adj_t, order)
            transformed_moment = self.moment_transforms[idx](moment_val)
            weighted_moment = transformed_moment * weight
            selected_moments.append(weighted_moment)
            moment_contributions.append(weight)

        if len(selected_moments) == 0:
            fallback_moment = self.stable_moment_calculation(x, adj_t, 1)
            return (
                [self.moment_transforms[0](fallback_moment)],
                torch.tensor([1.0], device=x.device),
                threshold_value.detach()
            )
        return selected_moments, torch.stack(moment_contributions), threshold_value.detach()

    def enhanced_attention_layer(self, moments, q):
        if len(moments) == 0:
            return q
        num_moments = len(moments)
        q_expanded = q.repeat(num_moments, 1)
        k_concat = torch.cat(moments, dim=0)
        attn_input = torch.cat([k_concat, q_expanded], dim=1)
        attn_input = F.dropout(attn_input, 0.3, training=self.training)
        attention_scores = self.enhanced_attention(attn_input)
        attention_scores = attention_scores.view(num_moments, -1, self.out_features).transpose(0, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        stacked_moments = torch.stack(moments, dim=1)
        weighted_output = (stacked_moments * attention_weights).sum(dim=1)
        return weighted_output

    def forward(self, input, adj, h0, lamda, alpha, l, beta=0.1):
        theta = math.log(lamda / l + 1)
        if adj.is_sparse:
            adj = ensure_sparse_coalesced(adj)
        if adj.is_sparse:
            h_agg = torch.sparse.mm(adj, input)
        else:
            h_agg = torch.matmul(adj, input)
        h_agg = (1 - alpha) * h_agg + alpha * h0

        h_i = self.weight(h_agg)
        h_i = theta * h_i + (1 - theta) * h_agg

        selected_moments, moment_weights, dynamic_threshold = self.adaptive_moment_selection(
            h0, adj, self.training
        )
        h_moment = self.enhanced_attention_layer(selected_moments, h_i)
        output = (1 - beta) * h_i + beta * h_moment

        if self.residual:
            output = output + self.res_proj(input)

        return output, {
            'selected_moments': len(selected_moments),
            'moment_weights': moment_weights,
            'dynamic_threshold': dynamic_threshold
        }
