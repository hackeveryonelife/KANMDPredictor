import torch
import torch.nn as nn
from kan_layer import KANLinear
from kan_mmconv import AdaptiveMMConv

# ========== 支持消融的基础模块 ==========
class KANBlockAblation(nn.Module):
    def __init__(self, dim_in, dim_out, grid=5, order=3, use_kan=True):
        super().__init__()
        self.act = nn.SiLU()
        if use_kan:
            self.layer1 = KANLinear(dim_in, dim_out, grid_size=grid, spline_order=order)
            self.layer2 = KANLinear(dim_out, dim_out, grid_size=grid, spline_order=order)
            self.shortcut = KANLinear(dim_in, dim_out, grid_size=grid, spline_order=order) if dim_in != dim_out else nn.Identity()
        else:
            self.layer1 = nn.Linear(dim_in, dim_out)
            self.layer2 = nn.Linear(dim_out, dim_out)
            self.shortcut = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        x = self.act(self.layer1(x))
        x = self.layer2(x)
        return x + res


class KANNonlinearFusionAblation(nn.Module):
    def __init__(self, input_dim, grid=5, order=3, use_kan=True, use_fusion=True):
        super().__init__()
        self.use_fusion = use_fusion
        if use_fusion:
            if use_kan:
                self.fc1 = KANLinear(2 * input_dim, input_dim, grid_size=grid, spline_order=order)
                # 逐维门控：每个维度独立选分支（与 kan_fusion.py 保持一致）
                self.fc2 = KANLinear(input_dim, input_dim, grid_size=grid, spline_order=order)
            else:
                self.fc1 = nn.Linear(2 * input_dim, input_dim)
                self.fc2 = nn.Linear(input_dim, input_dim)
            self.act = nn.SiLU()

    def forward(self, x1, x2):
        if not self.use_fusion:
            return (x1 + x2) / 2  # 消融门控：直接平均融合
        combined = torch.cat([x1, x2], dim=-1)
        gate = torch.sigmoid(self.fc2(self.act(self.fc1(combined))))
        return gate * x1 + (1 - gate) * x2


class KANGlobalFusionAblation(nn.Module):
    def __init__(self, input_dim, grid=5, order=3, use_kan=True, use_global=True):
        super().__init__()
        self.use_global = use_global
        if use_global:
            if use_kan:
                self.fc1 = KANLinear(input_dim, input_dim, grid_size=grid, spline_order=order)
                self.fc2 = KANLinear(input_dim, 1, grid_size=grid, spline_order=order)
            else:
                self.fc1 = nn.Linear(input_dim, input_dim)
                self.fc2 = nn.Linear(input_dim, 1)
            self.act = nn.SiLU()

    def forward(self, x):
        if not self.use_global:
            return x  # 消融全局上下文：直接返回原特征
        global_context = torch.mean(x, dim=0, keepdim=True)
        gate = torch.sigmoid(self.fc2(self.act(self.fc1(global_context))))
        return gate * x + (1 - gate) * global_context


class KANScoreHeadAblation(nn.Module):
    def __init__(self, dim, grid=5, order=3, use_img=False, img_dim=0, use_kan=True):
        super().__init__()
        self.use_img = use_img
        self.img_dim = img_dim
        input_dim = dim * 2 + (img_dim if use_img else 0)
        if use_kan:
            self.score_mlp = nn.Sequential(
                KANLinear(input_dim, dim, grid_size=grid, spline_order=order),
                nn.SiLU(),
                KANLinear(dim, dim // 2, grid_size=grid, spline_order=order),
                nn.SiLU(),
                KANLinear(dim // 2, 1, grid_size=grid, spline_order=order)
            )
        else:
            self.score_mlp = nn.Sequential(
                nn.Linear(input_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim // 2),
                nn.SiLU(),
                nn.Linear(dim // 2, 1)
            )

    def forward(self, mic_emb, dis_emb, pairs, img_feat=None):
        mic = mic_emb[pairs[:, 0]]
        dis = dis_emb[pairs[:, 1]]
        if self.use_img:
            if img_feat is not None:
                # ===== img 特征是样本级特征，行序与传入的 batch 对齐，

                pair_img = img_feat
            else:
                pair_img = torch.zeros(mic.shape[0], self.img_dim, device=mic.device)
            combined = torch.cat([mic, dis, pair_img], dim=-1)
        else:
            combined = torch.cat([mic, dis], dim=-1)
        logits = self.score_mlp(combined)
        return logits.flatten()


# ========== 支持完整消融的主模型 ==========
class KANMDPredictorAblation(nn.Module):
    def __init__(
            self,
            mic_dim=788,
            dis_dim=374,
            hidden_dim=128,
            grid=7,
            order=3,
            max_moment=6,
            lamda=1.0,
            alpha=0.1,
            beta=0.5,
            use_img=False,
            img_dim=0,
            lightweight=False,
            use_kan=True,        # 消融总开关：False=全部替换为MLP
            use_fusion=True,     # 消融自适应融合
            use_global=True      # 消融全局上下文
    ):
        super().__init__()
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.use_img = use_img

        # 输入投影
        self.mic_proj = KANBlockAblation(mic_dim, hidden_dim, grid=grid, order=order, use_kan=use_kan)
        self.dis_proj = KANBlockAblation(dis_dim, hidden_dim, grid=grid, order=order, use_kan=use_kan)

        # 多阶矩卷积
        self.gcn_mic = AdaptiveMMConv(hidden_dim, hidden_dim, max_moment=max_moment,
                                       kan_grid=grid, kan_order=order,
                                       lightweight=lightweight or not use_kan, use_kan=use_kan)
        self.gcn_dis = AdaptiveMMConv(hidden_dim, hidden_dim, max_moment=max_moment,
                                       kan_grid=grid, kan_order=order,
                                       lightweight=lightweight or not use_kan, use_kan=use_kan)
        self.gcn_md1 = AdaptiveMMConv(hidden_dim, hidden_dim, max_moment=max_moment,
                                       kan_grid=grid, kan_order=order,
                                       lightweight=lightweight or not use_kan, use_kan=use_kan)
        self.gcn_md2 = AdaptiveMMConv(hidden_dim, hidden_dim, max_moment=max_moment,
                                       kan_grid=grid, kan_order=order,
                                       lightweight=lightweight or not use_kan, use_kan=use_kan)

        # 融合模块
        self.fusion_mic = KANNonlinearFusionAblation(hidden_dim, grid=grid, order=order,
                                                      use_kan=use_kan, use_fusion=use_fusion)
        self.fusion_dis = KANNonlinearFusionAblation(hidden_dim, grid=grid, order=order,
                                                      use_kan=use_kan, use_fusion=use_fusion)
        self.global_mic = KANGlobalFusionAblation(hidden_dim, grid=grid, order=order,
                                                   use_kan=use_kan, use_global=use_global)
        self.global_dis = KANGlobalFusionAblation(hidden_dim, grid=grid, order=order,
                                                   use_kan=use_kan, use_global=use_global)

        # 输出映射
        self.mic_out = KANBlockAblation(hidden_dim, hidden_dim, grid=grid, order=order, use_kan=use_kan)
        self.dis_out = KANBlockAblation(hidden_dim, hidden_dim, grid=grid, order=order, use_kan=use_kan)

        # 打分头
        self.score_head = KANScoreHeadAblation(hidden_dim, grid=grid, order=order,
                                                use_img=use_img, img_dim=img_dim, use_kan=use_kan)

    def forward(self, mic_feat, dis_feat, adj_mic, adj_dis, adj_md, mic_n, dis_n, pairs=None, img_feat=None):
        mic_h = self.mic_proj(mic_feat)
        dis_h = self.dis_proj(dis_feat)
        all_h = torch.cat([mic_h, dis_h], dim=0)
        h0 = all_h

        # 分支1：内部相似图卷积
        mic_sim, _ = self.gcn_mic(mic_h, adj_mic, mic_h, self.lamda, self.alpha, 1, self.beta)
        dis_sim, _ = self.gcn_dis(dis_h, adj_dis, dis_h, self.lamda, self.alpha, 1, self.beta)

        # 分支2：关联二部图卷积
        all_md, _ = self.gcn_md1(all_h, adj_md, h0, self.lamda, self.alpha, 1, self.beta)
        all_md_res = all_md
        all_md, _ = self.gcn_md2(all_md, adj_md, h0, self.lamda, self.alpha, 2, self.beta)
        all_md = all_md + all_md_res
        mic_md = all_md[:mic_n]
        dis_md = all_md[mic_n:]

        # 融合 + 全局增强
        mic_final = self.global_mic(self.fusion_mic(mic_sim, mic_md))
        dis_final = self.global_dis(self.fusion_dis(dis_sim, dis_md))

        mic_emb = self.mic_out(mic_final)
        dis_emb = self.dis_out(dis_final)

        if pairs is not None:
            logits = self.score_head(mic_emb, dis_emb, pairs, img_feat)
            return logits, mic_emb, dis_emb
        else:
            return mic_emb, dis_emb
