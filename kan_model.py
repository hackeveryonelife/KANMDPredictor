import torch
import torch.nn as nn
from kan_layer import KANLinear
from kan_mmconv import AdaptiveMMConv
from kan_fusion import KANNonlinearFusion, KANGlobalFusion
class KANBlock(nn.Module):
    def __init__(self, dim_in, dim_out, grid=5, order=3):
        super().__init__()
        self.kan1 = KANLinear(dim_in, dim_out, grid_size=grid, spline_order=order)
        self.kan2 = KANLinear(dim_out, dim_out, grid_size=grid, spline_order=order)
        self.act = nn.SiLU()
        if dim_in != dim_out:
            self.shortcut = KANLinear(dim_in, dim_out, grid_size=grid, spline_order=order)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        res = self.shortcut(x)
        x = self.act(self.kan1(x))
        x = self.kan2(x)
        return x + res
# KAN样本级打分头（支持图像特征拼接）
class KANScoreHead(nn.Module):
    def __init__(self, dim, grid=5, order=3, use_img=False, img_dim=0):
        super().__init__()
        self.use_img = use_img
        self.img_dim = img_dim
        input_dim = dim * 2 + (img_dim if use_img else 0)
        self.score_mlp = nn.Sequential(
            KANLinear(input_dim, dim, grid_size=grid, spline_order=order),
            nn.SiLU(),
            KANLinear(dim, dim // 2, grid_size=grid, spline_order=order),
            nn.SiLU(),
            KANLinear(dim // 2, 1, grid_size=grid, spline_order=order)
        )
    def forward(self, mic_emb, dis_emb, pairs, img_feat=None):
        mic = mic_emb[pairs[:, 0]]
        dis = dis_emb[pairs[:, 1]]
        if self.use_img:
            if img_feat is not None:
                pair_img = img_feat
            else:
                pair_img = torch.zeros(mic.shape[0], self.img_dim, device=mic.device)
            combined = torch.cat([mic, dis, pair_img], dim=-1)
        else:
            combined = torch.cat([mic, dis], dim=-1)
        logits = self.score_mlp(combined)
        return logits.flatten()
class KANMDPredictor(nn.Module):
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
            beta=0.5,   # moment分支权重：原0.2太小，高阶矩被稀释到几乎无影响
            use_img=False,
            img_dim=0,
            lightweight=False,
            use_kan = True,
            use_fusion = True,
            use_global = True
    ):
        super().__init__()
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.use_img = use_img
        self.lightweight = lightweight
        # 输入特征投影
        self.mic_proj = KANBlock(mic_dim, hidden_dim, grid=grid, order=order)
        self.dis_proj = KANBlock(dis_dim, hidden_dim, grid=grid, order=order)
        self.gcn_mic = AdaptiveMMConv(
            hidden_dim, hidden_dim,
            max_moment=max_moment,
            kan_grid=grid,
            kan_order=order,
            lightweight=lightweight,
            use_kan=use_kan
        )
        self.gcn_dis = AdaptiveMMConv(
            hidden_dim, hidden_dim,
            max_moment=max_moment,
            kan_grid=grid,
            kan_order=order,
            lightweight=lightweight,
            use_kan=use_kan
        )
        self.gcn_md1 = AdaptiveMMConv(
            hidden_dim, hidden_dim,
            max_moment=max_moment,
            kan_grid=grid,
            kan_order=order,
            lightweight=lightweight,
            use_kan=use_kan
        )
        self.gcn_md2 = AdaptiveMMConv(
            hidden_dim, hidden_dim,
            max_moment=max_moment,
            kan_grid=grid,
            kan_order=order,
            lightweight=lightweight,
            use_kan=use_kan
        )
        # 自适应融合模块
        self.fusion_mic = KANNonlinearFusion(hidden_dim, grid=grid, order=order)
        self.fusion_dis = KANNonlinearFusion(hidden_dim, grid=grid, order=order)
        self.global_mic = KANGlobalFusion(hidden_dim, grid=grid, order=order)
        self.global_dis = KANGlobalFusion(hidden_dim, grid=grid, order=order)
        # 输出映射
        self.mic_out = KANBlock(hidden_dim, hidden_dim, grid=grid, order=order)
        self.dis_out = KANBlock(hidden_dim, hidden_dim, grid=grid, order=order)
        # 打分头
        self.score_head = KANScoreHead(hidden_dim, grid=grid, order=order,
                                       use_img=use_img, img_dim=img_dim)
    def forward(self, mic_feat, dis_feat, adj_mic, adj_dis, adj_md, mic_n, dis_n, pairs=None, img_feat=None):
        # 输入投影
        mic_h = self.mic_proj(mic_feat)
        dis_h = self.dis_proj(dis_feat)
        all_h = torch.cat([mic_h, dis_h], dim=0)
        h0 = all_h
        # 分支1：内部相似图卷积
        mic_sim, _ = self.gcn_mic(mic_h, adj_mic, mic_h, self.lamda, self.alpha, 1, self.beta)
        dis_sim, _ = self.gcn_dis(dis_h, adj_dis, dis_h, self.lamda, self.alpha, 1, self.beta)
        # 分支2：关联二部图卷积（双层+残差）
        all_md, _ = self.gcn_md1(all_h, adj_md, h0, self.lamda, self.alpha, 1, self.beta)
        all_md_res = all_md
        all_md, _ = self.gcn_md2(all_md, adj_md, h0, self.lamda, self.alpha, 2, self.beta)
        all_md = all_md + all_md_res
        mic_md = all_md[:mic_n]
        dis_md = all_md[mic_n:]
        # 双分支自适应融合 + 全局上下文增强
        mic_final = self.global_mic(self.fusion_mic(mic_sim, mic_md))
        dis_final = self.global_dis(self.fusion_dis(dis_sim, dis_md))
        # 输出映射
        mic_emb = self.mic_out(mic_final)
        dis_emb = self.dis_out(dis_final)
        if pairs is not None:
            logits = self.score_head(mic_emb, dis_emb, pairs, img_feat)
            return logits, mic_emb, dis_emb
        else:
            return mic_emb, dis_emb
