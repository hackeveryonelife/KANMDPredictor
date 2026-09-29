import numpy as np
import torch
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
from gpu_utils import ensure_sparse_coalesced


def compute_gip(adj):
    """高斯交互轮廓核相似性"""
    adj = np.asarray(adj, dtype=np.float64)
    n = adj.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norm2 = np.sum(adj ** 2, axis=1)
    gamma = 1.0 / (norm2.sum() / n)
    sq = norm2[:, None] + norm2[None, :] - 2.0 * (adj @ adj.T)
    sq = np.clip(sq, 0.0, None)
    return np.exp(-gamma * sq).astype(np.float32)


def knn_graph(matrix, k=20):
    """K近邻稀疏图，对称化+自环"""
    num = matrix.shape[0]
    knn = np.zeros_like(matrix)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    knn += np.eye(num)
    return knn


def normalize_adj(adj):
    """GCN对称归一化"""
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    deg_mat = np.diag(deg_inv_sqrt)
    return deg_mat @ adj @ deg_mat


def to_sparse(adj):
    """稠密邻接转稀疏COO张量"""
    coo = sp.coo_matrix(adj)
    indices = np.vstack([coo.row, coo.col])
    t = torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(coo.data),
        coo.shape
    )
    return ensure_sparse_coalesced(t)


def build_view_graphs(md_edges, mic_sim_gsm, dis_sim_gsm, mic_n, dis_n,
                      knn_k=20, gsm_weight=0.7):
    """构建三视图邻接图与节点特征（无泄漏协议）"""
    md_edges = np.asarray(md_edges, dtype=np.int64).reshape(-1, 2)
    md_adj = np.zeros((mic_n, dis_n), dtype=np.float32)
    if len(md_edges) > 0:
        md_adj[md_edges[:, 0], md_edges[:, 1]] = 1.0

    mic_gip = compute_gip(md_adj)
    dis_gip = compute_gip(md_adj.T)

    mic_sim = gsm_weight * mic_sim_gsm + (1 - gsm_weight) * mic_gip
    dis_sim = gsm_weight * dis_sim_gsm + (1 - gsm_weight) * dis_gip

    scaler = MinMaxScaler(feature_range=(-1, 1))
    mic_feat = scaler.fit_transform(mic_sim).astype(np.float32)
    dis_feat = scaler.fit_transform(dis_sim).astype(np.float32)

    adj_mic = to_sparse(normalize_adj(knn_graph(mic_sim, k=knn_k)))
    adj_dis = to_sparse(normalize_adj(knn_graph(dis_sim, k=knn_k)))

    total_node = mic_n + dis_n
    adj_md = np.zeros((total_node, total_node), dtype=np.float32)
    if len(md_edges) > 0:
        m_edges = md_edges[:, 0]
        d_edges = mic_n + md_edges[:, 1]
        adj_md[m_edges, d_edges] = 1.0
        adj_md[d_edges, m_edges] = 1.0
    np.fill_diagonal(adj_md, 1.0)
    adj_md = to_sparse(normalize_adj(adj_md))

    return mic_feat, dis_feat, adj_mic, adj_dis, adj_md


def build_triplets(pos_edges, dis_n, seed=None):
    """构建三元组 (anchor, pos, neg)"""
    pos_edges = np.asarray(pos_edges, dtype=np.int64).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    disease_map = {}
    for m_idx, d_idx in pos_edges:
        disease_map.setdefault(int(m_idx), []).append(int(d_idx))
    triplets = []
    for m_idx, pos_diseases in disease_map.items():
        pos_set = set(pos_diseases)
        candidates = [d for d in range(dis_n) if d not in pos_set]
        if not candidates:
            continue
        for pos_d in pos_diseases:
            neg_d = int(rng.choice(candidates))
            triplets.append([m_idx, pos_d, neg_d])
    if not triplets:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(triplets, dtype=np.int64).reshape(-1, 3)
