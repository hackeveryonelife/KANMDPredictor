import os
import warnings

warnings.filterwarnings("ignore", message="CUDA initialization")

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
from kan_model_ablation import KANMDPredictorAblation
from kan_train import train_kan_model
from gpu_utils import get_device, ensure_sparse_coalesced


from graph_builder import build_view_graphs, build_triplets


def load_data():
    dis_sim_gsm = np.loadtxt("HMDD3.2/D_GSM.csv", delimiter=",")
    mic_sim_gsm = np.loadtxt("HMDD3.2/M_GSM.csv", delimiter=",")
    pair_df = pd.read_csv("HMDD3.2/all_mirna_disease_pairs.csv", header=None)
    mic_num = pair_df[0].nunique()
    dis_num = pair_df[1].nunique()

    all_pos = pair_df[pair_df[2] == 1].values
    all_neg = pair_df[pair_df[2] == 0].values
    all_pos_edges = np.array([[int(m) - 1, int(d) - 1] for m, d, _ in all_pos], dtype=np.int64)
    md_adj = np.zeros((mic_num, dis_num), dtype=np.float32)
    for m, d, lab in pair_df.values:
        md_adj[int(m) - 1, int(d) - 1] = lab

    mic_feat, dis_feat, adj_mic, adj_dis, adj_md = build_view_graphs(
        all_pos_edges, mic_sim_gsm, dis_sim_gsm, mic_num, dis_num, knn_k=20, gsm_weight=0.7)
    device = get_device()
    adj_mic = adj_mic.to(device)
    adj_dis = adj_dis.to(device)
    adj_md = adj_md.to(device)

    np.random.seed(42)
    sampled_neg = all_neg[np.random.choice(len(all_neg), len(all_pos), replace=False)]
    balanced_samples = np.concatenate([all_pos, sampled_neg], axis=0)

    sample_pairs = []
    sample_labels = []
    for m, d, lab in balanced_samples:
        sample_pairs.append([int(m) - 1, int(d) - 1])
        sample_labels.append(lab)
    sample_pairs = np.array(sample_pairs, dtype=np.int64)
    sample_labels = np.array(sample_labels, dtype=np.float32)

    try:
        img_feat = np.load("HMDD3.2/img_fused_features_HMDDv3.2.npy", allow_pickle=True).astype(float)
        default_use_img = True
        default_img_dim = img_feat.shape[1]
    except:
        img_feat = None
        default_use_img = False
        default_img_dim = 0

    base_train_data = {
        "mic_sim_gsm": mic_sim_gsm, "dis_sim_gsm": dis_sim_gsm,
        "sample_pairs": sample_pairs, "sample_labels": sample_labels,
        "mic_n": mic_num, "dis_n": dis_num,
        "mic_feat": mic_feat, "dis_feat": dis_feat,
        "adj_mic": adj_mic, "adj_dis": adj_dis, "adj_md": adj_md,
    }
    image_data = {
        "img_feat": img_feat,
        "default_use_img": default_use_img,
        "default_img_dim": default_img_dim,
        "md_adj": md_adj,
        "all_pos_edges": all_pos_edges,
    }
    return base_train_data, image_data


def save_plot_data(model, base_train_data, image_data, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    mic_feat = torch.FloatTensor(base_train_data["mic_feat"]).to(device)
    dis_feat = torch.FloatTensor(base_train_data["dis_feat"]).to(device)
    adj_mic = base_train_data["adj_mic"].to(device)
    adj_dis = base_train_data["adj_dis"].to(device)
    adj_md = base_train_data["adj_md"].to(device)
    mic_n = base_train_data["mic_n"]
    dis_n = base_train_data["dis_n"]
    use_img = model.use_img
    img_feat = torch.FloatTensor(image_data["img_feat"]).to(device) if (use_img and image_data["img_feat"] is not None) else None

    pairs_all = torch.LongTensor(base_train_data["sample_pairs"]).to(device)
    with torch.no_grad():
        _, mic_emb, dis_emb = model(
            mic_feat, dis_feat, adj_mic, adj_dis, adj_md,
            mic_n, dis_n, pairs=pairs_all, img_feat=img_feat
        )
    np.save(os.path.join(out_dir, "mic_emb.npy"), mic_emb.cpu().numpy())
    np.save(os.path.join(out_dir, "dis_emb.npy"), dis_emb.cpu().numpy())

    if hasattr(model.gcn_mic, "moment_importance"):
        mw = model.gcn_mic.moment_importance.detach().cpu()
        mw_soft = torch.softmax(mw, dim=0).numpy()
        df_mw = pd.DataFrame({"moment_order": np.arange(1, len(mw_soft)+1), "weight": mw_soft})
        df_mw.to_csv(os.path.join(out_dir, "moment_weight.csv"), index=False)


ablation_configs = {
    "Full Model": {
        "use_kan": True, "max_moment": 6,
        "use_fusion": True, "use_img": True,
        "bce_weight": 0.6, "bpr_weight": 0.2, "triplet_weight": 0.2,
    },
    "w/o KAN (MLP)": {
        "use_kan": False, "max_moment": 6,
        "use_fusion": True, "use_img": True,
        "bce_weight": 0.6, "bpr_weight": 0.2, "triplet_weight": 0.2,
    },
    "w/o High-order Moment": {
        "use_kan": True, "max_moment": 1,
        "use_fusion": True, "use_img": True,
        "bce_weight": 0.6, "bpr_weight": 0.2, "triplet_weight": 0.2,
    },
    "w/o Adaptive Fusion": {
        "use_kan": True, "max_moment": 6,
        "use_fusion": False, "use_img": True,
        "bce_weight": 0.6, "bpr_weight": 0.2, "triplet_weight": 0.2,
    },
    "w/o Image Feature": {
        "use_kan": True, "max_moment": 6,
        "use_fusion": True, "use_img": False,
        "bce_weight": 0.6, "bpr_weight": 0.2, "triplet_weight": 0.2,
    },
}

if __name__ == "__main__":
    base_train_data, image_data = load_data()
    device = get_device()
    results = []
    root_out = "output_ablation"
    os.makedirs(root_out, exist_ok=True)

    for name, cfg in ablation_configs.items():
        safe_name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        out_subdir = os.path.join(root_out, safe_name)
        model_save_path = os.path.join(out_subdir, "model.pth")

        print(f"\n{'=' * 60}")
        print(f"Running ablation: {name}")
        print(f"输出目录: {out_subdir}")
        print(f"{'=' * 60}")

        if cfg["use_img"] and image_data["default_use_img"]:
            run_img_feat = image_data["img_feat"]
            run_img_dim = image_data["default_img_dim"]
            run_use_img = True
        else:
            run_img_feat = None
            run_img_dim = 0
            run_use_img = False

        mean_auc, mean_aupr, fold_results = train_kan_model(
            mic_sim_gsm=base_train_data["mic_sim_gsm"],
            dis_sim_gsm=base_train_data["dis_sim_gsm"],
            sample_pairs=base_train_data["sample_pairs"],
            sample_labels=base_train_data["sample_labels"],
            mic_n=base_train_data["mic_n"],
            dis_n=base_train_data["dis_n"],
            device=device,
            kan_grid=7,
            kan_order=3,
            lightweight=False,
            model_class=KANMDPredictorAblation,
            use_kan=cfg["use_kan"],
            max_moment=cfg["max_moment"],
            use_fusion=cfg["use_fusion"],
            use_global=True,
            use_img=run_use_img,
            img_feat=run_img_feat,
            img_dim=run_img_dim,
            bce_weight=cfg["bce_weight"],
            bpr_weight=cfg["bpr_weight"],
            triplet_weight=cfg["triplet_weight"],
            epochs=200,
            lr=0.0003,
            early_stop=True,
            patience=20,
            eval_interval=5,
            save_best_path=model_save_path,
            log_out_dir=out_subdir
        )

        # 计算所有指标的均值和标准差
        arr_auc = np.array(fold_results['auc'])
        arr_aupr = np.array(fold_results['aupr'])
        arr_acc = np.array(fold_results['acc'])
        arr_pre = np.array(fold_results['precision'])
        arr_rec = np.array(fold_results['recall'])
        arr_f1 = np.array(fold_results['f1'])

        results.append({
            "Method": name,
            "AUC_mean": round(arr_auc.mean(), 4),
            "AUC_std": round(arr_auc.std(), 4),
            "AUPR_mean": round(arr_aupr.mean(), 4),
            "AUPR_std": round(arr_aupr.std(), 4),
            "Acc_mean": round(arr_acc.mean(), 4),
            "Acc_std": round(arr_acc.std(), 4),
            "Precision_mean": round(arr_pre.mean(), 4),
            "Precision_std": round(arr_pre.std(), 4),
            "Recall_mean": round(arr_rec.mean(), 4),
            "Recall_std": round(arr_rec.std(), 4),
            "F1_mean": round(arr_f1.mean(), 4),
            "F1_std": round(arr_f1.std(), 4),
        })

        # 保存绘图数据
        model = KANMDPredictorAblation(
            mic_dim=base_train_data["mic_feat"].shape[1],
            dis_dim=base_train_data["dis_feat"].shape[1],
            hidden_dim=128,
            grid=7, order=3,
            max_moment=cfg["max_moment"],
            use_img=run_use_img,
            img_dim=run_img_dim,
            use_kan=cfg["use_kan"],
            use_fusion=cfg["use_fusion"],
            use_global=True
        ).to(device)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        save_plot_data(model, base_train_data, image_data, out_subdir, device)
        results[-1]["Parameter_Count"] = sum(p.numel() for p in model.parameters())
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  完成！AUC={mean_auc:.4f}, AUPR={mean_aupr:.4f}")

    # 输出完整汇总表
    df_result = pd.DataFrame(results)
    df_result.to_csv(os.path.join(root_out, "ablation_full_result.csv"), index=False)
    print("\n" + "=" * 60)
    print("消融实验全部完成，所有结果保存在 output_ablation/")
    print("=" * 60)
    print(df_result.to_string(index=False))
