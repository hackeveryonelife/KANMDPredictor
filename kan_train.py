import os
import warnings
warnings.filterwarnings("ignore", message="CUDA initialization")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np
import pandas as pd
from tqdm import tqdm
from kan_model import KANMDPredictor
from graph_builder import build_view_graphs, build_triplets
import copy
import json
import time

def bpr_loss(pos_logits, neg_logits):
    diff = pos_logits - neg_logits
    return -torch.mean(F.logsigmoid(diff))

def triplet_loss_fn(anchor, positive, negative, margin=0.5):
    distance_pos = torch.norm(anchor - positive, p=2, dim=1)
    distance_neg = torch.norm(anchor - negative, p=2, dim=1)
    losses = torch.clamp(distance_pos - distance_neg + margin, min=0.0)
    return losses.mean()

def eval_top1_hit(model, mic_feat, dis_feat, adj_mic, adj_dis, adj_md,
                   mic_n, dis_n, train_pos_edges, val_pairs, val_y, device,
                   batch_size=8192, save_cand_dir=None, fold_id=None):
    model.eval()
    val_y_np = val_y.cpu().numpy()
    val_pos = val_pairs.cpu().numpy()[val_y_np == 1]
    if len(val_pos) == 0:
        return 0.0, 0.0, 0.0
    train_pos_set = set(map(tuple, train_pos_edges.tolist()))
    val_pos_set = set(map(tuple, val_pos.tolist()))
    diseases = np.unique(val_pos[:, 1])
    hits = 0
    cand_scores_all = []
    cand_idx_all = []
    with torch.no_grad():
        for d in diseases:
            cand_m = [m for m in range(mic_n) if (int(m), int(d)) not in train_pos_set]
            if not cand_m:
                continue
            cand = np.array([[m, int(d)] for m in cand_m], dtype=np.int64)
            scores = []
            for s in range(0, len(cand), batch_size):
                batch = torch.LongTensor(cand[s:s + batch_size]).to(device)
                logits, _, _ = model(mic_feat, dis_feat, adj_mic, adj_dis, adj_md,
                                     mic_n, dis_n, pairs=batch, img_feat=None)
                scores.append(torch.sigmoid(logits).cpu().numpy())
            scores = np.concatenate(scores)
            cand_scores_all.append(scores)
            cand_idx_all.append(cand[:, 0])
            top1 = cand[int(np.argmax(scores))]
            if tuple(top1.tolist()) in val_pos_set:
                hits += 1
    n = len(diseases)
    p = hits / n if n > 0 else 0.0
    if save_cand_dir is not None and fold_id is not None:
        np.savez(os.path.join(save_cand_dir, "fold%d_cand_scores.npz" % fold_id),
                 disease_ids=diseases,
                 cand_scores=np.array(cand_scores_all, dtype=object),
                 cand_idx=np.array(cand_idx_all, dtype=object),
                 val_pos_pairs=val_pos,
                 mic_n=mic_n, dis_n=dis_n)
    return p, p, p


def train_kan_model(
        mic_sim_gsm,
        dis_sim_gsm,
        sample_pairs,
        sample_labels,
        mic_n,
        dis_n,
        device,
        epochs=300,
        lr=0.0003,
        kan_grid=7,
        kan_order=3,
        n_folds=5,
        use_img=False,
        img_dim=0,
        img_feat=None,
        lightweight=False,
        early_stop=True,
        patience=20,
        min_delta=1e-4,
        metric_for_early_stop="auc",
        eval_interval=2,
        model_class=KANMDPredictor,
        hidden_dim=128,
        max_moment=6,
        use_kan=True,
        use_fusion=True,
        use_global=True,
        bce_weight=0.6,
        bpr_weight=0.2,
        triplet_weight=0.2,
        l1_lambda=1e-5,
        save_best_path=None,
        log_out_dir=None,
        permute_seed=None,
        cv_indices=None,
        threshold=0.55,
        top1_eval=False
):
    # 创建日志目录
    if log_out_dir is not None:
        os.makedirs(log_out_dir, exist_ok=True)

    t_start = time.time()
    config_record = {
        "epochs": epochs, "lr": lr, "kan_grid": kan_grid, "kan_order": kan_order,
        "n_folds": n_folds, "hidden_dim": hidden_dim, "max_moment": max_moment,
        "use_kan": use_kan, "use_fusion": use_fusion, "use_global": use_global,
        "use_img": use_img, "img_dim": img_dim,
        "bce_weight": bce_weight, "bpr_weight": bpr_weight,
        "triplet_weight": triplet_weight, "l1_lambda": l1_lambda,
        "early_stop": early_stop, "patience": patience, "min_delta": min_delta,
        "metric_for_early_stop": metric_for_early_stop, "eval_interval": eval_interval,
        "threshold": threshold,
        "top1_eval": top1_eval,
        "num_samples": int(len(sample_labels)),
        "pos_samples": int(sample_labels.sum().item()),
        "neg_samples": int(len(sample_labels) - sample_labels.sum().item()),
        "protocol": "inductive_no_leakage",
        "knn_k": 20, "gsm_weight": 0.7,
        "seed": 42,
        "permutation_check": permute_seed is not None,
    }
    if log_out_dir is not None:
        with open(os.path.join(log_out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_record, f, indent=2, ensure_ascii=False)

    sample_pairs = torch.LongTensor(sample_pairs).to(device)
    sample_labels = torch.FloatTensor(sample_labels).to(device)
    img_feat_all = torch.FloatTensor(img_feat).to(device) if img_feat is not None else None
    pos_num = sample_labels.sum().item()
    neg_num = len(sample_labels) - pos_num
    print(f"总样本数: {len(sample_labels)}, 正样本: {int(pos_num)}, 负样本: {int(neg_num)}")
    print(f"轻量模式: {'开启' if lightweight else '关闭'}")
    bce_loss_fn = nn.BCEWithLogitsLoss()
    if cv_indices is not None:
        splits = list(cv_indices)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kf.split(sample_labels))
    fold_results = {
        "fold": [],
        'auc': [],
        'aupr': [],
        'acc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'top1_precision': [],
        'top1_recall': [],
        'top1_f1': []
    }

    best_global_auc = 0.0
    best_global_state = None

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        fid = fold
        print(f"\n===== Fold {fid}/{n_folds} =====")
        train_pairs = sample_pairs[train_idx]
        val_pairs = sample_pairs[val_idx]
        train_y = sample_labels[train_idx]
        val_y = sample_labels[val_idx]

        train_y_np = train_y.cpu().numpy()
        train_pairs_np = train_pairs.cpu().numpy()
        train_pos_edges = train_pairs_np[train_y_np == 1]
        if permute_seed is not None:
            # （图结构/边数/度分布不变，仅关联语义被随机化）
            rng_p = np.random.default_rng(permute_seed + fid)
            perm = rng_p.permutation(mic_n)
            train_pos_edges = train_pos_edges.copy()
            train_pos_edges[:, 0] = perm[train_pos_edges[:, 0]]
        mic_feat, dis_feat, adj_mic, adj_dis, adj_md = build_view_graphs(
            train_pos_edges, mic_sim_gsm, dis_sim_gsm, mic_n, dis_n,
            knn_k=20, gsm_weight=0.7)
        mic_feat = torch.FloatTensor(mic_feat).to(device)
        dis_feat = torch.FloatTensor(dis_feat).to(device)
        adj_mic = adj_mic.to(device)
        adj_dis = adj_dis.to(device)
        adj_md = adj_md.to(device)
        triplet_np = build_triplets(train_pos_edges, dis_n)
        triplet_samples = torch.LongTensor(triplet_np).to(device)

        pos_mask = (train_y == 1).cpu().numpy()
        pos_count = pos_mask.sum()
        neg_count = len(train_y) - pos_count
        print(f"训练集正样本: {pos_count}, 负样本: {neg_count} (1:1)")

        model = model_class(
            mic_dim=mic_feat.shape[1],
            dis_dim=dis_feat.shape[1],
            hidden_dim=hidden_dim,
            grid=kan_grid,
            order=kan_order,
            max_moment=max_moment,
            use_img=use_img,
            img_dim=img_dim,
            lightweight=lightweight,
            use_kan=use_kan,
            use_fusion=use_fusion,
            use_global=use_global
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best_auc = 0.0
        best_aupr = 0.0
        best_acc = 0.0
        best_pre = 0.0
        best_f1 = 0.0
        best_recall = 0.0
        best_epoch = 0
        best_state = None
        epochs_no_improve = 0
        last_val_auc = 0.0
        last_val_aupr = 0.0
        last_val_acc = 0.0
        last_val_pre = 0.0
        last_val_f1 = 0.0
        last_val_recall = 0.0
        epoch_log = {
            "epoch": [],
            "total_loss": [],
            "bce_loss": [],
            "bpr_loss": [],
            "triplet_loss": [],
            "l1_loss": [],
            "auc": [],
            "aupr": [],
            "acc": [],
            "pre": [],
            "recall": [],
            "f1": []
        }
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数量: {n_params:,}")
        pbar = tqdm(range(epochs), desc=f"Fold {fid}", unit="epoch", ncols=120)
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()
            train_img = img_feat_all[train_idx] if use_img else None
            train_logits, mic_emb, dis_emb = model(
                mic_feat, dis_feat, adj_mic, adj_dis, adj_md, mic_n, dis_n,
                pairs=train_pairs, img_feat=train_img
            )
            bce = bce_loss_fn(train_logits, train_y)
            pos_mask_t = (train_y == 1)
            neg_mask_t = ~pos_mask_t
            pos_logits = train_logits[pos_mask_t]
            neg_logits = train_logits[neg_mask_t]
            num_pairs = min(pos_mask_t.sum(), neg_mask_t.sum())
            if num_pairs == 0:
                bpr = torch.tensor(0.0, device=train_logits.device)
            else:
                pos_idx = torch.randperm(pos_mask_t.sum())[:num_pairs]
                neg_idx = torch.randperm(neg_mask_t.sum())[:num_pairs]
                bpr = bpr_loss(pos_logits[pos_idx], neg_logits[neg_idx])
            if triplet_samples.shape[0] == 0:
                triplet = torch.tensor(0.0, device=train_logits.device)
            else:
                anchor_idx = triplet_samples[:, 0]
                pos_d_idx = triplet_samples[:, 1]
                neg_d_idx = triplet_samples[:, 2]
                anchor = mic_emb[anchor_idx]
                positive = dis_emb[pos_d_idx]
                negative = dis_emb[neg_d_idx]
                triplet = triplet_loss_fn(anchor, positive, negative, margin=0.5)
            l1_loss = torch.tensor(0.0, device=train_logits.device)
            for name, param in model.named_parameters():
                if 'spline_weight' in name:
                    l1_loss = l1_loss + torch.norm(param, 1)

            loss = bce_weight * bce + bpr_weight * bpr + triplet_weight * triplet + l1_lambda * l1_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            do_eval = (epoch + 1) % eval_interval == 0 or epoch == 0
            if do_eval:
                model.eval()
                with torch.inference_mode():
                    val_img = img_feat_all[val_idx] if use_img else None
                    val_logits, _, _ = model(
                        mic_feat, dis_feat, adj_mic, adj_dis, adj_md, mic_n, dis_n,
                        pairs=val_pairs, img_feat=val_img
                    )
                    val_prob = torch.sigmoid(val_logits).cpu().numpy()
                    y_true = val_y.cpu().numpy()
                    y_pred = (val_prob > threshold).astype(int)
                    val_auc = roc_auc_score(y_true, val_prob)
                    val_aupr = average_precision_score(y_true, val_prob)
                    val_acc = accuracy_score(y_true, y_pred)
                    val_pre = precision_score(y_true, y_pred, zero_division=0)
                    val_recall = recall_score(y_true, y_pred, zero_division=0)
                    val_f1 = f1_score(y_true, y_pred, zero_division=0)
                last_val_auc, last_val_aupr = val_auc, val_aupr
                last_val_acc, last_val_pre, last_val_f1 = val_acc, val_pre, val_f1
                last_val_recall = val_recall
                if metric_for_early_stop == "auc":
                    cur_metric = val_auc
                    best_metric = best_auc
                else:
                    cur_metric = val_aupr
                    best_metric = best_aupr
                if cur_metric > best_metric + min_delta:
                    best_auc = val_auc
                    best_aupr = val_aupr
                    best_acc = val_acc
                    best_pre = val_pre
                    best_f1 = val_f1
                    best_recall = val_recall
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    if early_stop:
                        tqdm.write(
                            f"  ✓ Epoch {epoch + 1}: new best {metric_for_early_stop.upper()}="
                            f"{cur_metric:.4f}"
                        )
                else:
                    epochs_no_improve += 1
                if early_stop and epochs_no_improve >= patience:
                    tqdm.write(
                        f"  ⏹️  Early stopping at epoch {epoch + 1}: "
                        f"{metric_for_early_stop} not improved for {patience} validations "
                        f"(best={best_metric:.4f})"
                    )
                    break
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "triplet": f"{triplet.item():.4f}",
                "AUC": f"{last_val_auc:.4f}",
                "AUPR": f"{last_val_aupr:.4f}"
            })
            if (epoch + 1) % 10 == 0:
                tqdm.write(
                    f"  Epoch {epoch + 1:3d} | "
                    f"AUC: {last_val_auc:.4f} | AUPR: {last_val_aupr:.4f} | "
                    f"Acc: {last_val_acc:.4f} | Pre: {last_val_pre:.4f} | F1: {last_val_f1:.4f}"
                )
                epoch_log["epoch"].append(epoch + 1)
                epoch_log["total_loss"].append(loss.item())
                epoch_log["bce_loss"].append(bce.item())
                epoch_log["bpr_loss"].append(bpr.item())
                epoch_log["triplet_loss"].append(triplet.item())
                epoch_log["l1_loss"].append(l1_loss.item())
                epoch_log["auc"].append(last_val_auc)
                epoch_log["aupr"].append(last_val_aupr)
                epoch_log["acc"].append(last_val_acc)
                epoch_log["pre"].append(last_val_pre)
                epoch_log["recall"].append(last_val_recall)
                epoch_log["f1"].append(last_val_f1)
        pbar.close()
        if best_state is not None:
            model.load_state_dict(best_state)
            tqdm.write(
                f"  ✅ 已恢复最佳模型权重 "
                f"(AUPR={best_aupr:.4f}, AUC={best_auc:.4f})"
            )
        model.eval()
        with torch.inference_mode():
            val_img_final = img_feat_all[val_idx] if use_img else None
            logits_final, _, _ = model(
                mic_feat, dis_feat, adj_mic, adj_dis, adj_md, mic_n, dis_n,
                pairs=val_pairs, img_feat=val_img_final
            )
            val_prob_final = torch.sigmoid(logits_final).cpu().numpy()
            y_true_final = val_y.cpu().numpy()
            y_pred_final = (val_prob_final > threshold).astype(int)
            final_auc = roc_auc_score(y_true_final, val_prob_final)
            final_aupr = average_precision_score(y_true_final, val_prob_final)
            final_acc = accuracy_score(y_true_final, y_pred_final)
            final_pre = precision_score(y_true_final, y_pred_final, zero_division=0)
            final_recall = recall_score(y_true_final, y_pred_final, zero_division=0)
            final_f1 = f1_score(y_true_final, y_pred_final, zero_division=0)
        best_auc, best_aupr = final_auc, final_aupr
        best_acc, best_pre, best_f1 = final_acc, final_pre, final_f1
        best_recall = final_recall

        if best_auc > best_global_auc:
            best_global_auc = best_auc
            best_global_state = copy.deepcopy(model.state_dict())

        # 保存折日志
        df_log = pd.DataFrame(epoch_log)
        fpr, tpr, _ = roc_curve(y_true_final, val_prob_final)
        pr_prec, pr_rec, _ = precision_recall_curve(y_true_final, val_prob_final)
        if len(pr_prec) > 1:
            f1s = 2 * pr_prec[:-1] * pr_rec[:-1] / (pr_prec[:-1] + pr_rec[:-1] + 1e-12)
            best_thr = float(_[int(np.argmax(f1s))])
        else:
            best_thr = 0.55
        if log_out_dir is not None:
            df_log.to_csv(os.path.join(log_out_dir, f"fold{fid}_epoch_log.csv"), index=False)
            np.save(os.path.join(log_out_dir, f"fold{fid}_y_true.npy"), y_true_final)
            np.save(os.path.join(log_out_dir, f"fold{fid}_y_score.npy"), val_prob_final)
            np.savez(os.path.join(log_out_dir, f"fold{fid}_roc_curve.npz"),
                     fpr=fpr, tpr=tpr, auc=final_auc)
            np.savez(os.path.join(log_out_dir, f"fold{fid}_pr_curve.npz"),
                     precision=pr_prec, recall=pr_rec, aupr=final_aupr)
            pd.DataFrame([{"fold": fid, "best_epoch": best_epoch,
                           "best_val_f1_threshold": round(best_thr, 4)}]
                         ).to_csv(os.path.join(log_out_dir, f"fold{fid}_info.csv"), index=False)
        else:
            df_log.to_csv(f"fold{fid}_epoch_log.csv", index=False)
            np.save(f"fold{fid}_y_true.npy", y_true_final)
            np.save(f"fold{fid}_y_score.npy", val_prob_final)
            np.savez(f"fold{fid}_roc_curve.npz", fpr=fpr, tpr=tpr, auc=final_auc)
            np.savez(f"fold{fid}_pr_curve.npz", precision=pr_prec, recall=pr_rec, aupr=final_aupr)

        fold_results["fold"].append(fid)
        fold_results['auc'].append(best_auc)
        fold_results['aupr'].append(best_aupr)
        fold_results['acc'].append(best_acc)
        fold_results['precision'].append(best_pre)
        fold_results['recall'].append(best_recall)
        fold_results['f1'].append(best_f1)

        if top1_eval:
            tp1, tr1, tf1 = eval_top1_hit(
                model, mic_feat, dis_feat, adj_mic, adj_dis, adj_md,
                mic_n, dis_n, train_pos_edges, val_pairs, val_y, device,
                save_cand_dir=log_out_dir, fold_id=fid)
            fold_results['top1_precision'].append(tp1)
            fold_results['top1_recall'].append(tr1)
            fold_results['top1_f1'].append(tf1)
            print(f"  Top-1: Precision {tp1:.4f} | Recall {tr1:.4f} | F1 {tf1:.4f}")
        print(f"\nFold {fid} 最终结果:")
        print(f"  AUC:       {best_auc:.4f}")
        print(f"  AUPR:      {best_aupr:.4f}")
        print(f"  Accuracy:  {best_acc:.4f}")
        print(f"  Precision: {best_pre:.4f}")
        print(f"  Recall:    {best_recall:.4f}")
        print(f"  F1-score:  {best_f1:.4f}")

    if save_best_path is not None and best_global_state is not None:
        torch.save(best_global_state, save_best_path)
        print(f"\n✅ 全局最佳模型已保存至: {save_best_path}")
        print(f"   最佳AUC: {best_global_auc:.4f}")

    df_summary = pd.DataFrame(fold_results)
    if log_out_dir is not None:
        df_summary.to_csv(os.path.join(log_out_dir, "summary_result.csv"), index=False)
    else:
        df_summary.to_csv("summary_result.csv", index=False)
    config_record["runtime_seconds"] = round(time.time() - t_start, 2)
    if log_out_dir is not None:
        with open(os.path.join(log_out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_record, f, indent=2, ensure_ascii=False)
    print(f"\n>> 汇总指标已保存")

    mean_auc = np.mean(fold_results['auc'])
    mean_aupr = np.mean(fold_results['aupr'])
    return mean_auc, mean_aupr, fold_results
