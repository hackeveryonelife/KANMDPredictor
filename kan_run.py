import os
import warnings
warnings.filterwarnings("ignore", message="CUDA initialization")

import numpy as np
from kan_train import train_kan_model
from gpu_utils import get_device
from ablation_run import load_data

base_train_data, image_data = load_data()

print(f"节点规模: miRNA={base_train_data['mic_n']}, disease={base_train_data['dis_n']}, "
      f"总节点={base_train_data['mic_n'] + base_train_data['dis_n']}")
print(f"平衡总样本数: {len(base_train_data['sample_pairs'])}, "
      f"正样本: {int(base_train_data['sample_labels'].sum())}, "
      f"负样本: {len(base_train_data['sample_labels']) - int(base_train_data['sample_labels'].sum())}")

device = get_device()
if __name__ == "__main__":
    avg_auc, avg_aupr = train_kan_model(
        mic_sim_gsm=base_train_data["mic_sim_gsm"],
        dis_sim_gsm=base_train_data["dis_sim_gsm"],
        sample_pairs=base_train_data["sample_pairs"],
        sample_labels=base_train_data["sample_labels"],
        mic_n=base_train_data["mic_n"],
        dis_n=base_train_data["dis_n"],
        device=device,
        epochs=200,
        lr=0.0003,  # 调小学习率配合AdamW
        kan_grid=7,
        kan_order=3,
        use_img=image_data["default_use_img"],
        img_dim=image_data["default_img_dim"],
        img_feat=image_data["img_feat"],
        lightweight=True,
        early_stop=True,
        patience=20,
        min_delta=1e-4,
        metric_for_early_stop="auc",
        eval_interval=5
    )
    print("\n实验完成！")
    print(f"5折平均 AUC: {avg_auc:.4f}")
    print(f"5折平均 AUPR: {avg_aupr:.4f}")
