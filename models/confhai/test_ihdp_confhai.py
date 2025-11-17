import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import Tensor
import pandas as pd
from models import *
from utils import *
import argparse
import math
from ihdp import IHDP, policy_value
from sklearn.linear_model import LogisticRegression

GAMMAS = {
    "0.0": float(0.0),
    "0.1": float(0.1),
    "0.2": float(0.2),
    "0.5": float(0.5),
    "0.7": float(0.7),
    "1.0": float(1.0),
    "1.2": float(1.2),
    "1.5": float(1.5),
    "2.0": float(2.0),
    "2.5": float(2.5),
    "3.0": float(3.0),
    "3.5": float(3.5),
    "4.0": float(4.0),
    # "4.5": math.exp(4.5),
    # "5.0": math.exp(5.0),
    # "5.5": math.exp(5.5),
    # "6.0": math.exp(6.0),
    # "6.5": math.exp(6.5),
    # "7.0": math.exp(7.0),
    # "7.5": math.exp(7.5),
    # "8.0": math.exp(8.0),
    # "8.5": math.exp(8.5),
    # "9.0": math.exp(9.0),
    # "9.5": math.exp(9.5),
    # "10.": math.exp(10.0),
}



def estimate_propensity_scores(X_train, T_train):
    """
    使用逻辑回归训练一个倾向得分模型。
    Trains a propensity score model using Logistic Regression.

    Args:
        X_train (np.array): 训练集的协变量 (Covariates for the training set).
        T_train (np.array): 训练集的处理决策 (Treatment decisions for the training set).

    Returns:
        model: 训练好的逻辑回归模型 (The trained Logistic Regression model).
    """
    print("Estimating propensity scores using Logistic Regression...")
    # 初始化并训练逻辑回归模型
    # Initialize and train the Logistic Regression model
    propensity_model = LogisticRegression(solver='liblinear', C=1.0)
    propensity_model.fit(X_train, T_train)
    print("Propensity score model training complete.")
    return propensity_model


def run_ihdp_experiment():
    # ==============================================================================
    # 1. 设置实验参数 (Setup Experiment Parameters)
    # ==============================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--nepoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    # 对于 IHDP，我们通常只有一个专家群体，因此 gamma 设为一个值
    # For IHDP, we typically have one expert population, so gamma is a single value
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--start_trial", type=int, default=0)
    parser.add_argument("--C", type=float, default=0.0)
    parser.add_argument("--nrep", type=int, default=10)  # 建议从少量 trial 开始测试
    parser.add_argument("--folder", type=str, default='ihdp_exp1')
    parser.add_argument("--hidd", type=int, default=64)  # 隐藏层大小

    args = parser.parse_args()
    nepoch = args.nepoch
    lr = args.lr
    # 将单个 gamma 值包装成数组，以兼容原代码逻辑
    # Wrap the single gamma value in an array to be compatible with original logic
    usedGamma = np.array([args.gamma])
    c = args.C
    start_trial = args.start_trial
    nrep = args.nrep
    folder_name = args.folder
    hidd = args.hidd

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建结果保存路径
    # Create path for saving results
    path = os.path.join('./log', folder_name)
    os.makedirs(path, exist_ok=True)

    # ==============================================================================
    # 2. 主实验循环 (Main Experiment Loop)
    # ==============================================================================
    result_list = []
    for i in range(start_trial, nrep):
        print(f"\n{'=' * 20} TRIAL {i}/{nrep} {'=' * 20}")
        seed_everything(i)

        # --- 数据加载 ---
        # --- Data Loading ---
        # 加载 IHDP 数据集，hidden_confounding=False 表示使用完整的观察变量
        # Load IHDP dataset, hidden_confounding=False means using all observed covariates
        train_ds = IHDP(root=None, split="train", mode='mu', seed=i, hidden_confounding=False)
        test_ds = IHDP(root=None, split="test", mode='mu', seed=i, hidden_confounding=False)

        # 准备训练数据
        # Prepare training data
        x_train = Tensor(train_ds.x).to(device)
        t_train = Tensor(train_ds.t).to(device)
        y_train = Tensor(-train_ds.y).to(device)
        # 估算的名义倾向分数 (这里用一个简单的模型代替，实际中可以更复杂)
        # Estimated nominal propensity scores (using a simple model here, can be more complex)
        # q0_train = torch.full_like(t_train, 0.5)
        # p_treatment = torch.full_like(t_train, 0.5)  # Propensity for T=1 is 0.5
        # p_control = 1 - p_treatment  # Propensity for T=0 is 0.5

        var_CATE_trial = np.sqrt(np.var(test_ds.mu1 - test_ds.mu0))
        if var_CATE_trial > 15:
            continue

        # --- Use a trained model to estimate propensity scores ---

        # 1. 训练倾向得分模型
        #    注意：模型训练需要 NumPy 数组，所以我们使用 .cpu().numpy()
        #    Note: Model training requires NumPy arrays, so we use .cpu().numpy()
        propensity_model = estimate_propensity_scores(train_ds.x, train_ds.t)

        # 2. 使用训练好的模型来预测倾向得分
        #    predict_proba 返回一个 (N, 2) 的数组，分别对应 T=0 和 T=1 的概率
        #    predict_proba returns an (N, 2) array with probabilities for T=0 and T=1
        propensity_scores_train = propensity_model.predict_proba(train_ds.x)

        # 3. 将 NumPy 数组转换为 PyTorch Tensor 并移动到正确的设备
        #    Convert the NumPy array to a PyTorch Tensor and move to the correct device
        q0_train = torch.from_numpy(propensity_scores_train).float().to(device)


        # 准备测试数据
        # Prepare test data
        x_test = Tensor(test_ds.x).to(device)
        # 真实的潜在结果用于最终评估
        # True potential outcomes for final evaluation
        y0_test = test_ds.y0
        y1_test = test_ds.y1

        # 创建 DataLoader
        # Create DataLoader
        train_dataset = TensorDataset(x_train, q0_train, t_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        # 获取输入维度
        # Get input dimension
        d = x_train.shape[1]

        # --- 模型训练与评估 ---
        # --- Model Training and Evaluation ---

        # 1. 纯算法 (Algorithm Only - AO)
        model_ao = train_ips(d, 2, train_loader, nepoch, lr=lr, hidd=hidd)
        model_ao.to(device)
        pi_ao = torch.softmax(model_ao(x_test), dim=1)[:, 1].cpu().detach().numpy()
        risk_ao = policy_value(pi=pi_ao, y1=y1_test, y0=y0_test)
        print(f"Risk - Algorithm Only (AO): {risk_ao:.4f}")

        for k_log_gamma, v_gamma_val in GAMMAS.items():


            # 2. 混淆鲁棒性算法 (Confounding-Robust AO - ConfAO)
            # 在单专家场景下，hid 始终为0
            # In a single-expert scenario, hid is always 0
            hid_train = torch.zeros(len(x_train), dtype=torch.long)
            # model_confao, _ = train_confips(d, 2, train_loader, nepoch, lr=lr, gamma=usedGamma[hid_train], hidd=hidd)
            model_confao, _ = train_confips(d, 2, train_loader, nepoch, lr=lr, gamma=v_gamma_val, hidd=hidd)
            model_confao.to(device)
            pi_confao = torch.softmax(model_confao(x_test), dim=1)[:, 1].cpu().detach().numpy()
            risk_confao = policy_value(pi=pi_confao, y1=y1_test, y0=y0_test)
            print(f"Risk - Confounding-Robust AO (ConfAO): {risk_confao:.4f}")

            import copy
            modelconf = copy.deepcopy(model_confao)

            # 3. 标准人机协作 (Human-AI Team - HAI)
            # 注意: HAI 的 router 需要预训练或共同训练，这里简化流程
            # Note: HAI's router needs pre-training or joint training, process is simplified here
            model_hai, router_hai = train_hai(modelconf, d, 2, train_loader, nepoch, lr=lr, C=c, hidd=hidd)
            model_hai.to(device)
            router_hai.to(device)
            # router 输出一个 logit，通过 sigmoid 得到延缓概率
            pi_router_hai = torch.sigmoid(router_hai(x_test)).cpu().detach().numpy().flatten()  # Defer prob
            pi_model_hai = torch.softmax(model_hai(x_test), dim=1)[:, 1].cpu().detach().numpy()
            pi_final_hai = pi_router_hai * test_ds.t + (1 - pi_router_hai) * pi_model_hai
            risk_hai = policy_value(pi=pi_final_hai, y1=y1_test, y0=y0_test)
            print(f"Risk - Human-AI Team (HAI): {risk_hai:.4f}")

            deferral_decisions_hai = (pi_router_hai > 0.5)
            num_deferred_hai = deferral_decisions_hai.sum()
            total_samples_hai = len(pi_router_hai)
            defer_rate_hai = num_deferred_hai / total_samples_hai

            print(f"Deferral Rate - HAI: {defer_rate_hai:.4f} ({num_deferred_hai} / {total_samples_hai} samples deferred)")

            # =============================================================================

            pi_model_hai = torch.softmax(model_hai(x_test), dim=1)[:, 1].cpu().detach().numpy()
            # 注意：这里的 pi_router_hai 应该用作权重，而不是决策
            pi_final_hai = pi_router_hai * test_ds.t + (1 - pi_router_hai) * pi_model_hai
            risk_hai = policy_value(pi=pi_final_hai, y1=y1_test, y0=y0_test)
            print(f"Risk - Human-AI Team (HAI): {risk_hai:.4f}")

            # 4. 混淆鲁棒性人机协作 (Confounding-Robust HAI - ConfHAI)
            # model_confhai, router_confhai, _ = train_confhai(modelconf, router_hai, d, 2, train_loader, nepoch, lr=lr,
            #                                                  gamma=usedGamma[hid_train], C=c, hidd=hidd)
            model_confhai, router_confhai, _ = train_confhai(modelconf, router_hai, d, 2, train_loader, nepoch, lr=lr,
                                                             gamma=v_gamma_val, C=c, hidd=hidd)
            model_confhai.to(device)
            router_confhai.to(device)
            # router 输出一个 logit，通过 sigmoid 得到延缓概率
            pi_router_confhai = torch.sigmoid(router_confhai(x_test)).cpu().detach().numpy().flatten()  # Defer prob
            pi_model_confhai = torch.softmax(model_confhai(x_test), dim=1)[:, 1].cpu().detach().numpy()
            pi_final_confhai = pi_router_confhai * test_ds.t + (1 - pi_router_confhai) * pi_model_confhai
            risk_confhai = policy_value(pi=pi_final_confhai, y1=y1_test, y0=y0_test)
            print(f"Risk - Confounding-Robust HAI (ConfHAI): {risk_confhai:.4f}")

            avg_defer_prob = pi_router_confhai.mean()
            print(f"Average Deferral Probability: {avg_defer_prob:.4f}")

            deferral_decisions_confhai = (pi_router_confhai > 0.5)
            num_deferred_confhai = deferral_decisions_confhai.sum()
            total_samples_confhai = len(pi_router_confhai)
            defer_rate_hai = num_deferred_confhai / total_samples_confhai

            print(f"Deferral Rate - HAI: {defer_rate_hai:.4f} ({num_deferred_confhai} / {total_samples_confhai} samples deferred)")

            # 计算基准策略的风险
            # Calculate risk for baseline policies
            risk_human = policy_value(pi=test_ds.t, y1=y1_test, y0=y0_test)
            oracle_policy = (y1_test > y0_test).astype(float)
            risk_oracle = policy_value(pi=oracle_policy, y1=y1_test, y0=y0_test)

            # 保存本次 trial 的结果
            # Save results for this trial
            result_list.append({
                'trial': i,
                'log_gamma': v_gamma_val,
                'Human': risk_human,
                'Oracle': risk_oracle,
                'AO': risk_ao,
                'ConfAO': risk_confao,
                'HAI': risk_hai,
                'ConfHAI': risk_confhai,
                'avg_defer_prob': avg_defer_prob,
                'defer_rate_hai': defer_rate_hai,
            })

    # ==============================================================================
    # 3. 保存并显示结果 (Save and Display Results)
    # ==============================================================================
    result_df = pd.DataFrame(result_list)
    output_path = os.path.join(path, f'ihdp_confai_{start_trial}_{nrep}_results.csv')
    result_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 20} FINAL RESULTS {'=' * 20}")
    print(f"All trial results saved to: {output_path}")

    # 打印均值和标准误
    # Print mean and standard error
    print("\n--- Average Policy Value (Higher is Better) ---")
    print(result_df.mean())
    print("\n--- Standard Error of the Mean ---")
    print(result_df.std() / np.sqrt(nrep))


if __name__ == '__main__':
    run_ihdp_experiment()