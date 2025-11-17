import os
import json
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from models.crlogit.subgrad import get_implicit_grad_centered, opt_wrapper, opt_w_restarts, qk_dpi_dtheta, logistic_pol_asgn
from models.crlogit.methods import ConfoundingRobustPolicy


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def policy_value(pi, y1, y0):
    pi_tensor = torch.as_tensor(pi, dtype=torch.float)
    y1_tensor = torch.as_tensor(y1, dtype=torch.float)
    y0_tensor = torch.as_tensor(y0, dtype=torch.float)
    return (pi_tensor * y1_tensor + (1 - pi_tensor) * y0_tensor).mean()


def update_expert_preds(preds, expert_labels):
    defer_count = 0
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds, dtype=torch.long)

    updated_preds = preds.clone().long()

    if not isinstance(expert_labels, torch.Tensor):
        expert_labels_tensor = torch.tensor(expert_labels, dtype=torch.long)
    else:
        expert_labels_tensor = expert_labels.long()

    for j in range(len(preds)):
        if updated_preds[j].item() == 2:
            defer_count += 1
            updated_preds[j] = expert_labels_tensor[j]

    deferral_rate = (defer_count / len(preds)) if len(preds) > 0 else 0.0
    return updated_preds, defer_count, deferral_rate


def update_expert(ds, feature):
    tau_true = torch.tensor(ds.mu1 - ds.mu0)
    oracle_pi = (tau_true > 0).int()
    cond_train = np.asarray(ds.df[feature])
    for i in range(len(ds.t)):
        if cond_train[i] == 1:
            ds.t[i] = oracle_pi[i].item()
            if oracle_pi[i] == 0:
                ds.y[i] = ds.y0[i]
            else:
                ds.y[i] = ds.y1[i]
    return ds


def get_expert_by_feature(ds, feature):
    expert_t = ds.t
    expert_y = ds.y
    return expert_t, expert_y


def save_policies(policies_dict: dict, file_path: Path):
    serializable_policies = {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v for k, v in
                             policies_dict.items()}
    with file_path.open(mode="w") as fp:
        json.dump(serializable_policies, fp, indent=4)


def load_policies(file_path: Path):
    with file_path.open(mode="r") as fp:
        policies = json.load(fp)
    return {k: torch.tensor(v) for k, v in policies.items()}


def real_risk_prob_ihdp(prob_1, Y1, Y0):
    prob_1 = np.asarray(prob_1)
    return prob_1 * Y1 + (1 - prob_1) * Y0


def ihdp_q0_baseline_p(x_train, t_train, x):
    propensity_model = LogisticRegression(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=10000,
                                          random_state=42)
    propensity_model.fit(X=x_train, y=t_train)
    return propensity_model.predict_proba(x)[:, [1]]


def default_policy(q_0, p):
    return q_0


def run_crlogit_policy(ihdp_train_ds, ihdp_val_ds, ihdp_test_ds, GAMMAS: dict):
    """
    运行 CRLogit 策略并返回所有 gamma 对应的 Policy Value 列表。
    """
    print("--- Running CRLogit Policy ---")

    # 准备 CRLogit 需要的各种数据和参数
    GAMS = np.fromiter(GAMMAS.keys(), dtype=float)
    x_train_ = np.hstack([ihdp_train_ds.x, np.ones([ihdp_train_ds.x.shape[0], 1])])
    x_test_ = np.hstack([ihdp_test_ds.x, np.ones([ihdp_test_ds.x.shape[0], 1])])

    q_0_train_ = ihdp_q0_baseline_p(x_train=x_train_, t_train=ihdp_train_ds.t, x=x_train_).squeeze(1)

    baseline_policy = lambda x: ihdp_q0_baseline_p(x_train=x_train_, t_train=ihdp_train_ds.t, x=x)
    def_policy = lambda p: default_policy(q_0=q_0_train_, p=p)
    real_risk_ihdp = lambda p1: real_risk_prob_ihdp(Y1=(-1) * ihdp_test_ds.y1, Y0=(-1) * ihdp_test_ds.y0, prob_1=p1)

    x_all = np.concatenate([ihdp_train_ds.x, ihdp_val_ds.x, ihdp_test_ds.x])
    q_0_test = ihdp_q0_baseline_p(x_train=ihdp_train_ds.x, t_train=ihdp_train_ds.t, x=ihdp_test_ds.x).squeeze(1)
    logit_model = sm.GLM(q_0_test, x_test_, family=sm.families.Binomial())
    logit_result = logit_model.fit()
    th_expert = logit_result.params

    opt_config_robust = {'N_RST': 15, 'GRAD_': get_implicit_grad_centered, 'WGHTS_': opt_wrapper,
                         'GRAD_CTR': get_implicit_grad_centered, 'POL_PROB_1': logistic_pol_asgn,
                         'POL_GRAD': qk_dpi_dtheta, 'DEFAULT_POL': th_expert,
                         'BASELINE_POL': def_policy, 'P_1': def_policy, 'averaging': True,
                         'give_initial': True, 'sharp': True}

    robust_opt_params = {'optimizer': opt_w_restarts, 'pol_opt': 'ogd',
                         'unc_set_type': 'interval', 'opt_params': opt_config_robust,
                         'BASELINE_POL': th_expert, 'type': 'logistic-interval'}

    method_params = [robust_opt_params]

    test_data = {'x_test': x_test_, 't_test': ihdp_test_ds.t, 'y_test': (-1) * ihdp_test_ds.y,
                 'u_test': (ihdp_test_ds.u).squeeze(1)}
    eval_conf = {'eval': True, 'eval_type': 'ihdp', 'eval_data': test_data, 'oracle_risk': real_risk_ihdp}

    # 注意：v1.0 的代码只支持一个 method_param，所以我们直接用 [0]
    conf_rob_pol = ConfoundingRobustPolicy(baseline_pol=baseline_policy, save=False, verbose=True)
    conf_rob_pol.fit(x_train_, ihdp_train_ds.t, (-1) * ihdp_train_ds.y, q_0_train_, GAMS, method_params[0],
                     eval_conf=eval_conf)

    # CRLogit 的 Policy Value (PVS) 返回的是负值，所以乘以 -1
    pv_cr_logit_list = -1 * conf_rob_pol.PVS

    return pv_cr_logit_list


def plot_defer_distribution(p_defer_probs, bins=30, save_path="defer_figure/p_defer_distribution.png"):
    """
    Plots and saves histogram and density of P(action=2)

    Args:
        p_defer_probs (torch.Tensor or np.ndarray): Vector of deferral probabilities
        bins (int): Number of histogram bins
        save_path (str): Path to save the output image
    """

    # Ensure it's a 1D NumPy array
    if isinstance(p_defer_probs, torch.Tensor):
        p_defer_probs = p_defer_probs.detach().cpu().numpy()
    elif isinstance(p_defer_probs, list):
        p_defer_probs = np.array(p_defer_probs)
    elif isinstance(p_defer_probs, float):
        p_defer_probs = np.array([p_defer_probs])

    # Make sure it's 1D
    if p_defer_probs.ndim > 1:
        p_defer_probs = p_defer_probs.flatten()

    # Plot
    plt.figure(figsize=(8, 4))
    sns.histplot(p_defer_probs, bins=bins, kde=True, color='skyblue')
    plt.title("Distribution of P(action=Defer)")
    plt.xlabel("P(action=2)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"[Saved] Defer probability distribution plot → {save_path}")
