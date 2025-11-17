# =============================================================================
#           文件: compute_blearner_bounds_ihdp.py (最终、重构后的版本)
# =============================================================================
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from models.blearner import BLearner
from models.blearner.nuisance import XGBKernel, KernelSuperquantileRegressor
from xgboost import XGBRegressor, XGBClassifier


def train_blearner_for_gamma(ds_train, ds_valid, gamma):
    """
    一个清晰的辅助函数，只负责为单个 gamma 值训练一个 B-Learner 模型并返回。
    """
    X_train, Y_train, A_train = ds_train.x, ds_train.y, ds_train.t
    X_val, Y_val, A_val = ds_valid.x, ds_valid.y, ds_valid.t

    xgb_params = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'tree_method': 'hist', 'learning_rate': 0.05,
        'n_estimators': 500, 'max_depth': 3, 'min_child_weight': 5,
    }
    tau = gamma / (1.0 + gamma)

    propensity_model = LogisticRegression(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=10000)
    mu_model = XGBRegressor(**xgb_params)
    cate_bounds_model = XGBRegressor(**xgb_params)
    quantile_model_upper = XGBRegressor(**xgb_params, objective='reg:quantileerror', quantile_alpha=tau)
    quantile_model_lower = XGBRegressor(**xgb_params, objective='reg:quantileerror', quantile_alpha=1 - tau)

    kernel_model_config = XGBKernel(
        XGBRegressor(**{k: v for k, v in xgb_params.items() if k not in ['objective', 'quantile_alpha']}))
    cvar_model_upper = KernelSuperquantileRegressor(kernel=kernel_model_config, tau=tau, tail="right")
    cvar_model_lower = KernelSuperquantileRegressor(kernel=kernel_model_config, tau=1 - tau, tail="left")

    cate_bounds_est = BLearner(
        propensity_model=propensity_model, mu_model=mu_model,
        quantile_plus_model=quantile_model_upper, quantile_minus_model=quantile_model_lower,
        cvar_plus_model=cvar_model_upper, cvar_minus_model=cvar_model_lower,
        cate_bounds_model=cate_bounds_model, use_rho=True, gamma=gamma)

    cate_bounds_est.fit(X=X_train, A=A_train, Y=Y_train, X_val=X_val, A_val=A_val, Y_val=Y_val)

    return cate_bounds_est


def compute_all_intervals_BLearner(dir_path, trial, ds_train, ds_valid, ds_test, GAMMAS):
    """
    主调用函数：为所有的 gamma 值计算 B-Learner 边界。
    """
    intervals_rf = {}

    for k_gamma, v_gamma in GAMMAS.items():
        print(f"  - Computing B-Learner for gamma={k_gamma}...")

        # 1. 为当前 gamma 训练一个 B-Learner 模型
        fitted_blearner = train_blearner_for_gamma(ds_train, ds_valid, v_gamma)

        # 2. 使用训练好的模型，分别在 train/val/test 数据集上进行预测
        def get_all_bounds(dataset):
            X = dataset.x

            def to_1d_array(arr): return np.asarray(arr).flatten()

            tau_bottom, tau_top = map(to_1d_array, fitted_blearner.effect(X))
            Y_0b, Y_0t, Y_1b, Y_1t = map(to_1d_array, fitted_blearner.outcome_bounds(X))
            tau_mean = to_1d_array(fitted_blearner.mu1(X) - fitted_blearner.mu0(X))

            return {
                "tau_mean": tau_mean, "tau_bottom": tau_bottom, "tau_top": tau_top,
                "Y_0_bottom": Y_0b, "Y_0_top": Y_0t,
                "Y_1_bottom": Y_1b, "Y_1_top": Y_1t,
            }

        train_bounds = get_all_bounds(ds_train)
        val_bounds = get_all_bounds(ds_valid)
        test_bounds = get_all_bounds(ds_test)

        # 3. 将所有结果组合成与 v1.0 兼容的字典格式
        tau_hat = {}
        for split_name, bounds_dict in [("train", train_bounds), ("val", val_bounds), ("test", test_bounds)]:
            for bound_name, values in bounds_dict.items():
                tau_hat[f"{bound_name}_{split_name}"] = values

        intervals_rf[k_gamma] = tau_hat

    return intervals_rf