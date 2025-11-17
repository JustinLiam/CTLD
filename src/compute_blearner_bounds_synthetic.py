import json
import os
from pathlib import Path
from joblib import Parallel, delayed
import copy
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
# from econml.grf import RegressionForest
from sklearn_quantile import RandomForestQuantileRegressor

from datasets.ihdp import IHDP
from models.blearner.nuisance import (
    RFKernel, KernelSuperquantileRegressor,
    KernelQuantileRegressor,XGBKernel
)

from sklearn.model_selection import GridSearchCV


from models.blearner import BLearner
from models.crlogit.data_scenarios import *
from sklearn import model_selection
from datasets.synthetic_log import SYN_LOG

GAMMAS = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
from xgboost import XGBRegressor, XGBClassifier


# def save_intervals(intervals, file_path):
#     tmp_intervals = intervals
#     for k, v in tmp_intervals.items():
#         for k1, v2 in v.items():
#             tmp_intervals[k][k1] = (v2.numpy()).tolist()
#     with file_path.open(mode="w") as fp:
#         json.dump(tmp_intervals, fp)

def save_intervals(intervals, file_path):
    tmp_intervals = copy.deepcopy(intervals)
    for k, v in tmp_intervals.items():
        for k1, v2 in v.items():
            if isinstance(v2, torch.Tensor):
                tmp_intervals[k][k1] = v2.numpy().tolist()
            elif isinstance(v2, np.ndarray):
                tmp_intervals[k][k1] = v2.tolist()
    with file_path.open(mode="w") as fp:
        json.dump(tmp_intervals, fp)


def load_intervals(file_path):
    with file_path.open(mode="r") as fp:
        intervals = json.load(fp)
    for k, v in intervals.items():
        for k1, v2 in v.items():
            intervals[k][k1] = torch.Tensor(v2)
    return intervals


def compute_intervals_BLearner(ds_train, ds_test, gamma, ds_valid=None):
    X_train = ds_train.x
    Y_train = ds_train.y
    A_train = ds_train.t

    X_val = ds_valid.x
    Y_val = ds_valid.y
    A_val = ds_valid.t

    X_test = ds_test.x
    A_test = ds_test.t

    # n_estimators = 20
    # max_depth = 3
    # max_features = "sqrt"
    # min_samples_leaf = 6
    # use_rho = True
    #
    # # Train model
    # tau = gamma / (1 + gamma)
    # propensity_model = LogisticRegression(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=100)
    # # Mu model
    # mu_model = RandomForestRegressor(
    #     n_estimators=n_estimators,
    #     max_depth=max_depth,
    #     # max_features=max_features,
    #     min_samples_leaf=min_samples_leaf,
    #     n_jobs=-2)
    #
    # # Quantile and CVaR models
    # # Models for the tau quantile and cvar
    # quantile_model_upper = RandomForestQuantileRegressor(n_estimators=n_estimators,
    #                                                      max_depth=max_depth,
    #                                                      # max_features=max_features,
    #                                                      min_samples_leaf=min_samples_leaf,
    #                                                      n_jobs=-2,
    #                                                      q=tau)
    # # quantile_model_upper = KernelQuantileRegressor(
    # #     kernel=RFKernel(
    # #         RegressionForest(
    # #             n_estimators=n_estimators,
    # #             max_depth=max_depth,
    # #             max_features=max_features,
    # #             min_samples_leaf=min_samples_leaf,
    # #             n_jobs=-2)
    # #     ),
    # #     tau=tau
    # # )
    # cvar_model_upper = KernelSuperquantileRegressor(
    #     kernel=RFKernel(
    #         RandomForestRegressor(
    #             n_estimators=n_estimators,
    #             max_depth=max_depth,
    #             # max_features=max_features,
    #             min_samples_leaf=min_samples_leaf,
    #             n_jobs=-2)
    #     ),
    #     tau=tau,
    #     tail="right")
    #
    # # Models for the 1-tau quantile and cvar
    # quantile_model_lower = RandomForestQuantileRegressor(n_estimators=n_estimators,
    #                                                      max_depth=max_depth,
    #                                                      # max_features=max_features,
    #                                                      min_samples_leaf=min_samples_leaf,
    #                                                      n_jobs=-2,
    #                                                      q=1 - tau)
    # # quantile_model_lower = KernelQuantileRegressor(
    # #     kernel=RFKernel(
    # #         RegressionForest(
    # #             n_estimators=n_estimators,
    # #             max_depth=max_depth,
    # #             max_features=max_features,
    # #             min_samples_leaf=min_samples_leaf,
    # #             n_jobs=-2)
    # #     ),
    # #     tau=1 - tau
    # # )
    #
    # cvar_model_lower = KernelSuperquantileRegressor(
    #     kernel=RFKernel(
    #         RandomForestRegressor(
    #             n_estimators=n_estimators,
    #             max_depth=max_depth,
    #             # max_features=max_features,
    #             min_samples_leaf=min_samples_leaf,
    #             n_jobs=-2)
    #     ),
    #     tau=1 - tau,
    #     tail="left")
    # cate_bounds_model = RegressionForest(
    #     n_estimators=n_estimators,
    #     max_depth=max_depth,
    #     min_samples_leaf=min_samples_leaf,
    #     n_jobs=-2)
    n_estimators = 500
    max_depth = 6  # 6
    max_features = "sqrt"
    min_samples_leaf = 15
    use_rho = True

    learning_rate = 0.1  # 0.01
    min_child_weight = 3  # 1
    max_depth = 5
    n_estimators = 200



    # xgb_model = XGBRegressor(random_state=42)
    # param_grid = {
    #     'learning_rate': [0.01, 0.05, 0.1, 0,2, 0.5],
    #     'max_depth': [3, 5, 7, 9, 11],
    #     'n_estimators': [50, 100, 200, 300, 500, 1000],
    #     "min_child_weight": [1, 3, 5, 7]
    # }
    #
    # # Initialize GridSearchCV
    # grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5,
    #                            n_jobs=-1)
    #
    # # Fit GridSearchCV to find the best hyperparameters
    # grid_search.fit(X_train, Y_train)
    #
    # # Get the best hyperparameters
    # best_params = grid_search.best_params_
    # print(best_params)
    #
    # learning_rate = best_params['learning_rate']
    # max_depth = best_params['max_depth']
    # n_estimators = best_params['n_estimators']
    # min_child_weight = best_params['min_child_weight']

    # Train model
    tau = gamma / (1 + gamma)
    propensity_model = LogisticRegression(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=10000)
    # propensity_model = XGBClassifier(
    #     learning_rate=learning_rate,
    #     n_estimators=n_estimators,
    #     max_depth=max_depth,
    #     min_child_weight=min_child_weight,
    # )

    # Mu model
    mu_model = XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
    )

    # mu_model = RandomForestRegressor(
    #     n_estimators=n_estimators,
    #     max_depth=max_depth,
    #     # max_features=max_features,
    #     min_samples_leaf=min_samples_leaf,
    #     n_jobs=-2)

    # Quantile and CVaR models
    # Models for the tau quantile and cvar
    # quantile_model_upper = RandomForestQuantileRegressor(n_estimators=n_estimators,
    #                                                      max_depth=max_depth,
    #                                                      # max_features=max_features,
    #                                                      min_samples_leaf=min_samples_leaf,
    #                                                      n_jobs=-2,
    #                                                      q=tau)
    # quantile_model_upper = KernelQuantileRegressor(
    #     # kernel=RFKernel(
    #     #     RegressionForest(
    #     #         n_estimators=n_estimators,
    #     #         max_depth=max_depth,
    #     #         max_features=max_features,
    #     #         min_samples_leaf=min_samples_leaf,
    #     #         n_jobs=-2)
    #     # )
    #     kernel=XGBRegressor(
    #         learning_rate=learning_rate,
    #         n_estimators=n_estimators,
    #         max_depth=max_depth,
    #         min_child_weight=min_child_weight
    #     ),
    #     tau=tau
    # )
    #
    quantile_model_upper = XGBRegressor(
        objective='reg:quantileerror',
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=1,
        quantile_alpha=tau,
    )
    cvar_model_upper = KernelSuperquantileRegressor(
        # kernel=RFKernel(
        #     RandomForestRegressor(
        #         n_estimators=n_estimators,
        #         max_depth=max_depth,
        #         # max_features=max_features,
        #         min_samples_leaf=min_samples_leaf,
        #         n_jobs=-2)
        # ),
        kernel=XGBKernel(XGBRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight
        )),
        tau=tau,
        tail="right")

    # Models for the 1-tau quantile and cvar
    # quantile_model_lower = RandomForestQuantileRegressor(n_estimators=n_estimators,
    #                                                      max_depth=max_depth,
    #                                                      # max_features=max_features,
    #                                                      min_samples_leaf=min_samples_leaf,
    #                                                      n_jobs=-2,
    #                                                      q=1 - tau)
    # quantile_model_lower = KernelQuantileRegressor(
    #     # kernel=RFKernel(
    #     #     RegressionForest(
    #     #         n_estimators=n_estimators,
    #     #         max_depth=max_depth,
    #     #         max_features=max_features,
    #     #         min_samples_leaf=min_samples_leaf,
    #     #         n_jobs=-2)
    #     # ),
    #     kernel=XGBRegressor(
    #         learning_rate=learning_rate,
    #         n_estimators=n_estimators,
    #         max_depth=max_depth,
    #         min_child_weight=min_child_weight
    #     ),
    #     tau=1 - tau
    # )
    quantile_model_lower = XGBRegressor(
        objective='reg:quantileerror',
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        quantile_alpha=1 - tau
    )

    cvar_model_lower = KernelSuperquantileRegressor(
        # kernel=RFKernel(
        #     RandomForestRegressor(
        #         n_estimators=n_estimators,
        #         max_depth=max_depth,
        #         # max_features=max_features,
        #         min_samples_leaf=min_samples_leaf,
        #         n_jobs=-2)
        # ),
        kernel=
        XGBKernel(XGBRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight
        )),
        tau=1 - tau,
        tail="left")
    # cate_bounds_model = RegressionForest(
    #     n_estimators=n_estimators,
    #     max_depth=max_depth,
    #     min_samples_leaf=min_samples_leaf,
    #     n_jobs=-2)
    cate_bounds_model = XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight
    )
    # CATE bound model
    cate_bounds_est = BLearner(propensity_model=propensity_model,
                               quantile_plus_model=quantile_model_upper,
                               quantile_minus_model=quantile_model_lower,
                               mu_model=mu_model,
                               cvar_plus_model=cvar_model_upper,
                               cvar_minus_model=cvar_model_lower,
                               cate_bounds_model=cate_bounds_model,
                               use_rho=use_rho,
                               gamma=gamma)

    cate_bounds_est.fit(X=X_train, A=A_train, Y=Y_train)

    tau_bottom, tau_top = cate_bounds_est.effect(X_test)
    tau_bottom = tau_bottom.reshape(-1, 1)
    tau_top = tau_top.reshape(-1, 1)
    tau_bottom_test = torch.Tensor((tau_bottom.reshape(-1, 1)).transpose())
    tau_top_test = torch.Tensor((tau_top.reshape(-1, 1)).transpose())

    tau_bottom_train, tau_top_train = cate_bounds_est.effect(X_train)
    tau_bottom_train = torch.Tensor((tau_bottom_train.reshape(-1, 1)).transpose())
    tau_top_train = torch.Tensor((tau_top_train.reshape(-1, 1)).transpose())

    tau_bottom_val, tau_top_val = cate_bounds_est.effect(X_val)
    tau_bottom_val = torch.Tensor((tau_bottom_val.reshape(-1, 1)).transpose())
    tau_top_val = torch.Tensor((tau_top_val.reshape(-1, 1)).transpose())



    tau_mean_train = torch.Tensor(
        ((cate_bounds_est.mu1(X_train) - cate_bounds_est.mu0(X_train)).reshape(-1, 1)).transpose().tolist())
    tau_mean_val = torch.Tensor(
        ((cate_bounds_est.mu1(X_val) - cate_bounds_est.mu0(X_val)).reshape(-1, 1)).transpose().tolist())
    tau_mean_test = torch.Tensor(
        ((cate_bounds_est.mu1(X_test) - cate_bounds_est.mu0(X_test)).reshape(-1, 1)).transpose().tolist())


    Y_0_bottom_train, Y_0_top_train, Y_1_bottom_train, Y_1_top_train = cate_bounds_est.outcome_bounds(X_train)
    Y_0_bottom_val, Y_0_top_val, Y_1_bottom_val, Y_1_top_val = cate_bounds_est.outcome_bounds(X_val)
    Y_0_bottom_test, Y_0_top_test, Y_1_bottom_test, Y_1_top_test = cate_bounds_est.outcome_bounds(X_test)


    Y_0_bottom_train = torch.Tensor((Y_0_bottom_train.reshape(-1, 1)).transpose())
    Y_0_top_train = torch.Tensor((Y_0_top_train.reshape(-1, 1)).transpose())
    Y_1_bottom_train = torch.Tensor((Y_1_bottom_train.reshape(-1, 1)).transpose())
    Y_1_top_train = torch.Tensor((Y_1_top_train.reshape(-1, 1)).transpose())

    Y_0_bottom_val = torch.Tensor((Y_0_bottom_val.reshape(-1, 1)).transpose())
    Y_0_top_val = torch.Tensor((Y_0_top_val.reshape(-1, 1)).transpose())
    Y_1_bottom_val = torch.Tensor((Y_1_bottom_val.reshape(-1, 1)).transpose())
    Y_1_top_val = torch.Tensor((Y_1_top_val.reshape(-1, 1)).transpose())


    Y_0_bottom_test = torch.Tensor((Y_0_bottom_test.reshape(-1, 1)).transpose())
    Y_0_top_test = torch.Tensor((Y_0_top_test.reshape(-1, 1)).transpose())
    Y_1_bottom_test = torch.Tensor((Y_1_bottom_test.reshape(-1, 1)).transpose())
    Y_1_top_test = torch.Tensor((Y_1_top_test.reshape(-1, 1)).transpose())

    tau_hat = {
        "tau_mean_train": tau_mean_train,
        "tau_bottom_train": tau_bottom_train,
        "tau_top_train": tau_top_train,
        "tau_mean_test": tau_mean_test,
        "tau_bottom_test": tau_bottom_test,
        "tau_top_test": tau_top_test,
        "tau_mean_val": tau_mean_val,
        "tau_bottom_val": tau_bottom_val,
        "tau_top_val": tau_top_val,
        "Y_0_bottom_train": Y_0_bottom_train,
        "Y_0_top_train": Y_0_top_train,
        "Y_1_bottom_train": Y_1_bottom_train,
        "Y_1_top_train": Y_1_top_train,
        "Y_0_bottom_test": Y_0_bottom_test,
        "Y_0_top_test": Y_0_top_test,
        "Y_1_bottom_test": Y_1_bottom_test,
        "Y_1_top_test": Y_1_top_test,
        "Y_0_bottom_val": Y_0_bottom_val,
        "Y_0_top_val": Y_0_top_val,
        "Y_1_bottom_val": Y_1_bottom_val,
        "Y_1_top_val": Y_1_top_val
    }

    # return tau_mean_train, tau_bottom_train, tau_top_train, \
    #        tau_mean_test, \
    #        Y_0_bottom_train, Y_0_top_train, Y_1_bottom_train, Y_1_top_train,\
    #        Y_0_bottom_test, Y_0_top_test, Y_1_bottom_test, Y_1_top_test

    return tau_hat


def compute_all_intervals_BLearner(dir_path, trial, ds_train, ds_test, ds_valid=None):
    intervals_rf = {}
    output_dir = dir_path / f"trial-{trial:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "intervals_rf.json"
    if file_path.exists():
        with file_path.open(mode="r") as fp:
            intervals_rf = load_intervals(file_path=file_path)
    else:
        for gamma in GAMMAS:
            print(gamma)
            tau_hat_rf = compute_intervals_BLearner(ds_train=ds_train,
                                                        ds_valid=ds_valid,
                                                        ds_test=ds_test, gamma=np.exp(gamma))
            # intervals_rf.update({gamma: tau_hat_rf})
            intervals_rf.update({str(gamma): tau_hat_rf})
        save_intervals(intervals=intervals_rf, file_path=file_path)

    return intervals_rf
