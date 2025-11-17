from pathlib import Path
import json
import os
from joblib import Parallel, delayed
import math
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

import wandb
from lightning.pytorch.loggers import WandbLogger

from sklearn import model_selection

from src.utils import policy_value
from models.confhai.data import *
from src.compute_blearner_bounds_synthetic import compute_all_intervals_BLearner
from models.lce_surrogate.lce_model import LCEModel
from models.lce_policy.lce_policy import LCE_Policy
from models.ctld_policy.ctld_policy import CTLD_Policy
from src.efficient_bounds_estimator import EfficientBoundsEstimator
from datasets.synthetic_log_confhai import SYN_LOG, HELOC_LOG
from sklearn.model_selection import train_test_split

from matplotlib import rcParams
import seaborn as sns
from src.utils import seed_everything
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 50

rc = {
    "figure.constrained_layout.use": True,
    "axes.titlesize": 20,
}
sns.set_theme(style="darkgrid", palette="colorblind", rc=None)

sns.set(style="whitegrid", palette="colorblind")
params = {
    "figure.constrained_layout.use": True,
    "axes.labelsize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    "font.size": 18,
}
import matplotlib.pyplot as plt
plt.rcParams.update(params)

_FUNCTION_COLOR = "#ad8bd6"



def ggplot_log_style_deferral(figsize, log_y=False, loc_maj_large=True):
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Give plot a gray background like ggplot.
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 16
    ax.set_facecolor('#EBEBEB')
    # Remove border around plot.
    [ax.spines[side].set_visible(False) for side in ax.spines]
    # Style the grid.
    ax.grid(which='major', color='white', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)

    return ax



def policy_regret(pi, y0, y1):
    pv =policy_value(pi, y0=y0, y1=y1)
    baseline_pv = y0.mean()
    return pv  - baseline_pv

GAMS = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# GAMS = [0.5, 1.0]
c = 0
nrep = 10
# wgamma = np.array([0.5, 0.5, 0.5]).ravel()
wgamma = np.array([1.0]).ravel()
hidd = 2


# Plots Setup
ax = ggplot_log_style_deferral(figsize=(682 / 72, 512 / 72), log_y=False)
colors = ['b', 'g', 'r', 'm', 'b', 'purple', 'brown', 'c', ]
markers = ['>', '+', '.', ',', 'o', 'v', 'x', 's', 'D', '|']


def save_policies(policies, file_path):
    tmp_policies = policies
    for k, v in tmp_policies.items():
        tmp_policies[k] = (v.numpy()).tolist()
    with file_path.open(mode="w") as fp:
        json.dump(tmp_policies, fp)


def load_policies(file_path):
    with file_path.open(mode="r") as fp:
        policies = json.load(fp)
    for k, v in policies.items():
        policies[k] = torch.Tensor(v)
    return policies


def get_deferral_costs(y_expert, a_expert, y0_hat, y1_hat):
    deferral_costs = []
    for j in range(len(a_expert)):
        if a_expert[j] == 0:
            deferral_costs.append(y1_hat[j] - y_expert[j])
        else:
            deferral_costs.append(y0_hat[j] - y_expert[j])

    return torch.Tensor(deferral_costs).unsqueeze(0)


def update_expert_preds(preds, expert_labels):
    defer_count = 0
    # updated_preds = preds.detach().clone()
    updated_preds = torch.Tensor(preds)
    for j in range(len(preds)):
        if preds[j] == 2:
            defer_count = defer_count + 1
            updated_preds[j] = expert_labels[j]

    deferral_rate = (defer_count / len(preds))
    return updated_preds, defer_count, deferral_rate


def extract_results(res_tau_hat):
    res_tau_mean_train = res_tau_hat["tau_mean_train"]
    res_tau_bottom_train = res_tau_hat["tau_bottom_train"]
    res_tau_top_train = res_tau_hat["tau_top_train"]
    res_tau_mean_test = res_tau_hat["tau_mean_test"]
    res_tau_bottom_test = res_tau_hat["tau_bottom_test"]
    res_tau_top_test = res_tau_hat["tau_top_test"]
    res_Y_0_bottom_train = res_tau_hat["Y_0_bottom_train"]
    res_Y_0_top_train = res_tau_hat["Y_0_top_train"]
    res_Y_1_bottom_train = res_tau_hat["Y_1_bottom_train"]
    res_Y_1_top_train = res_tau_hat["Y_1_top_train"]
    res_Y_0_bottom_test = res_tau_hat["Y_0_bottom_test"]
    res_Y_0_top_test = res_tau_hat["Y_0_top_test"]
    res_Y_1_bottom_test = res_tau_hat["Y_1_bottom_test"]
    res_Y_1_top_test = res_tau_hat["Y_1_top_test"]

    return res_tau_mean_train, res_tau_bottom_train, res_tau_top_train, \
           res_tau_mean_test, res_tau_bottom_test, res_tau_top_test, \
           res_Y_0_bottom_train, res_Y_0_top_train, res_Y_1_bottom_train, res_Y_1_top_train, \
           res_Y_0_bottom_test, res_Y_0_top_test, res_Y_1_bottom_test, res_Y_1_top_test


def get_confhai_result(results_dir):
    human_per_log_gamma = []
    ao_per_log_gamma = []
    confAo_per_log_gamma = []
    hAi_per_log_gamma = []
    confHAi_per_log_gamma = []
    confHAiPerson_per_log_gamma = []

    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith(".csv"):
            print(filename)
            file_path = os.path.join(results_dir, filename)

            # Read the CSV file
            data = pd.read_csv(file_path)
            human_per_log_gamma.append(np.asarray(data["Human"]))

            ao_per_log_gamma.append(np.asarray(data["AO"]))
            confAo_per_log_gamma.append(np.asarray(data["ConfAO"]))
            hAi_per_log_gamma.append(np.asarray(data["HAI"]))
            confHAi_per_log_gamma.append(np.asarray(data["ConfHAI"]))
            # confHAiPerson_per_log_gamma.append(np.asarray(data["ConfHAIPerson"]))

    labels = ["Human's Policy", "AO", "CRLogit Policy", "HAI", "ConfHAI Policy"]
    results = [human_per_log_gamma, ao_per_log_gamma, confAo_per_log_gamma, hAi_per_log_gamma, confHAi_per_log_gamma]
    return results, labels


if __name__ == '__main__':
    project_path = Path(os.getcwd())
    dir_path = project_path / "heloc_log_confhai_exp"

    num_teatments = 2

    cfg = OmegaConf.load("./config/config_heloc.yaml")

    bounds_model_config = OmegaConf.to_container(cfg.nuisance_estimator, resolve=True)
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    WANDB_PROJECT_NAME = cfg.exp.wandb_project
    WANDB_RUN_NAME = cfg.exp.wandb_run_name
    TRAIN_NUISANCE_MODELS = cfg.exp.nuisance_models_train_flag

    main_wandb_logger = WandbLogger(
        project=WANDB_PROJECT_NAME,
        name=WANDB_RUN_NAME,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    confhai_results_dir = project_path / "models/confhai/log/exp1"

    confhai_results, confhai_labels = get_confhai_result(results_dir=confhai_results_dir)
    # confhai_colors = ['blue', 'orange', 'green', 'red', 'purple']
    confhai_colors = ['g', 'orange', 'm', 'red', 'crimson', 'purple', 'brown', 'c']
    confhai_markers = ['o', '^', 'v', '*', 'x', 's', 'D', 'p']
    # for i in range(len(confhai_labels)):
    #     if confhai_labels[i] == "AO" or confhai_labels[i] == "HAI":
    #         continue
    #     result = confhai_results[i]
    #     means = np.mean(np.asarray(result), axis=1)
    #     sds = np.std(np.asarray(result), axis=1) / np.sqrt(nrep)
    #     plt.plot(GAMS, means, label=confhai_labels[i], color=confhai_colors[i], marker=confhai_markers[i])
    #     plt.fill_between(GAMS, means -sds, means + sds, color=confhai_colors[i], alpha=0.1)

    results_collector = []

    pv_conf_cate_list_all_trials = []
    pv_all_control_list_all_trials = []
    pv_ctld_list_all_trials = []
    pv_oracle_list_all_trails = []


    calls_list = []
    deferral_rates_random_defer_list = np.arange(0.1, 1.1, 0.1).tolist()

    for trial in range(nrep):
        print("---------------------------", trial, "---------------------------")
        seed_everything(trial)
        learn_policies_lg_ctld_flag = True
        ctld_policies = {}
        output_dir = dir_path / f"trial-{trial:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)


        file_path_ctld = output_dir / "ctld_policies.json"

        if file_path_ctld.exists():
            with file_path_ctld.open(mode="r") as fp:
                ctld_policies = load_policies(file_path=file_path_ctld)
                learn_policies_lg_ctld_flag = False

        # Datasets
        df = pd.read_csv('./datasets/heloc_dataset_v1 (1).csv')
        df['RiskPerformance'] = df['RiskPerformance'].map({'Good': 0, 'Bad': 1})
        feature_cols = [
            'ExternalRiskEstimate',
            'MSinceOldestTradeOpen',
            'MSinceMostRecentTradeOpen',
            'AverageMInFile',
            'NumSatisfactoryTrades',
            'NumTrades60Ever2DerogPubRec',
            'NumTrades90Ever2DerogPubRec',
            'PercentTradesNeverDelq',
            'MSinceMostRecentDelq',
            'MaxDelq2PublicRecLast12M',
            'MaxDelqEver',
            'NumTotalTrades',
            'NumTradesOpeninLast12M',
            'PercentInstallTrades',
            'MSinceMostRecentInqexcl7days',
            'NumInqLast6M',
            'NumInqLast6Mexcl7days',
            'NetFractionRevolvingBurden',
            'NetFractionInstallBurden',
            'NumRevolvingTradesWBalance',
            'NumInstallTradesWBalance',
            'NumBank2NatlTradesWHighUtilization',
            'PercentTradesWBalance'
        ]
        credit_col = 'RiskPerformance'

        df_train, df_tmp = train_test_split(df, test_size=0.3, random_state=trial)
        df_valid, df_test = train_test_split(df_tmp, test_size=0.5, random_state=trial)

        Gamma = [1.0, 1.0, 1.0]

        x_train, u_train, t_train, y_train, true_Q_obs_train, q0_train, y_all_train, q0_all_train, hid_train, T_h_train = generate_log_data_heloc(
            df_train, feature_cols, credit_col, len(df_train), Gamma, seed=trial)
        x_valid, u_valid, t_valid, y_valid, true_Q_obs_valid, q0_valid, y_all_valid, q0_all_valid, hid_valid, T_h_valid = generate_log_data_heloc(
            df_valid, feature_cols, credit_col, len(df_valid), Gamma, seed=trial + 2000)
        x_test, u_test, t_test, y_test, true_Q_obs_test, q0_test, y_all_test, q0_all_test, hid_test, T_h_test = generate_log_data_heloc(
            df_test, feature_cols, credit_col, len(df_test), Gamma, seed=trial + 1000)

        ds_train = HELOC_LOG(x_train, t_train, y_train, y_all_train, q0_all_train, hid_train, T_h_train, u_train)
        ds_valid = HELOC_LOG(x_valid, t_valid, y_valid, y_all_valid, q0_all_valid, hid_valid, T_h_valid, u_valid)
        ds_test = HELOC_LOG(x_test, t_test, y_test, y_all_test, q0_all_test, hid_test, T_h_test, u_test)

        Y0_test = y_all_test[:, 0]
        Y1_test = y_all_test[:, 1]

        expert_test_t = np.zeros(len(t_test))
        expert_test_y = Y0_test

        d = x_train.shape[1]  # dimension of x
        # Estimate CAPO Bounds in Parallel


        # Policy Value lists per trial
        pv_ctld_list = []
        pv_oracle_list = []

        for ind_g, gamma in enumerate(GAMS):
            print(gamma)



            pi_oracle = (Y1_test < Y0_test) * 1
            pv_oracle = policy_regret(pi=pi_oracle, y0=Y0_test, y1=Y1_test)
            pv_oracle_list.append(pv_oracle)


            # --- 2. EfficientBoundsEstimator and Dependent Policies ---
            nuisance_model_save_path = os.path.join(
                "./saved_nuisance_models",
                "heloc",
                f"trial_{trial:03d}",
                f"gamma_{gamma}"
            )
            estimator = EfficientBoundsEstimator(gamma=gamma, bounds_model_config=bounds_model_config,
                                                 trainer_config=trainer_config,
                                                 model_save_path=nuisance_model_save_path)

            if not TRAIN_NUISANCE_MODELS and os.path.isdir(nuisance_model_save_path):
                try:
                    estimator.load(nuisance_model_save_path)
                except FileNotFoundError as e:
                    print(f"Load failed: {e}. Falling back to training.")
                    estimator.fit(ds_train=ds_train, ds_valid=ds_valid, logger=main_wandb_logger,
                                  trial_idx=trial, k_log_gamma=str(gamma))
            else:
                if TRAIN_NUISANCE_MODELS:
                    print("Training flag is True. Starting new training.")
                else:
                    print(f"Directory not found: {nuisance_model_save_path}. Starting new training.")
                estimator.fit(ds_train=ds_train, ds_valid=ds_valid, logger=main_wandb_logger,
                              trial_idx=trial, k_log_gamma=str(gamma))

            ctld_policy_model = nn.Sequential(
                nn.Linear(len(feature_cols), 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 3)
            )
            policy_log_prefix = f"Trial_{trial}/Gamma_{gamma}/Policy"

            try:
                ctld_policy_trainer = CTLD_Policy(policy_model=ctld_policy_model, bounds_model=estimator, higher_better=False,
                                                  k_log_gamma=str(gamma))
                ctld_policy_trainer.fit(ds_train=ds_train, ds_valid=ds_valid, trial_idx=trial, logger = main_wandb_logger,
                                   devices=[0] if torch.cuda.is_available() else None, cache_dir="cache/heloc")

                model_save_dir = Path(f"./saved_ctld_policies/heloc//trial_{trial}/")
                model_save_dir.mkdir(parents=True, exist_ok=True)
                model_save_path = model_save_dir / f"ctld_policy_gamma_{gamma}.pth"

                torch.save(ctld_policy_trainer.policy_model.state_dict(), model_save_path)
                print(f"--- CTLD policy model saved to {model_save_path} ---")
                current_expet_test = torch.Tensor(expert_test_t).squeeze(0)
                pi_pred_ctld, p_defer_probs = ctld_policy_trainer.predict(ds_test=ds_test)
                pi_final_ctld, _, dr_ctld = update_expert_preds(preds=pi_pred_ctld, expert_labels=current_expet_test)
                pv_ctld = policy_value(pi=pi_final_ctld, y1=ds_test.y1, y0=ds_test.y0).item()

                # plot_defer_distribution(p_defer_probs)
            except Exception as e_lce:
                print(f"ERROR CTLD: {e_lce}")
                pv_ctld, dr_ctld = float('nan'), float('nan')
            policy_value_ctld_policies = policy_regret(pi=pi_final_ctld, y0=Y0_test, y1=Y1_test)
            pv_ctld_list.append(policy_value_ctld_policies)
            results_collector.append({
                'trial': trial,
                'gamma': gamma,
                'method': 'CTLD',
                'policy_regret': policy_value_ctld_policies
            })
            results_collector.append({
                'trial': trial,
                'gamma': gamma,
                'method': 'Oracle',
                'policy_regret': pv_oracle
            })

        pv_oracle_list_all_trails.append(pv_oracle_list)
        pv_ctld_list_all_trials.append(pv_ctld_list)

    results_df = pd.DataFrame(results_collector)
    output_filename = "heloc_experiment_results_only_ctld-1-10.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\nResults from all {nrep} trials have been saved to: {output_filename}")
