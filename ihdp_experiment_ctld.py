import os
import gc
import math
from pathlib import Path
import numpy as np
import random
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import wandb
import warnings
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, DictConfig
from models.crlogit.data_scenarios import *

from src.efficient_bounds_estimator import EfficientBoundsEstimator
from models.ctld_policy.ctld_policy import CTLD_Policy
from models.lce_policy.lce_policy_original import LCE_Policy
from datasets import IHDP
from src.utils import seed_everything, update_expert, get_expert_by_feature, policy_value, update_expert_preds, run_crlogit_policy, plot_defer_distribution
from src.compute_blearner_bounds_ihdp import compute_all_intervals_BLearner

import torch._dynamo
torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore", message=".*XGBoost is not compiled with CUDA support.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch._dynamo.*")



if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    try:
        if mp.get_start_method(allow_none=True) != 'spawn': mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    except Exception as e:
        print(f"MP start method failed: {e}")

    project_path = Path(os.getcwd())
    # =============================================================================
    # Phase--Initialization
    # =============================================================================
    cfg = OmegaConf.load("./config/config_ihdp.yaml")
    seed_everything(cfg.exp.global_seed)

    WANDB_PROJECT_NAME = cfg.exp.wandb_project
    WANDB_RUN_NAME = cfg.exp.wandb_run_name
    TRAIN_NUISANCE_MODELS = cfg.exp.nuisance_models_train_flag
    trials = cfg.exp.num_trials
    start_trial = cfg.exp.start_trial
    exp_feature = cfg.exp.exp_feature


    main_wandb_logger = WandbLogger(
        project=WANDB_PROJECT_NAME,
        name=WANDB_RUN_NAME,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    log_gamma_bases = cfg.log_gammas
    GAMMAS = {key: math.exp(value) for key, value in log_gamma_bases.items()}


    train_dss, val_dss, test_dss = [], [], []
    print(f"--- Phase 0: Loading IHDP datasets for {trials} trials ---")
    for trial_seed in range(trials):
        _train_ds = IHDP(root=None, split="train", mode='mu', seed=trial_seed, hidden_confounding=True)
        train_dss.append(update_expert(_train_ds, feature=exp_feature))
        _val_ds = IHDP(root=None, split="valid", mode='mu', seed=trial_seed, hidden_confounding=True)
        val_dss.append(update_expert(_val_ds, feature=exp_feature))
        _test_ds = IHDP(root=None, split="test", mode='mu', seed=trial_seed, hidden_confounding=True)
        test_dss.append(update_expert(ds=_test_ds, feature=exp_feature))
    print(f"--- All {len(train_dss)} IHDP datasets loaded ---")

    all_policy_results = {
        "Oracle": {"PV": {k: [] for k in GAMMAS.keys()}},
        "Expert": {"PV": {k: [] for k in GAMMAS.keys()}},
        "CRLogit": {"PV": {k: [] for k in GAMMAS.keys()}},
        "Random_Defer": {"PV": {k: [] for k in GAMMAS.keys()}},
        "B-Learner": {"PV": {k: [] for k in GAMMAS.keys()}, "DR": {k: [] for k in GAMMAS.keys()}},
        "Pessimistic": {"PV": {k: [] for k in GAMMAS.keys()}},
        "CARED": {"PV": {k: [] for k in GAMMAS.keys()}, "DR": {k: [] for k in GAMMAS.keys()}},
        "CTLD": {"PV": {k: [] for k in GAMMAS.keys()}, "DR": {k: [] for k in GAMMAS.keys()}},
    }
    var_CATE = []
    exclude_count = 0

    for trial_idx in range(start_trial, trials):
        print(f"\n================ Processing Trial {trial_idx} ================")

        ihdp_train_ds = train_dss[trial_idx]
        ihdp_val_ds = val_dss[trial_idx]
        ihdp_test_ds = test_dss[trial_idx]
        Y_0_test, Y_1_test = ihdp_test_ds.y0, ihdp_test_ds.y1
        expert_policy_test, _ = get_expert_by_feature(ds=ihdp_test_ds, feature=exp_feature)
        expert_policy_test = torch.tensor(expert_policy_test, dtype=torch.long)

        var_CATE_trial = np.sqrt(np.var(ihdp_test_ds.mu1 - ihdp_test_ds.mu0))
        var_CATE.append(var_CATE_trial)
        if var_CATE_trial > 15:
            print(f"--- Trial {trial_idx} SKIPPED due to high CATE variance ({var_CATE_trial:.2f}) ---")
            for k_log_gamma in GAMMAS.keys():
                for policy_name, metrics in all_policy_results.items():
                    for metric_name in metrics:
                        all_policy_results[policy_name][metric_name][k_log_gamma].append(float('nan'))
            continue

        bounds_model_config = OmegaConf.to_container(cfg.nuisance_estimator, resolve=True)
        trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)

        # Oracle Policy and Expert Policy
        tau_true = torch.tensor(ihdp_test_ds.mu1 - ihdp_test_ds.mu0)
        oracle_pi = (tau_true > 0).int()
        pv_oracle = policy_value(pi=oracle_pi, y1=Y_1_test, y0=Y_0_test).item()
        pv_expert = policy_value(pi=expert_policy_test, y1=Y_1_test, y0=Y_0_test).item()

        # CRLogit Policy
        try:
            pv_cr_logit_all_gammas = run_crlogit_policy(ihdp_train_ds, ihdp_val_ds, ihdp_test_ds, GAMMAS)
            cr_logit_pv_map = {k: v for k, v in zip(GAMMAS.keys(), pv_cr_logit_all_gammas)}
        except Exception as e:
            print(f"ERROR during CRLogit execution for Trial {trial_idx}: {e}")
            cr_logit_pv_map = {k: float('nan') for k in GAMMAS.keys()}

        blearner_bounds_all_gammas = compute_all_intervals_BLearner(dir_path=Path("./ihdp_blearner_results"),
                                                 trial=trial_idx,
                                                 ds_train=ihdp_train_ds,
                                                 ds_valid=ihdp_val_ds,
                                                 ds_test=ihdp_test_ds,
                                                 GAMMAS=GAMMAS)
        # blearner_results.append(results)


        for k_log_gamma, v_gamma_val in GAMMAS.items():
            print(f"\n----------- Processing Gamma {k_log_gamma} for Trial {trial_idx} -----------")

            # --- 1. B-Learner Bounds and Dependent Policies ---
            tau_hat_blearner = blearner_bounds_all_gammas[k_log_gamma]
            tau_top_test_b = torch.from_numpy(np.asarray(tau_hat_blearner['tau_top_test'])).float()
            tau_bottom_test_b = torch.from_numpy(np.asarray(tau_hat_blearner['tau_bottom_test'])).float()
            tau_mean_test_b = torch.from_numpy(np.asarray(tau_hat_blearner['tau_mean_test'])).float()

            Y_0_bottom_test_b = torch.from_numpy(np.asarray(tau_hat_blearner['Y_0_bottom_test'])).float()
            Y_0_top_test_b = torch.from_numpy(np.asarray(tau_hat_blearner['Y_0_top_test'])).float()
            Y_1_bottom_test_b = torch.from_numpy(np.asarray(tau_hat_blearner['Y_1_bottom_test'])).float()
            Y_1_top_test_b = torch.from_numpy(np.asarray(tau_hat_blearner['Y_1_top_test'])).float()

            # CATE Interval Policy
            pi_cate = ((tau_top_test_b > 0) & (tau_bottom_test_b > 0)).long()
            defer_indices = torch.where((tau_top_test_b >= 0) & (tau_bottom_test_b <= 0))[0]
            # expert_policy_tensor = torch.as_tensor(expert_policy_test)
            pi_cate[defer_indices] = expert_policy_test[defer_indices]
            pv_cate = policy_value(pi=pi_cate, y1=Y_1_test, y0=Y_0_test).item()
            dr_cate = len(defer_indices) / len(pi_cate) if len(pi_cate) > 0 else 0

            #  Pessimistic Policy ---
            pi_pess = np.zeros(len(expert_policy_test))
            y1b, y0t = Y_1_bottom_test_b, Y_0_top_test_b
            y1t, y0b = Y_1_top_test_b, Y_0_bottom_test_b
            for j in range(len(pi_pess)):
                if (y1b[j] - y0t[j]) > 0:
                    pi_pess[j] = 1
                elif (y1t[j] - y0b[j]) < 0:
                    pi_pess[j] = 0
                elif y1b[j] > y0b[j]:
                    pi_pess[j] = 1
                else:
                    pi_pess[j] = 0
            pv_pess = policy_value(pi=pi_pess, y1=Y_1_test, y0=Y_0_test).item()

            # Random Defer Policy
            pv_random_defer_list = []
            deferral_rates_random_defer_list = np.arange(0.1, 1.1, 0.1).tolist()
            n_test = len(ihdp_test_ds.x)
            current_policy_test, _ = get_expert_by_feature(ds=ihdp_test_ds, feature=exp_feature)
            expert_policy_tensor = torch.as_tensor(expert_policy_test, dtype=torch.long)

            base_policy = (tau_mean_test_b > 0).long()
            for def_rate in deferral_rates_random_defer_list:
                pi_random_defer = base_policy.clone()
                num_to_defer = math.floor(def_rate * n_test)
                deferral_indices = random.sample(range(n_test), k=num_to_defer)
                if deferral_indices:
                    pi_random_defer[deferral_indices] = expert_policy_tensor[deferral_indices]
                policy_value_random_defer = policy_value(pi=pi_random_defer, y1=Y_1_test, y0=Y_0_test).item()
                pv_random_defer_list.append(policy_value_random_defer)
            avg_pv_random = np.mean(pv_random_defer_list)

            # --- 2. EfficientBoundsEstimator and Dependent Policies ---
            nuisance_model_save_path = os.path.join(
                "./saved_nuisance_models",
                WANDB_PROJECT_NAME,
                WANDB_RUN_NAME,
                f"trial_{trial_idx:03d}",
                f"gamma_{k_log_gamma}"
            )
            estimator = EfficientBoundsEstimator(gamma=v_gamma_val, bounds_model_config=bounds_model_config, trainer_config=trainer_config, model_save_path=nuisance_model_save_path)

            if not TRAIN_NUISANCE_MODELS and os.path.isdir(nuisance_model_save_path):
                try:
                    estimator.load(nuisance_model_save_path)
                except FileNotFoundError as e:
                    print(f"Load failed: {e}. Falling back to training.")
                    estimator.fit(ds_train=train_dss[trial_idx], ds_valid=val_dss[trial_idx], logger=main_wandb_logger,
                                  trial_idx=trial_idx, k_log_gamma=k_log_gamma)
            else:
                if TRAIN_NUISANCE_MODELS:
                    print("Training flag is True. Starting new training.")
                else:
                    print(f"Directory not found: {nuisance_model_save_path}. Starting new training.")
                estimator.fit(ds_train=train_dss[trial_idx], ds_valid=val_dss[trial_idx], logger=main_wandb_logger,
                              trial_idx=trial_idx, k_log_gamma=k_log_gamma)


            diag_log_prefix = f"Trial_{trial_idx}/Gamma_{k_log_gamma}/Nuisance_Diagnostics"

            ctld_policy_model = nn.Sequential(
                nn.Linear(ihdp_train_ds.x.shape[1], 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 3)
            )

            lce_model = nn.Linear(ihdp_train_ds.x.shape[1], 3)
            policy_log_prefix = f"Trial_{trial_idx}/Gamma_{k_log_gamma}/Policy"

            try:
                lce_trainer = LCE_Policy(
                    tau_hat=tau_hat_blearner,
                    policy_model=lce_model,
                    gamma=v_gamma_val,
                    higher_better=True
                )
                lce_trainer.fit(ds_train=ihdp_train_ds, ds_valid=ihdp_val_ds,
                                   devices=[0] if torch.cuda.is_available() else None)
                pi_pred_lce = lce_trainer.predict(ds_test=ihdp_test_ds)
                pi_final_lce, _, dr_lce = update_expert_preds(preds=pi_pred_lce, expert_labels=expert_policy_test)
                pv_lce = policy_value(pi=pi_final_lce, y1=Y_1_test, y0=Y_0_test).item()
            except Exception as e:
                print(f"ERROR LCE: {e}")
                pv_lce, dr_lce = float('nan'), float('nan')

            try:
                ctld_policy_trainer = CTLD_Policy(policy_model=ctld_policy_model, bounds_model=estimator, higher_better=True,
                                                  k_log_gamma=k_log_gamma)
                ctld_policy_trainer.fit(ds_train=ihdp_train_ds, ds_valid=ihdp_val_ds, trial_idx=trial_idx, logger = main_wandb_logger,
                                   devices=[0] if torch.cuda.is_available() else None)

                model_save_dir = Path(f"./saved_ctld_policies/{WANDB_PROJECT_NAME}/{WANDB_RUN_NAME}/trial_{trial_idx}/")
                model_save_dir.mkdir(parents=True, exist_ok=True)
                model_save_path = model_save_dir / f"ctld_policy_gamma_{k_log_gamma}.pth"

                torch.save(ctld_policy_trainer.policy_model.state_dict(), model_save_path)
                print(f"--- ctld policy model saved to {model_save_path} ---")

                pi_pred_ctld, p_defer_probs = ctld_policy_trainer.predict(ds_test=test_dss[trial_idx])
                pi_final_ctld, _, dr_ctld = update_expert_preds(preds=pi_pred_ctld, expert_labels=expert_policy_test)
                pv_ctld = policy_value(pi=pi_final_ctld, y1=Y_1_test, y0=Y_0_test).item()


            except Exception as e_lce:
                print(f"ERROR ctld: {e_lce}")
                pv_ctld, dr_ctld = float('nan'), float('nan')



            print(f"--- Policy Evaluation for Trial {trial_idx}, Gamma {k_log_gamma} ---")

            print(f"--- {policy_log_prefix} ---")
            current_policy_test, _ = get_expert_by_feature(ds=ihdp_test_ds, feature=exp_feature)

            pv_crlogit = cr_logit_pv_map.get(k_log_gamma, float('nan'))
            results_to_store = {
                "Oracle": {"PV": pv_oracle},
                "Expert": {"PV": pv_expert},
                "CRLogit": {"PV": pv_crlogit},
                "B-Learner": {"PV": pv_cate, "DR": dr_cate},
                "Pessimistic": {"PV": pv_pess},
                "Random_Defer": {"PV": avg_pv_random},
                "CARED": {"PV": pv_lce, "DR": dr_lce},
                "CTLD": {"PV": pv_ctld, "DR": dr_ctld},
            }
            for policy, metrics in results_to_store.items():
                for metric, value in metrics.items():
                    all_policy_results[policy][metric][k_log_gamma].append(value)

            wandb.log(
                {f"Trial_{trial_idx}/Gamma_{k_log_gamma}/Policy/{p}/{m}": v for p, ms in results_to_store.items() for
                 m, v in ms.items()})

            del ctld_policy_trainer, lce_trainer, estimator
            gc.collect()

    # =============================================================================
    # Phase--results aggregation and final logging
    # =============================================================================
    print("\n--- Aggregating and Logging Final Summary Statistics ---")

    for policy_name, metrics in all_policy_results.items():
        for metric_name, gammas_data in metrics.items():
            for gamma_key, values in gammas_data.items():
                if not values: continue
                values_arr = np.array(values)
                valid_n = np.sum(~np.isnan(values_arr))
                if valid_n > 0:
                    mean = np.nanmean(values_arr)
                    se = np.nanstd(values_arr, ddof=1) / np.sqrt(valid_n)
                else:
                    mean = float('nan')
                    se = float('nan')

                summary_key_mean = f"Final_Summary/{policy_name}/{metric_name}/Gamma_{gamma_key}_Mean"
                summary_key_se = f"Final_Summary/{policy_name}/{metric_name}/Gamma_{gamma_key}_SE"
                wandb.summary[summary_key_mean] = mean
                wandb.summary[summary_key_se] = se
                print(f"{summary_key_mean}: {mean:.4f} ± {se:.4f}")

    print("Experiment finished. Finalizing wandb run.")
    main_wandb_logger.finalize("success")
