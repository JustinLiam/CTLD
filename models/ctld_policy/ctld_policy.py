import os
import numpy as np
import torch
# from sklearn import clone
import time
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from models.ctld_policy.ctld_model import CTLD_Model, asymmetric_softmax_transformation

import logging
import warnings

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*is smaller than the logging interval.*")

torch.manual_seed(42)
torch.set_float32_matmul_precision('medium')


class CTLD_Policy:

    def __init__(self,
                 policy_model,
                 bounds_model,
                 higher_better=True,
                 k_log_gamma: str = None):

        self.bounds_model = bounds_model
        self.higher_better = higher_better
        self.policy_model = policy_model
        self.policy_trainer = None
        self.log_gamma = k_log_gamma
        self.eps = 1e-8

    def _get_bounds(self, ds, device="cuda"):
        x_tensor = torch.from_numpy(ds.x).float().to(device)

        self.bounds_model = self.bounds_model.to(device)

        with torch.no_grad():
            outcome_bounds = self.bounds_model(x_tensor)

        Y_0_bottom = outcome_bounds['Y_0_bottom'].float().reshape(-1)
        Y_0_top = outcome_bounds['Y_0_top'].float().reshape(-1)
        Y_1_bottom = outcome_bounds['Y_1_bottom'].float().reshape(-1)
        Y_1_top = outcome_bounds['Y_1_top'].float().reshape(-1)

        return Y_0_bottom, Y_0_top, Y_1_bottom, Y_1_top

    def _construct_pseudo_posterior(self, ds, device="cuda"):
        Y_0_bottom, Y_0_top, Y_1_bottom, Y_1_top = self._get_bounds(ds)


        if not self.higher_better:
            Y_0_bottom, Y_0_top = -Y_0_top, -Y_0_bottom
            Y_1_bottom, Y_1_top = -Y_1_top, -Y_1_bottom

        I_0_len = Y_0_top - Y_0_bottom
        I_1_len = Y_1_top - Y_1_bottom

        intersection_min = torch.max(Y_0_bottom, Y_1_bottom)
        intersection_max = torch.min(Y_0_top, Y_1_top)
        intersection_len = torch.clamp(intersection_max - intersection_min, min=0)

        union_len = I_0_len + I_1_len - intersection_len

        p_defer = (intersection_len / (union_len + self.eps)).clamp(min=0.0, max=1.0)

        mu1_hat = (Y_1_top + Y_1_bottom) / 2
        mu0_hat = (Y_0_top + Y_0_bottom) / 2

        score_1 = mu1_hat - mu0_hat
        score_0 = -score_1
        action_scores = torch.stack([score_0, score_1], dim=1)
        scaler = union_len + self.eps
        scaled_action_scores = action_scores / scaler.unsqueeze(1)

        p_action = torch.softmax(scaled_action_scores, dim=1)

        p_defer_reshaped = p_defer.unsqueeze(1)
        pseudo_posterior_target = torch.cat((p_action, p_defer_reshaped), dim=1)

        return pseudo_posterior_target


    def fit(self, ds_train, ds_valid, trial_idx: int,
            batch_size=64, patience=10, max_epochs=100, lr=5e-4, weight_decay=1e-3, optimizer_name='Adam',
            logger=None, devices=[0], cache_dir="cache"):

        cache_path = os.path.join(cache_dir, f"trial_{trial_idx}", self.log_gamma)
        os.makedirs(cache_path, exist_ok=True)
        train_targets_path = os.path.join(cache_path, "train_targets.pt")
        valid_targets_path = os.path.join(cache_path, "valid_targets.pt")

        device = f"cuda:{devices[0]}" if isinstance(devices, list) else f"cuda:{devices}"

        if os.path.exists(train_targets_path):
            logging.info(f"Loading cached training targets from: {train_targets_path}")
            pseudo_posterior_train = torch.load(train_targets_path)
        else:
            logging.info("Constructing pseudo-posterior targets for training set...")
            start_time = time.time()
            pseudo_posterior_train = self._construct_pseudo_posterior(ds=ds_train, device=device)
            logging.info(f"Construction took {time.time() - start_time:.2f} seconds.")
            logging.info(f"Caching training targets to: {train_targets_path}")

            torch.save(pseudo_posterior_train.cpu(), train_targets_path)

        if os.path.exists(valid_targets_path):
            logging.info(f"Loading cached validation targets from: {valid_targets_path}")
            pseudo_posterior_valid = torch.load(valid_targets_path)
        else:
            logging.info("Constructing pseudo-posterior targets for validation set...")
            pseudo_posterior_valid = self._construct_pseudo_posterior(ds=ds_valid, device=device)
            logging.info(f"Caching validation targets to: {valid_targets_path}")
            torch.save(pseudo_posterior_valid.cpu(), valid_targets_path)

        pseudo_posterior_train_cpu = pseudo_posterior_train.cpu()
        pseudo_posterior_valid_cpu = pseudo_posterior_valid.cpu()

        p_defer_targets = pseudo_posterior_train[:, 2]
        logging.info(f"Target Deferral Probabilities (Train Set): Mean={p_defer_targets.mean():.4f}")

        x_train_tensor = torch.from_numpy(ds_train.x).float()
        x_valid_tensor = torch.from_numpy(ds_valid.x).float()

        train_loader = DataLoader(TensorDataset(x_train_tensor, pseudo_posterior_train_cpu), batch_size=batch_size,
                                  num_workers=0, pin_memory=True, shuffle=True, drop_last=True)
        valid_loader = DataLoader(TensorDataset(x_valid_tensor, pseudo_posterior_valid_cpu), batch_size=batch_size,
                                  num_workers=0, pin_memory=True, shuffle=False, drop_last=True)

        logging.info(f"Starting training with hyperparameters: lr={lr}, max_epochs={max_epochs}, patience={patience}")

        ctld_model = CTLD_Model(pmodel=self.policy_model, lr=lr, weight_decay=weight_decay,
                                optimizer_name=optimizer_name, gamma=self.log_gamma)

        loss_monitor = "val_loss"
        ckpt_cb = ModelCheckpoint(monitor=loss_monitor, mode="min", save_top_k=1, save_weights_only=True)

        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            callbacks=[EarlyStopping(monitor=loss_monitor, patience=patience, mode="min"), ckpt_cb],
            accelerator="gpu",
            devices=devices
        )

        trainer.fit(ctld_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        best_path = ckpt_cb.best_model_path
        if best_path and os.path.exists(best_path):
            logging.info(f"Loading best model from: {best_path}")
            ctld_model = CTLD_Model.load_from_checkpoint(best_path, pmodel=self.policy_model)
            self.policy_model = ctld_model.pmodel
        else:
            logging.warning("Best model path not found or invalid. Using last epoch model.")

        self.policy_trainer = trainer
        logging.info("Training finished.")

    def predict(self, ds_test):

        self.policy_model.eval()

        device = next(self.policy_model.parameters()).device
        x_test_tensor = torch.from_numpy(ds_test.x).float()
        test_loader = DataLoader(TensorDataset(x_test_tensor), batch_size=64, shuffle=False)

        all_predictions = []
        all_p_defer = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch = batch[0].to(device)
                logits_batch = self.policy_model(x_batch)

                probs_batch = asymmetric_softmax_transformation(logits_batch)

                pred_batch = torch.argmax(probs_batch, dim=1)

                all_predictions.append(pred_batch.cpu())

                all_p_defer.append(probs_batch[:, 2].cpu())

        ctld_pi = torch.cat(all_predictions, dim=0)
        p_defer_probs = torch.cat(all_p_defer, dim=0)

        counts = torch.bincount(ctld_pi, minlength=3)
        total = counts.sum().item()

        return ctld_pi, p_defer_probs

    def get_cate_bounds(self, ds):

        Y_0_bottom, Y_0_top, Y_1_bottom, Y_1_top = self._get_bounds(ds)
        tau_bottom = Y_1_bottom - Y_0_top
        tau_top = Y_1_top - Y_0_bottom
        tau_mean = ((Y_1_top + Y_1_bottom) / 2) - ((Y_0_top + Y_0_bottom) / 2)

        def reshape_tensor(t): return t.reshape(-1)

        return reshape_tensor(tau_bottom), reshape_tensor(tau_mean), reshape_tensor(tau_top)
