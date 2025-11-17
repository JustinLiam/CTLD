import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import warnings
import logging
import os
import gc
import wandb
from contextlib import contextmanager
from typing import Union
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*is smaller than the logging interval.*")



@contextmanager
def managed_dataloader(*args, **kwargs):
    loader = DataLoader(*args, **kwargs)
    try:
        yield loader
    finally:
        del loader
        gc.collect()


# 1) Propensity score estimator
class PropensityModel(L.LightningModule):
    def __init__(self, input_dim: int, num_treatments: int, lr: float = 1e-3, log_prefix: str = ""):
        super(PropensityModel, self).__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(self.hparams.input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, self.hparams.num_treatments)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X):
        return self.model(X)

    def _step(self, batch, batch_idx, step_name):
        X, A, _, = batch
        A_indices = torch.argmax(A, dim=1)
        A_pred_logits = self(X)
        loss = self.criterion(A_pred_logits, A_indices)
        log_key = f"{self.hparams.log_prefix}/{step_name}_prop_loss"
        self.log(log_key, loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, _, _ = batch
        return torch.softmax(self(X), dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4
        )


# 2) Conditional quantile model
class QuantileLoss(nn.Module):
    def __init__(self, quantile: float):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        loss = torch.max((self.quantile - 1) * error, self.quantile * error)
        return loss.mean()


class ConditionalQuantileModel(L.LightningModule):
    def __init__(self, quantile: float, input_dim: int, n_treatments: int, lr: float = 1e-3, log_key_suffix: str = "", log_prefix: str = ""):
        super(ConditionalQuantileModel, self).__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(self.hparams.input_dim + self.hparams.n_treatments, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        self.criterion = QuantileLoss(self.hparams.quantile)

    def forward(self, X, A):
        inputs = torch.cat([X, A], dim=1)
        return self.model(inputs)

    def _step(self, batch, batch_idx, step_name):
        X, A, Y = batch
        Y_pred = self(X, A)
        loss = self.criterion(Y_pred.squeeze(), Y)
        original_key = f"{step_name}_quantile{self.hparams.log_key_suffix}_loss"
        log_key = f"{self.hparams.log_prefix}/{original_key}"
        self.log(log_key, loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)


class OutcomeModel(L.LightningModule):
    def __init__(self, input_dim: int, num_treatments: int, type: str, quantile: float,
                 quantile_model: ConditionalQuantileModel, lr: float = 1e-3, is_for_alpha_minus: bool = False, log_prefix: str = ""):
        super(OutcomeModel, self).__init__()
        self.save_hyperparameters(ignore=['quantile_model'])
        self.quantile_model = quantile_model
        self.add_module("quantile_model", self.quantile_model)
        assert self.hparams.type in ['upper', 'lower', 'standard'], "Type must be in ['upper', 'lower', 'standard']"
        self.model = nn.Sequential(
            nn.Linear(self.hparams.input_dim + self.hparams.num_treatments, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        self.criterion = nn.HuberLoss()

    def forward(self, X, A):
        inputs = torch.cat([X, A], dim=1)
        return self.model(inputs)

    def _step(self, batch, batch_idx, step_name):
        X, A, Y = batch

        # self.quantile_model = self.quantile_model.to(X.device)

        Y_pred = self(X, A).squeeze()

        quantile_pred = self.quantile_model(X, A).squeeze()
        if self.hparams.type == 'upper':
            mask = (Y >= quantile_pred)
        else:  # 'lower'
            mask = (Y <= quantile_pred)

        if mask.sum() > 0:
            loss = self.criterion(Y_pred[mask], Y[mask])
        else:
            # loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            # loss = torch.tensor(0.0, requires_grad=True)
            # loss = torch.tensor(0.0, device=Y_pred.device, requires_grad=True)
            # loss = torch.tensor(0.0, device=Y_pred.device)
            # loss = torch.tensor(float("nan"), device=Y_pred.device, requires_grad=True)
            # print(f"[WARNING] No valid samples in batch {batch_idx}, logging NaN loss.")
            loss = torch.mean(Y_pred ** 2)

        log_suffix_prefix = "minus" if self.hparams.is_for_alpha_minus else "plus"
        original_key = f"{step_name}_{self.hparams.type}_{log_suffix_prefix}_loss"
        log_key = f"{self.hparams.log_prefix}/{original_key}"
        self.log(log_key, loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return loss


    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

NuisanceModel = Union[PropensityModel, ConditionalQuantileModel, OutcomeModel]

class EfficientBoundsEstimator(nn.Module):

    def __init__(self, gamma: float, bounds_model_config: dict, trainer_config: dict, model_save_path: str = "./saved_nuisance_models"):
        super().__init__()
        self.gamma = gamma
        self.config = bounds_model_config
        self.trainer_config = trainer_config
        # self.models = {}
        # self.models = nn.ModuleDict()
        self.models: nn.ModuleDict[str, NuisanceModel] = nn.ModuleDict()
        self.model_save_path = model_save_path
        self.model_names = [
            "propensity", "quant_plus", "quant_minus",
            "outcomes_upper_plus", "outcomes_lower_plus",
            "outcomes_upper_minus", "outcomes_lower_minus"
        ]

        if self.model_save_path and not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
            print(f"Created directory for saving models: {self.model_save_path}")

        device_str = self.config.get('device', 'auto')
        # if device_str == 'auto':
        #     self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # else:
        #     self.device = device_str
        #
        # self.accelerator = self.config.get('device', 'cpu')
        # logging.info(f"EfficientBoundsEstimator will use device: '{self.device}'")

    def _one_hot_encode(self, t, num_classes):
        return np.eye(num_classes)[t.astype(int).reshape(-1)]

    def fit(self, ds_train, ds_valid, logger: WandbLogger, trial_idx: int, k_log_gamma: str):

        x_train, a_train, y_train = torch.FloatTensor(ds_train.x), torch.FloatTensor(
            self._one_hot_encode(ds_train.t, self.config['n_treatments'])), torch.FloatTensor(ds_train.y)
        x_val, a_val, y_val = torch.FloatTensor(ds_valid.x), torch.FloatTensor(
            self._one_hot_encode(ds_valid.t, self.config['n_treatments'])), torch.FloatTensor(ds_valid.y)

        # train_loader = DataLoader(TensorDataset(x_train, a_train, y_train),
        #                           batch_size=self.config.get('batch_size', 64), num_workers=4)
        # val_loader = DataLoader(TensorDataset(x_val, a_val, y_val), batch_size=self.config.get('batch_size', 64),
        #                         num_workers=4)

        # train_loader = DataLoader(
        #     TensorDataset(x_train, a_train, y_train),
        #     batch_size=self.config.get('batch_size', 64),
        #     num_workers=4,
        #     pin_memory=True,
        #     persistent_workers=True,
        #     prefetch_factor=2
        # )
        # val_loader = DataLoader(
        #     TensorDataset(x_val, a_val, y_val),
        #     batch_size=self.config.get('batch_size', 64),
        #     num_workers=4,
        #     pin_memory=True,
        #     persistent_workers=True,
        #     prefetch_factor=2
        # )
        with managed_dataloader(TensorDataset(x_train, a_train, y_train),
                                batch_size=self.config.get('batch_size', 64),
                                num_workers=0,
                                pin_memory=True
                                # persistent_workers=True,
                                # prefetch_factor=2
                                ) as train_loader, \
                managed_dataloader(TensorDataset(x_val, a_val, y_val),
                                   batch_size=self.config.get('batch_size', 64),
                                   num_workers=0,
                                   pin_memory=True
                                   # persistent_workers=True,
                                   # prefetch_factor=2
                                   ) as val_loader:

            patience = self.config['patience']

            log_prefix = f"Trial_{trial_idx}/Gamma_{k_log_gamma}"


            # Train Propensity Model
            print("Training Propensity Model...")
            prop_monitor_key = f"{log_prefix}/val_prop_loss"
            prop_trainer_conf = self.trainer_config.copy()
            prop_trainer_conf["callbacks"] = [EarlyStopping(monitor=prop_monitor_key, patience=30, mode="min")]
            prop_trainer_conf["logger"] = logger
            prop_trainer_conf["accelerator"] = "gpu"
            prop_trainer_conf["devices"] = 1
            prop_trainer_conf["precision"] = "16-mixed"
            prop_model = PropensityModel(input_dim=self.config['input_dim'], num_treatments=self.config['n_treatments'],
                                         lr=self.config['propensity_model']['lr'], log_prefix=log_prefix)
            original_prop_model = prop_model
            # if hasattr(torch, 'compile'):
            #     print("Compiling the model for faster training...")
            #     prop_model = torch.compile(prop_model)
            prop_trainer = L.Trainer(**prop_trainer_conf)
            prop_trainer.fit(prop_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            self.models["propensity"] = prop_model.eval()

            if self.model_save_path:
                model_path = os.path.join(self.model_save_path, "propensity.pth")
                torch.save(original_prop_model.state_dict(), model_path)
                print(f"Saved propensity model to {model_path}")
            del prop_trainer, prop_model, original_prop_model
            gc.collect()

            alpha_plus = self.gamma / (1.0 + self.gamma)
            alpha_minus = 1.0 / (1.0 + self.gamma)

            # Train Quantile Models
            print(f"Training Quantile Model for alpha_plus ({alpha_plus:.4f})...")
            quant_plus_monitor_key = f"{log_prefix}/val_quantile_plus_loss"
            quant_plus_conf = self.trainer_config.copy()
            quant_plus_conf["callbacks"] = [EarlyStopping(monitor=quant_plus_monitor_key, patience=patience, mode="min")]
            quant_plus_conf["logger"] = logger
            quant_plus_conf["accelerator"] = "gpu"
            quant_plus_conf["devices"] = 1
            quant_plus_conf["precision"] = "16-mixed"
            quant_model_plus = ConditionalQuantileModel(quantile=alpha_plus, input_dim=self.config['input_dim'],
                                                        n_treatments=self.config['n_treatments'],
                                                        lr=self.config['quantile_model']['lr'], log_key_suffix="_plus", log_prefix=log_prefix )
            original_quant_model_plus = quant_model_plus
            # if hasattr(torch, 'compile'):
            #     print("Compiling the Quantile Model for faster training...")
            #     quant_model_plus = torch.compile(quant_model_plus)
            quant_plus_trainer = L.Trainer(**quant_plus_conf)
            quant_plus_trainer.fit(quant_model_plus, train_dataloaders=train_loader, val_dataloaders=val_loader)
            self.models["quant_plus"] = quant_model_plus.eval()

            if self.model_save_path:
                model_path = os.path.join(self.model_save_path, "quant_plus.pth")
                torch.save(original_quant_model_plus.state_dict(), model_path)
                print(f"Saved quant_plus model to {model_path}")
            del quant_plus_trainer, quant_model_plus, original_quant_model_plus
            gc.collect()

            print(f"Training Quantile Model for alpha_minus ({alpha_minus:.4f})...")

            quant_minus_conf = self.trainer_config.copy()
            quant_minus_monitor_key = f"{log_prefix}/val_quantile_minus_loss"
            quant_minus_conf["callbacks"] = [
                EarlyStopping(monitor=quant_minus_monitor_key, patience=patience, mode="min")]
            quant_minus_conf["logger"] = logger
            quant_minus_conf["accelerator"] = "gpu"
            quant_minus_conf["devices"] = 1
            quant_minus_conf["precision"] = "16-mixed"
            quant_model_minus = ConditionalQuantileModel(quantile=alpha_minus, input_dim=self.config['input_dim'],
                                                         n_treatments=self.config['n_treatments'],
                                                         lr=self.config['quantile_model']['lr'], log_key_suffix="_minus", log_prefix=log_prefix )
            original_quant_model_minus = quant_model_minus
            # if hasattr(torch, 'compile'):
            #     print("Compiling the Quantile Model for faster training...")
            #     quant_model_minus = torch.compile(quant_model_minus)
            quant_minus_trainer = L.Trainer(**quant_minus_conf)
            quant_minus_trainer.fit(quant_model_minus, train_dataloaders=train_loader, val_dataloaders=val_loader)
            self.models["quant_minus"] = quant_model_minus.eval()

            if self.model_save_path:
                model_path = os.path.join(self.model_save_path, "quant_minus.pth")
                torch.save(original_quant_model_minus.state_dict(), model_path)
                print(f"Saved quant_minus model to {model_path}")
            del quant_minus_trainer, quant_model_minus, original_quant_model_minus
            gc.collect()


            outcome_models_to_train = {
                "outcomes_upper_plus": {'type': 'upper', 'alpha': alpha_plus, 'quant_model': self.models["quant_plus"],
                                  'is_minus': False},
                "outcomes_lower_plus": {'type': 'lower', 'alpha': alpha_plus, 'quant_model': self.models["quant_plus"],
                                  'is_minus': False},
                "outcomes_upper_minus": {'type': 'upper', 'alpha': alpha_minus, 'quant_model': self.models["quant_minus"],
                                   'is_minus': True},
                "outcomes_lower_minus": {'type': 'lower', 'alpha': alpha_minus, 'quant_model': self.models["quant_minus"],
                                   'is_minus': True},
            }

            for name, params in outcome_models_to_train.items():
                print(f"Training  Outcome Model {name}...")

                log_key_suffix_prefix = "minus" if params['is_minus'] else "plus"
                monitor_key = f"{log_prefix}/val_{params['type']}_{log_key_suffix_prefix}_loss"

                om_conf = self.trainer_config.copy()
                om_conf["callbacks"] = [EarlyStopping(monitor=monitor_key, patience=patience, mode="min")]
                # om_conf["callbacks"] = [EarlyStopping(monitor=monitor_key, patience=10, mode="min")]
                om_conf["logger"] = logger
                om_conf["accelerator"] = "gpu"
                om_conf["devices"] = 1
                om_conf["precision"] = "16-mixed"
                model = OutcomeModel(input_dim=self.config['input_dim'], num_treatments=self.config['n_treatments'],
                                     type=params['type'], quantile=params['alpha'], quantile_model=params['quant_model'],
                                     lr=self.config['outcome_model']['lr'], is_for_alpha_minus=params['is_minus'], log_prefix=log_prefix )
                original_outcome_model = model
                # if hasattr(torch, 'compile'):
                #     print("Compiling the Outcome Model for faster training...")
                #     model = torch.compile(model)
                trainer = L.Trainer(**om_conf)
                trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                self.models[name] = model.eval()

                if self.model_save_path:
                    model_path = os.path.join(self.model_save_path, f"{name}.pth")
                    torch.save(original_outcome_model.state_dict(), model_path)
                    print(f"Saved {name} to {model_path}")

                del trainer, model, original_outcome_model
                gc.collect()

            print("All nuisance models trained.")
            return self

    def load(self, path: str):
        print(f"Attempting to load models from: {path}")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}. Cannot load models.")

        alpha_plus = self.gamma / (1.0 + self.gamma)
        alpha_minus = 1.0 / (1.0 + self.gamma)

        # 1. Propensity Model
        prop_model = PropensityModel(input_dim=self.config['input_dim'], num_treatments=self.config['n_treatments'],
                                     lr=self.config['propensity_model']['lr'])
        # 2. Quantile Models
        quant_model_plus = ConditionalQuantileModel(quantile=alpha_plus, input_dim=self.config['input_dim'],
                                                    n_treatments=self.config['n_treatments'],
                                                    lr=self.config['quantile_model']['lr'], log_key_suffix="_plus")
        quant_model_minus = ConditionalQuantileModel(quantile=alpha_minus, input_dim=self.config['input_dim'],
                                                     n_treatments=self.config['n_treatments'],
                                                     lr=self.config['quantile_model']['lr'], log_key_suffix="_minus")

        _loaded_models = {
            "propensity": prop_model,
            "quant_plus": quant_model_plus,
            "quant_minus": quant_model_minus,
        }

        # 3. Outcome Models
        outcome_model_params = {
            "outcomes_upper_plus": {'type': 'upper', 'alpha': alpha_plus, 'quant_model': _loaded_models["quant_plus"],
                              'is_minus': False},
            "outcomes_lower_plus": {'type': 'lower', 'alpha': alpha_plus, 'quant_model': _loaded_models["quant_plus"],
                              'is_minus': False},
            "outcomes_upper_minus": {'type': 'upper', 'alpha': alpha_minus, 'quant_model': _loaded_models["quant_minus"],
                               'is_minus': True},
            "outcomes_lower_minus": {'type': 'lower', 'alpha': alpha_minus, 'quant_model': _loaded_models["quant_minus"],
                               'is_minus': True},
        }
        for name, params in outcome_model_params.items():
            _loaded_models[name] = OutcomeModel(input_dim=self.config['input_dim'],
                                                num_treatments=self.config['n_treatments'], type=params['type'],
                                                quantile=params['alpha'], quantile_model=params['quant_model'],
                                                lr=self.config['outcome_model']['lr'],
                                                is_for_alpha_minus=params['is_minus'])

        for name in self.model_names:
            model_path = os.path.join(path, f"{name}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}. Cannot complete loading.")

            model_instance = _loaded_models[name]

            state_dict = torch.load(model_path)

            new_state_dict = {key.replace('_orig_mod.', ''): value
                              for key, value in state_dict.items()}

            model_instance.load_state_dict(new_state_dict)

            self.models[name] = model_instance.eval()

            # if hasattr(torch, 'compile'):
            #     print(f"Compiling the loaded model {name} for faster inference...")
            #     self.models[name] = torch.compile(self.models[name])

            print(f"Successfully loaded {name} from {model_path}")

        print("All nuisance models loaded successfully.")
        return self

    def forward(self, x_data):

        self.eval()

        # X_tensor = torch.tensor(x_data, dtype=torch.float32).to(self.device)
        # X_tensor = torch.tensor(x_data, dtype=torch.float32)
        X_tensor = torch.as_tensor(x_data, dtype=torch.float32)
        model_device = next(self.parameters()).device
        X_tensor = X_tensor.to(model_device)
        # X_tensor = X_tensor.to(next(self.models["propensity"].parameters()).device)
        capo_bounds_output = {}
        num_treatments = self.config['n_treatments']

        gamma = self.gamma
        gamma_inv = 1.0 / gamma
        alpha_plus = gamma / (1.0 + gamma)
        alpha_minus = 1.0 / (1.0 + gamma)

        with torch.no_grad():
            propensity_scores_all_actions = torch.softmax(self.models["propensity"](X_tensor), dim=1)

            for a_idx in range(num_treatments):
                # a_one_hot = torch.zeros(X_tensor.shape[0], num_treatments, device=self.device)
                a_one_hot = torch.zeros(X_tensor.shape[0], num_treatments, device=X_tensor.device)
                a_one_hot[:, a_idx] = 1

                current_propensity = propensity_scores_all_actions[:, a_idx:(a_idx + 1)]

                mu_bar_plus_pred = self.models["outcomes_upper_plus"](X_tensor, a_one_hot)
                mu_underbar_plus_pred = self.models["outcomes_lower_plus"](X_tensor, a_one_hot)

                coeff_upper = (current_propensity + (1 - current_propensity) * gamma) * alpha_minus
                coeff_lower = (current_propensity + (1 - current_propensity) * gamma_inv) * alpha_plus

                Q_plus_a = coeff_upper * mu_bar_plus_pred + coeff_lower * mu_underbar_plus_pred

                mu_bar_minus_pred = self.models["outcomes_upper_minus"](X_tensor, a_one_hot)
                mu_underbar_minus_pred = self.models["outcomes_lower_minus"](X_tensor, a_one_hot)

                coeff_upper_minus = (current_propensity + (1 - current_propensity) * gamma_inv) * alpha_plus
                coeff_lower_minus = (current_propensity + (1 - current_propensity) * gamma) * alpha_minus

                Q_minus_a = coeff_upper_minus * mu_bar_minus_pred + coeff_lower_minus * mu_underbar_minus_pred

                Q_top = torch.max(Q_plus_a, Q_minus_a)
                Q_bottom = torch.min(Q_plus_a, Q_minus_a)

                # capo_bounds_output[f"Y_{a_idx}_top"] = Q_top.cpu().numpy()
                # capo_bounds_output[f"Y_{a_idx}_bottom"] = Q_bottom.cpu().numpy()

                capo_bounds_output[f"Y_{a_idx}_top"] = Q_top
                capo_bounds_output[f"Y_{a_idx}_bottom"] = Q_bottom

        return capo_bounds_output

