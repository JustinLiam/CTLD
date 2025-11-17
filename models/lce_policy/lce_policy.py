import os

import numpy as np
import ray
import torch
from sklearn import clone
from ray.air import RunConfig, CheckpointConfig

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.backends.opt_einsum import strategy
from torch.utils.data import DataLoader
import torch.nn as nn

from models.blearner import BLearner
from models.lce_surrogate.lce_model import LCEModel, LCEModel_optuna

import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from ray import tune

# ray.shutdown()

import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
# ray.init(log_to_driver=True)

torch.manual_seed(42)
torch.set_float32_matmul_precision('medium')


def extract_results(res_tau_hat):
    res_tau_mean_train = res_tau_hat["tau_mean_train"]
    res_tau_bottom_train = res_tau_hat["tau_bottom_train"]
    res_tau_top_train = res_tau_hat["tau_top_train"]

    res_tau_mean_test = res_tau_hat["tau_mean_test"]
    res_tau_bottom_test = res_tau_hat["tau_bottom_test"]
    res_tau_top_test = res_tau_hat["tau_top_test"]

    res_tau_mean_val = res_tau_hat["tau_mean_val"]
    res_tau_bottom_val = res_tau_hat["tau_bottom_val"]
    res_tau_top_val = res_tau_hat["tau_top_val"]

    res_Y_0_bottom_train = res_tau_hat["Y_0_bottom_train"]
    res_Y_0_top_train = res_tau_hat["Y_0_top_train"]
    res_Y_1_bottom_train = res_tau_hat["Y_1_bottom_train"]
    res_Y_1_top_train = res_tau_hat["Y_1_top_train"]

    res_Y_0_bottom_test = res_tau_hat["Y_0_bottom_test"]
    res_Y_0_top_test = res_tau_hat["Y_0_top_test"]
    res_Y_1_bottom_test = res_tau_hat["Y_1_bottom_test"]
    res_Y_1_top_test = res_tau_hat["Y_1_top_test"]

    res_Y_0_bottom_val = res_tau_hat["Y_0_bottom_val"]
    res_Y_0_top_val = res_tau_hat["Y_0_top_val"]
    res_Y_1_bottom_val = res_tau_hat["Y_1_bottom_val"]
    res_Y_1_top_val = res_tau_hat["Y_1_top_val"]

    return res_tau_mean_train, res_tau_bottom_train, res_tau_top_train, \
           res_tau_mean_test, res_tau_bottom_test, res_tau_top_test, \
           res_tau_mean_val, res_tau_bottom_val, res_tau_top_val, \
           res_Y_0_bottom_train, res_Y_0_top_train, res_Y_1_bottom_train, res_Y_1_top_train, \
           res_Y_0_bottom_test, res_Y_0_top_test, res_Y_1_bottom_test, res_Y_1_top_test, \
           res_Y_0_bottom_val, res_Y_0_top_val, res_Y_1_bottom_val, res_Y_1_top_val


def objective(trial, train_loader, val_loader, devices, policy_model,
              costs_matrix_train, costs_matrix_valid):
    lr = trial.suggest_loguniform('lr', 1e-4, 0.1)  # 1e-5, 1e-2
    optimizer_name = trial.suggest_categorical('optimizer_name', ['SGD', 'Adam', 'AdamW'])
    # weight_decay = trial.suggest_loguniform('weight_decay', 0.001, 0.1)
    # weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)

    patience = trial.suggest_int("patience", 5, 20)
    max_epochs = trial.suggest_int("max_epochs", 30, 50)

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=patience, verbose=False,
                                        mode="min")
    # pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    trainer = Trainer(callbacks=[early_stop_callback],
                      max_epochs=max_epochs,
                      log_every_n_steps=20,
                      accelerator="gpu", devices=devices,
                      strategy="ddp")

    lce_model = LCEModel(pmodel=policy_model, training_costs=costs_matrix_train,
                         validation_costs=costs_matrix_valid,
                         lr=lr, weight_decay=weight_decay, optimizer_name=optimizer_name)
    trainer.fit(model=lce_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    train_loss = trainer.callback_metrics['val_loss'].item()
    return train_loss


def objective_ray(trial, train_loader, val_loader, devices, policy_model,
                  costs_matrix_train, costs_matrix_valid):
    lr = trial["lr"]

    max_epochs = trial["max_epochs"]
    weight_decay = trial["weight_decay"]
    optimizer_name = trial["optimizer_name"]
    patience = trial["patience"]
    # n_layers = trial["n_layers"]
    # dropout = trial["dropout"]

    # output_dims = []
    # if policy_model == None:
    #     output_dims = [np.random.randint(3, 24) for i in range(n_layers)]
    #     print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", output_dims)
    #     input_dim = 24
    #     num_classes = 3
    #     layers = []
    #     for output_dim in output_dims:
    #         layers.append(nn.Linear(input_dim, output_dim))
    #         layers.append(nn.BatchNorm1d(output_dim))
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(dropout))
    #         input_dim = output_dim
    #
    #     layers.append(nn.Linear(input_dim, num_classes))
    #     layers.append(nn.Sigmoid())
    #
    #     policy_model = nn.Sequential(*layers)

    print("################################################### Policy Model", policy_model)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=patience, verbose=False,
                                        mode="min")
    # pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    trainer = Trainer(callbacks=[early_stop_callback],
                      max_epochs=max_epochs,
                      log_every_n_steps=20,
                      accelerator="gpu", devices=devices,
                      strategy="ddp")

    lce_model = LCEModel(pmodel=policy_model, training_costs=costs_matrix_train,
                         validation_costs=costs_matrix_valid,
                         lr=lr, weight_decay=weight_decay, optimizer_name=optimizer_name)
    trainer.fit(model=lce_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    val_loss = trainer.callback_metrics['val_loss'].item()
    # tune.report(val_loss=val_loss)
    ray.train.report({"val_loss": val_loss})
    # ray.train.report({"output_dims": output_dims})


    # return val_loss


class LCE_Policy:
    """A class for learning a policy with the lce surrogate loss function using a B-Learner CATE estimtor.

    Parameters
    ----------
    propensity_model : classification model (scikit-learn or other)
    Estimator for Pr[A=1 | X=x].  Must implement `fit` and `predict_proba` methods.
    quantile_plus_model : quantile regression model (e.g. RandomForestQuantileRegressor)
    Estimator for the 1-tau conditional quantile. Must implement `fit` and `predict` methods.
    quantile_minus_model : quantile regression model (e.g. RandomForestQuantileRegressor)
    Estimator for the tau conditional quantile. Must implement `fit` and `predict` methods.
    mu_model : regression model (scikit-learn or other)
    Estimator for the conditional outcome E[Y | X=x, A=a] when `use_rho=False` or for the modified
    conditional outcome rho_+(x, a)=E[Gamma^{-1}Y+(1-Gamma^{-1}){q + 1/1(1-tau)*max{Y-q, 0}} | X=x, A=a]
    Must implement `fit` and `predict` methods.
    cvar_plus_model : superquantile model (default=None)
    Estimator for the conditional right tau tail CVaR when `use_rho=False`. Must implement `fit` and `predict` methods.
    Only used when `use_rho=False`.
    cvar_minus_model : superquantile model (default=None)
    Estimator for the conditional left tau tail CVaR when `use_rho=False`. Must implement `fit` and `predict` methods.
    Only used when `use_rho=False`.
    cate_bounds_model: the final stage model for estimating CATE bounds for the B-Learner estimator
    polic_model: the model used to learn the policy
    tau_hat: (default=None): The CATE Bounds and CAPO bounds if already calculated by the user.Otherwise, when `tau_hat=None`,
    then the recieved nuisance models above are used to calculate the bounds.
    use_rho :  bool (default=False)
    Whether to construct rho using a direct regression with plug-in quantiles (`use_rho=True`) or to estimate rho by
    estimating the conditional outcome and conditional CVaR models separately (`use_rho=False`).
    gamma : float, >=1
    Sensitivity model parameter. Must be greater than 1.
    higher_better: bool (default=True)
    Whether the convention for better outcomes is "higher is better" ('higher_better=True') or the opposite
    ('higher_better=False')
    """
    def __init__(self,
                 policy_model,
                 propensity_model=None,
                 quantile_plus_model=None,
                 quantile_minus_model=None,
                 cvar_plus_model=None,
                 cvar_minus_model=None,
                 mu_model=None,
                 cate_bounds_model=None,
                 tau_hat=None,
                 use_rho=False,
                 gamma=1,
                 higher_better=True,
                 baseline_p0=False,
                 with_deferral=True):
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.use_rho = use_rho
        self.higher_better = higher_better
        self.with_deferral = with_deferral
        if tau_hat:
            self.propensity_model = clone(propensity_model, safe=False)
            self.quantile_plus_model = clone(quantile_plus_model, safe=False)
            self.quantile_minus_model = clone(quantile_minus_model, safe=False)
            self.mu_model = clone(mu_model, safe=False)
            self.cvar_plus_model = clone(cvar_plus_model, safe=False)
            self.cvar_minus_model = clone(cvar_minus_model, safe=False)
            self.cate_bounds_model = clone(cate_bounds_model, safe=False)
        self.policy_model = clone(policy_model, safe=False)

        self.tau_hat = tau_hat

        self.bounds_model = None
        self.policy_trainer = None
        self.baseline_p0 = baseline_p0

    def _get_bounds(self, bounds_est, ds):
        x = ds.x

        # CATE Bounds
        tau_bottom, tau_top = bounds_est.effect(x)
        tau_bottom = torch.Tensor((tau_bottom.reshape(-1, 1)).transpose())
        tau_top = torch.Tensor((tau_top.reshape(-1, 1)).transpose())
        tau_mean = torch.Tensor(
            ((bounds_est.mu1(x) - bounds_est.mu0(x)).reshape(-1, 1)).transpose().tolist())

        # Outcome bounds
        Y_0_bottom, Y_0_top, Y_1_bottom, Y_1_top = bounds_est.outcome_bounds(x)
        Y_0_bottom = torch.Tensor((Y_0_bottom.reshape(-1, 1)).transpose())
        Y_0_top = torch.Tensor((Y_0_top.reshape(-1, 1)).transpose())
        Y_1_bottom = torch.Tensor((Y_1_bottom.reshape(-1, 1)).transpose())
        Y_1_top = torch.Tensor((Y_1_top.reshape(-1, 1)).transpose())

        return tau_bottom, tau_mean, tau_top, Y_0_bottom, Y_0_top, Y_1_bottom, Y_1_top

    def _fit_bounds_model(self, ds_train, ds_valid=None):
        X_train = ds_train.x
        Y_train = ds_train.y
        A_train = ds_train.t

        if ds_valid is None:
            X_val = None
            Y_val = None
            A_val = None
        else:
            X_val = ds_valid.x
            Y_val = ds_valid.y
            A_val = ds_valid.t

        # X_test = ds_test.x

        cate_bounds_est = BLearner(propensity_model=self.propensity_model,
                                   quantile_plus_model=self.quantile_plus_model,
                                   quantile_minus_model=self.quantile_minus_model,
                                   mu_model=self.mu_model,
                                   cvar_plus_model=self.cvar_plus_model,
                                   cvar_minus_model=self.cvar_minus_model,
                                   cate_bounds_model=self.cate_bounds_model,
                                   use_rho=self.use_rho,
                                   gamma=self.gamma)

        cate_bounds_est.fit(X=X_train, A=A_train, Y=Y_train, X_val=X_val,
                            A_val=A_val, Y_val=Y_val)
        self.bounds_model = cate_bounds_est

    def _get_deferral_costs(self, y_expert, a_expert, y0_hat, y1_hat):
        deferral_costs = []
        for j in range(len(a_expert)):
            if a_expert[j] == 0:
                deferral_costs.append(y1_hat[j] - y_expert[j])
            else:
                deferral_costs.append(y0_hat[j] - y_expert[j])

        return torch.Tensor(deferral_costs).unsqueeze(0)

    def _get_deferral_costs_try(self, y_expert, a_expert, y0_hat, y1_hat, C0, C1):
        deferral_costs = []
        for j in range(len(a_expert)):
            if a_expert[j] == 0:
                deferral_costs.append(C1[j], y1_hat[j] - y_expert[j])
            else:
                deferral_costs.append(C0[j], y0_hat[j] - y_expert[j])

        return torch.Tensor(deferral_costs).unsqueeze(0)

    def _calc_costs_matrix(self, ds, a_expert=None, y_expert=None):

        if a_expert is None and y_expert is None:
            y_expert = ds.y
            a_expert = ds.t
        # # Current expert's policy and it's outcome
        # y_expert = ds.y
        # a_expert = ds.t

        tau_bottom, tau_mean, tau_top, \
        Y_0_bottom, Y_0_top, \
        Y_1_bottom, Y_1_top = self._get_bounds(self.bounds_model, ds)

        # Calculating the costs
        if self.higher_better:
            treatment_costs = (Y_0_top - Y_1_bottom)
            control_costs = (Y_1_top - Y_0_bottom)

            deferral_costs_opt = self._get_deferral_costs(y0_hat=Y_0_bottom.squeeze(0),
                                                          y1_hat=Y_1_bottom.squeeze(0),
                                                          y_expert=y_expert, a_expert=a_expert)

            deferral_costs_cons = self._get_deferral_costs(y0_hat=Y_0_top.squeeze(0),
                                                           y1_hat=Y_1_top.squeeze(0),
                                                           y_expert=y_expert, a_expert=a_expert)

            costs_matrix = (np.concatenate(((np.concatenate((control_costs, treatment_costs
                                                             ), axis=0)), deferral_costs_opt),
                                           axis=0)).transpose()

        else:
            treatment_costs = Y_1_top - Y_0_bottom
            control_costs = Y_0_top - Y_1_bottom

            deferral_costs_cons = -1 * self._get_deferral_costs(y0_hat=Y_0_bottom.squeeze(0),
                                                                y1_hat=Y_1_bottom.squeeze(0),
                                                                y_expert=y_expert, a_expert=a_expert)

            deferral_costs_opt = -1 * self._get_deferral_costs(y0_hat=Y_0_top.squeeze(0),
                                                               y1_hat=Y_1_top.squeeze(0),
                                                               y_expert=y_expert, a_expert=a_expert)

            costs_matrix = (np.concatenate(((np.concatenate((control_costs, treatment_costs
                                                             ), axis=0)), deferral_costs_cons),
                                           axis=0)).transpose()

            costs_matrix_opt = (np.concatenate(((np.concatenate((control_costs, treatment_costs
                                                                 ), axis=0)), deferral_costs_opt),
                                               axis=0)).transpose()


        return costs_matrix

    def _calc_costs_matrix_outcomes(self, ds, Y_0_bottom, Y_0_top, Y_1_bottom, Y_1_top, a_expert=None, y_expert=None):

        if a_expert is None and y_expert is None:
            y_expert = ds.y
            a_expert = ds.t

        # Calculating the costs
        if self.higher_better:
            treatment_costs = (Y_0_top - Y_1_bottom)
            control_costs = (Y_1_top - Y_0_bottom)

            if not self.with_deferral:
                costs_matrix_pess = (np.concatenate((control_costs, treatment_costs
                                                ), axis=0)).transpose()
                return costs_matrix_pess


            deferral_costs_cons = self._get_deferral_costs(y0_hat=Y_0_bottom.squeeze(0),
                                                           y1_hat=Y_1_bottom.squeeze(0),
                                                           y_expert=y_expert, a_expert=a_expert)

            # deferral_costs_cons_try = self._get_deferral_costs_try(y0_hat=Y_0_bottom.squeeze(0),
            #                                                        y1_hat=Y_1_bottom.squeeze(0),
            #                                                        y_expert=y_expert, a_expert=a_expert,
            #                                                        C0=control_costs.squeeze(0),
            #                                                        C1=treatment_costs.squeeze(0))

            deferral_costs_opt = self._get_deferral_costs(y0_hat=Y_0_top.squeeze(0),
                                                          y1_hat=Y_1_top.squeeze(0),
                                                          y_expert=y_expert, a_expert=a_expert)

            costs_matrix = (np.concatenate(((np.concatenate((control_costs, treatment_costs
                                                             ), axis=0)), deferral_costs_cons),
                                           axis=0)).transpose()
            # costs_matrix_fact = (np.concatenate(((np.concatenate((control_costs_fact, treatment_costs_fact
            #                                                  ), axis=0)), deferral_costs_opt),
            #                                axis=0)).transpose()
        else:
            treatment_costs = Y_1_top - Y_0_bottom
            control_costs = Y_0_top - Y_1_bottom

            deferral_costs_cons = -1 * self._get_deferral_costs(y0_hat=Y_0_bottom.squeeze(0),
                                                                y1_hat=Y_1_bottom.squeeze(0),
                                                                y_expert=y_expert, a_expert=a_expert)

            deferral_costs_opt = -1 * self._get_deferral_costs(y0_hat=Y_0_top.squeeze(0),
                                                               y1_hat=Y_1_top.squeeze(0),
                                                               y_expert=y_expert, a_expert=a_expert)

            costs_matrix = (np.concatenate(((np.concatenate((control_costs, treatment_costs
                                                             ), axis=0)), deferral_costs_cons),
                                           axis=0)).transpose()

            costs_matrix_opt = (np.concatenate(((np.concatenate((control_costs, treatment_costs
                                                                 ), axis=0)), deferral_costs_opt),
                                               axis=0)).transpose()

        return costs_matrix

    def fit(self, ds_train, ds_valid,
            batch_size=32, patience=5, max_epochs=100, lr=0.001, weight_decay=0.0001, devices=[0]):
        a_expert_train = None
        y_expert_train = None
        a_expert_valid = None
        y_expert_valid = None
        if self.baseline_p0:
            a_expert_train = np.zeros(len(ds_train.t))
            y_expert_train = ds_train.y0
            a_expert_valid = np.zeros(len(ds_valid.t))
            y_expert_valid = ds_valid.y0

        if self.tau_hat is None:
            self._fit_bounds_model(ds_train=ds_train, ds_valid=ds_valid)
            costs_matrix_train = self._calc_costs_matrix(ds=ds_train,
                                                         a_expert=a_expert_train,
                                                         y_expert=y_expert_train)
            costs_matrix_valid = self._calc_costs_matrix(ds=ds_valid,
                                                         a_expert=a_expert_valid,
                                                         y_expert=y_expert_valid)
        else:
            _, _, _, \
            _, _, _, \
            _, _, _, \
            Y_0_bottom_train, Y_0_top_train, Y_1_bottom_train, Y_1_top_train, \
            _, _, _, _, \
            Y_0_bottom_val, Y_0_top_val, Y_1_bottom_val, Y_1_top_val = extract_results(self.tau_hat)

            costs_matrix_train = self._calc_costs_matrix_outcomes(ds=ds_train,
                                                                  Y_0_bottom=Y_0_bottom_train,
                                                                  Y_0_top=Y_0_top_train,
                                                                  Y_1_bottom=Y_1_bottom_train,
                                                                  Y_1_top=Y_1_top_train,
                                                                  a_expert=a_expert_train,
                                                                  y_expert=y_expert_train)
            costs_matrix_valid = self._calc_costs_matrix_outcomes(ds=ds_valid,
                                                                  Y_0_bottom=Y_0_bottom_val,
                                                                  Y_0_top=Y_0_top_val,
                                                                  Y_1_bottom=Y_1_bottom_val,
                                                                  Y_1_top=Y_1_top_val,
                                                                  a_expert=a_expert_valid,
                                                                  y_expert=y_expert_valid)

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(ds_valid, batch_size=8, shuffle=False, drop_last=True)

        config = {
            "lr": tune.loguniform(1e-4, 0.1),
            "optimizer_name": tune.choice(['SGD', 'Adam', 'AdamW']),
            "weight_decay": tune.loguniform(1e-10, 1e-3),
            "patience": tune.randint(5, 20),
            "max_epochs": tune.randint(30, 60),
            # "n_layers": tune.randint(1, 5),
            # "dropout":  tune.loguniform(0.2, 0.6),
        }
        # a config for cases the model is None, and we need to figure out the MLP archetcture
        # config = {
        #     "lr": tune.loguniform(1e-4, 0.1),
        #     "optimizer_name": tune.choice(['SGD', 'Adam', 'AdamW']),
        #     "weight_decay": tune.loguniform(1e-10, 1e-3),
        #     "patience": tune.randint(3, 5),
        #     "max_epochs": tune.randint(10, 30),
        #     "n_layers": tune.randint(1, 5),
        #     "dropout":  tune.loguniform(0.2, 0.6),
        # }

        objective_with_params_ray = lambda trial: objective_ray(trial=trial,
                                                                train_loader=train_loader,
                                                                val_loader=valid_loader,
                                                                devices=devices,
                                                                policy_model=self.policy_model,
                                                                costs_matrix_train=costs_matrix_train,
                                                                costs_matrix_valid=costs_matrix_valid)
        checkpoint_config = CheckpointConfig(
            num_to_keep=5,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min"
        )
        # run_config = RunConfig(
        #     checkpoint_config=checkpoint_config,
        #     name="my_lce_experiment"
        # )
        # study = tune.run(
        #     objective_with_params_ray,
        #     config=config,
        #     num_samples=10,  # 50
        #     resources_per_trial={"gpu": 0.3},
        #     run_config=run_config
        # )
        study = tune.run(
            objective_with_params_ray,
            config=config,
            num_samples=10,  # 50
            resources_per_trial={"gpu": 0.3},
            keep_checkpoints_num=1,
            checkpoint_score_attr="val_loss",
        )
        # Ray best params
        # best_trial = study.best_trial
        best_trial = study.get_best_trial(metric="val_loss", mode="min")

        # # Gets best checkpoint for trial based on val loss
        # best_checkpoint = study.get_best_checkpoint(best_trial, metric="val_loss")

        best_params = best_trial.config

        patience = best_params["patience"]
        max_epochs = best_params["max_epochs"]

        lr = best_params["lr"]
        optimizer_name = best_params["optimizer_name"]
        weight_decay = best_params["weight_decay"]
        # dropout = best_params["dropout"]

        # # Create and train the final model with the best hyperparameters
        # if self.policy_model == None:
        #     # Build the final model
        #     output_dims = best_trial.last_result["output_dims"]
        #     input_dim = 24
        #     num_classes = 3
        #     layers = []
        #     for output_dim in output_dims:
        #         layers.append(nn.Linear(input_dim, output_dim))
        #         layers.append(nn.BatchNorm1d(output_dim))
        #         layers.append(nn.ReLU())
        #         layers.append(nn.Dropout(dropout))
        #         input_dim = output_dim
        #
        #     layers.append(nn.Linear(input_dim, num_classes))
        #     layers.append(nn.Sigmoid())
        #
        #     self.policy_model = nn.Sequential(*layers)

        # optimizer_name = "AdamW"

        final_model = LCEModel(pmodel=self.policy_model,
                               training_costs=costs_matrix_train,
                               validation_costs=costs_matrix_valid,
                               lr=lr,
                               weight_decay=weight_decay,
                               optimizer_name=optimizer_name)
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=patience, verbose=False,
                                            mode="min")
        trainer = Trainer(callbacks=[early_stop_callback],
                          max_epochs=max_epochs, log_every_n_steps=8, accelerator="gpu", devices=devices
                          , strategy='ddp', val_check_interval=1.0)
        trainer.fit(final_model, train_dataloaders=train_loader, val_dataloaders=valid_loader,
                    )

        # Access the best model from the trainer
        self.policy_model = trainer.model
        self.policy_trainer = trainer

    def predict(self, ds_test):
        test_loader = DataLoader(ds_test, batch_size=15, shuffle=False, drop_last=False)  # bs=15
        lce_pi = self.policy_model.predict(test_loader)

        return lce_pi

    def get_cate_bounds(self, ds):
        tau_bottom, tau_mean, tau_top, \
        Y_0_bottom, Y_0_top, \
        Y_1_bottom, Y_1_top = self._get_bounds(self.bounds_model, ds)

        return tau_bottom, tau_mean, tau_top
