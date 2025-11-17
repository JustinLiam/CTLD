import os
import torch
import pyreadr
import requests
import numpy as np
import pandas as pd

from pathlib import Path

from torch.utils import data

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

_CONTINUOUS_COVARIATES = [
    "bw",
    "b.head",
    "preterm",
    "birth.o",
    "nnhealth",
    "momage",
]

_BINARY_COVARIATES = [
    "sex",
    "twin",
    "mom.lths",
    "mom.hs",
    "mom.scoll",
    "cig",
    "first",
    "booze",
    "drugs",
    "work.dur",
    "prenatal",
    "ark",
    "ein",
    "har",
    "mia",
    "pen",
    "tex",
    "was",
]

_HIDDEN_COVARIATE = [
    "b.marr",
]


class IHDP(data.Dataset):
    def __init__(self, root, split, mode, seed, hidden_confounding, beta_u=None):
        # root = Path.home() / "quince_datasets" if root is None else Path(root)
        data_path = Path(__file__).parent.parent.parent / "datasets" / "ihdp" / "ihdp.RData"
        print("loaded data from", data_path)
        # Download data if necessary
        if not data_path.exists():
            root.mkdir(parents=True, exist_ok=True)
            r = requests.get(
                "https://github.com/vdorie/npci/raw/master/examples/ihdp_sim/data/ihdp.RData"
            )
            with open(data_path, "wb") as f:
                f.write(r.content)
        df = pyreadr.read_r(data_path)["ihdp"]
        # Make observational as per Hill 2011
        #  移除所有"接受治疗且母亲是非白人"的样本
        df = df[~((df["treat"] == 1) & (df["momwhite"] == 0))]
        df = df[
            _CONTINUOUS_COVARIATES + _BINARY_COVARIATES + _HIDDEN_COVARIATE + ["treat"]
        ]
        # Standardize continuous covariates
        df[_CONTINUOUS_COVARIATES] = preprocessing.StandardScaler().fit_transform(
            df[_CONTINUOUS_COVARIATES]
        )
        # Generate response surfaces
        rng = np.random.default_rng(seed)
        x = df[_CONTINUOUS_COVARIATES + _BINARY_COVARIATES]
        u = df[_HIDDEN_COVARIATE]
        t = df["treat"]
        beta_x = rng.choice(
            [0.0, 0.1, 0.2, 0.3, 0.4], size=(24,), p=[0.6, 0.1, 0.1, 0.1, 0.1]
        )
        beta_u = (
            rng.choice(
                [0.1, 0.2, 0.3, 0.4, 0.5], size=(1,), p=[0.2, 0.2, 0.2, 0.2, 0.2]
            )
            if beta_u is None
            else np.asarray([beta_u])
        )
        mu0 = np.exp((x + 0.5).dot(beta_x) + (u + 0.5).dot(beta_u))
        df["mu0"] = mu0
        mu1 = (x + 0.5).dot(beta_x) + (u + 0.5).dot(beta_u)
        omega = (mu1[t == 1] - mu0[t == 1]).mean(0) - 4
        mu1 -= omega
        df["mu1"] = mu1
        eps = rng.normal(size=t.shape)
        y0 = mu0 + eps
        df["y0"] = y0
        y1 = mu1 + eps
        df["y1"] = y1
        y = t * y1 + (1 - t) * y0
        df["y"] = y
        # Train test split
        df_train, df_test = model_selection.train_test_split(
            df, test_size=0.1, random_state=seed
        )
        self.mode = mode
        self.split = split
        # Set x, y, and t values
        self.y_mean = (
            df_train["y"].to_numpy(dtype="float32").mean(keepdims=True)
            if mode == "mu"
            else np.asarray([0.0], dtype="float32")
        )
        self.y_std = (
            df_train["y"].to_numpy(dtype="float32").std(keepdims=True)
            if mode == "mu"
            else np.asarray([1.0], dtype="float32")
        )
        covars = _CONTINUOUS_COVARIATES + _BINARY_COVARIATES
        covars = covars + _HIDDEN_COVARIATE if not hidden_confounding else covars
        self.dim_input = len(covars)
        self.dim_treatment = 1
        self.dim_output = 1
        if self.split == "test":
            self.x = df_test[covars].to_numpy(dtype="float32")
            self.u = df_test[_HIDDEN_COVARIATE].to_numpy(dtype="float32")
            self.t = df_test["treat"].to_numpy(dtype="float32")
            self.mu0 = df_test["mu0"].to_numpy(dtype="float32")
            self.mu1 = df_test["mu1"].to_numpy(dtype="float32")
            self.y0 = df_test["y0"].to_numpy(dtype="float32")
            self.y1 = df_test["y1"].to_numpy(dtype="float32")
            df = df_test
            self.df = df
            if mode == "mu":
                self.y = self.mu1 - self.mu0
            elif mode == "pi":
                self.y = self.t
            else:
                raise NotImplementedError("Not a valid mode")
        else:
            df_train, df_valid = model_selection.train_test_split(
                df_train, test_size=0.3, random_state=seed
            )
            if split == "train":
                df = df_train
                self.df = df
            elif split == "valid":
                df = df_valid
                self.df = df
            else:
                raise NotImplementedError("Not a valid dataset split")
            self.x = df[covars].to_numpy(dtype="float32")
            self.u = df[_HIDDEN_COVARIATE].to_numpy(dtype="float32")
            self.t = df["treat"].to_numpy(dtype="float32")
            self.mu0 = df["mu0"].to_numpy(dtype="float32")
            self.mu1 = df["mu1"].to_numpy(dtype="float32")
            self.y0 = df["y0"].to_numpy(dtype="float32")
            self.y1 = df["y1"].to_numpy(dtype="float32")
            if mode == "mu":
                self.y = df["y"].to_numpy(dtype="float32")
            elif mode == "pi":
                self.y = self.t
            else:
                raise NotImplementedError("Not a valid mode")


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # inputs = (
        #     torch.from_numpy(self.x[idx]).float()
        #     if self.mode == "pi"
        #     else torch.from_numpy(np.hstack([self.x[idx], self.t[idx]])).float()
        # )
        # targets = torch.from_numpy((self.y[idx] - self.y_mean) / self.y_std).float()
        inputs = (
            torch.from_numpy(self.x[idx]).float()
            if self.mode == "pi"
            else torch.from_numpy(self.x[idx]).float()
        )
        targets_t = torch.from_numpy(np.asarray(self.t[idx])).float()
        targets = torch.from_numpy((self.y[idx] - self.y_mean) / self.y_std).float()
        return inputs, targets_t, idx, targets




def policy_value(pi, y1, y0):
    return (pi * y1 + (1 - pi) * y0).mean()

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    trial = 28
    ihdp_train_ds = IHDP(root=None, split="train", mode='mu', seed=trial, hidden_confounding=True)
    ihdp_val_ds = IHDP(root=None, split="valid", mode='mu', seed=trial, hidden_confounding=True)
    ihdp_test_ds = IHDP(root=None, split="test", mode='mu', seed=trial, hidden_confounding=True)

    current_expert_train = ihdp_train_ds.t
    current_expert_test = ihdp_test_ds.t

    tau_true_test = torch.tensor(ihdp_test_ds.mu1 - ihdp_test_ds.mu0)
    tau_true_train = torch.tensor(ihdp_train_ds.mu1 - ihdp_train_ds.mu0)

    oracle_policy_train = (tau_true_train > 0) * 1
    oracle_policy_test = (tau_true_test > 0) * 1

    print("before")
    print(policy_value(pi=current_expert_train, y1=ihdp_train_ds.y1, y0=ihdp_train_ds.y0))
    cig_train = np.asarray(ihdp_train_ds.df["cig"])
    for i in range(len(current_expert_train)):
        if cig_train[i] == 0:
            current_expert_train[i] = oracle_policy_train[i]

    print("after")
    print(policy_value(pi=current_expert_train, y1=ihdp_train_ds.y1, y0=ihdp_train_ds.y0))


    print("before")
    print(policy_value(pi=current_expert_test, y1=ihdp_test_ds.y1, y0=ihdp_test_ds.y0))
    cig_test = np.asarray(ihdp_test_ds.df["cig"])
    for i in range(len(current_expert_test)):
        if cig_test[i] == 0:
            current_expert_test[i] = oracle_policy_test[i]

    print("after")
    print(policy_value(pi=current_expert_test, y1=ihdp_test_ds.y1, y0=ihdp_test_ds.y0))

    seed = 42
    data_view = {
        'y0_observed': ihdp_train_ds.y0.flatten(),
        'y1_observed': ihdp_train_ds.y1.flatten(),
        'mu0_true': ihdp_train_ds.mu0.flatten(),
        'mu1_true': ihdp_train_ds.mu1.flatten(),
        'CATE_true': (ihdp_train_ds.mu1 - ihdp_train_ds.mu0).flatten(),
        'expert_action': ihdp_train_ds.t.flatten(),
    }
    df_view = pd.DataFrame(data_view)

    print("\n--- First 5 rows of generated data (Original Method) ---")
    print(df_view.head())
    print("\n--- Descriptive Statistics (Original Method) ---")
    print(df_view.describe())

    # --- 2. 估算真实的混杂因子 Lambda ---
    print("\n--- Estimating the 'True' Confounding Factor Lambda (Original Method) ---")
    X_obs = ihdp_train_ds.x
    X_full = np.concatenate([ihdp_train_ds.x, ihdp_train_ds.u], axis=1)
    T_expert = ihdp_train_ds.t

    model_e_xu = LogisticRegression(solver='liblinear', C=1.0).fit(X_full, T_expert)
    e_xu_hat = model_e_xu.predict_proba(X_full)[:, 1]

    model_e_x = LogisticRegression(solver='liblinear', C=1.0).fit(X_obs, T_expert)
    e_x_hat = model_e_x.predict_proba(X_obs)[:, 1]

    eps = 1e-8
    odds_true = e_xu_hat / (1 - e_xu_hat + eps)
    odds_obs = e_x_hat / (1 - e_x_hat + eps)
    odds_ratio = odds_true / (odds_obs + eps)

    deviation = np.maximum(odds_ratio, 1 / (odds_ratio + eps))
    empirical_lambda = np.max(deviation)

    print(f"\nThe estimated 'True' Lambda for this dataset is: {empirical_lambda:.4f}")

    # --- 3. 可视化CATE分布 ---
    plt.figure(figsize=(8, 5))
    plt.hist(df_view['CATE_true'], bins=30, alpha=0.7)
    plt.axvline(0, color='r', linestyle='--', label='CATE = 0')
    plt.title(f"CATE Distribution (Original Method, Seed={seed})")
    plt.xlabel("Conditional Average Treatment Effect (CATE)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_dir = "plots_output"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"original_cate_dist_seed_{seed}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot of CATE distribution saved to: {file_path}")