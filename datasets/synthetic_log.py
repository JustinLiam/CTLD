import torch
import pyreadr
import requests
import numpy as np

from pathlib import Path

from torch.utils import data

from sklearn import preprocessing
from sklearn import model_selection

from models.crlogit.data_scenarios import *

def generate_log_data(mu_x, n, beta_cons, beta_x, beta_x_T):
    # generate n datapoints from the same multivariate normal distribution
    d = len(mu_x)
    u = (np.random.rand(n) > 0.5)
    x = np.zeros([n, d])
    for i in range(n):
        x[i, :] = np.random.multivariate_normal(mean=mu_x * (2 * u[i] - 1), cov=np.eye(d))
    x_ = np.hstack([x, np.ones([n, 1])])
    # generate propensities
    true_Q = REAL_PROP_LOG(x_, u)
    T = np.array(np.random.uniform(size=n) < true_Q).astype(int).flatten()
    T = T.reshape([n, 1]);
    T_sgned = np.asarray([1 if T[i] == 1 else -1 for i in range(n)]).flatten()
    nominal_propensities_pos = logistic_pol_asgn(beta_T_conf, x_)

    q0 = np.asarray([nominal_propensities_pos[i] if T[i] == 1 else 1 - nominal_propensities_pos[i] for i in range(n)])
    true_Q_obs = np.asarray([true_Q[i] if T[i] == 1 else 1 - true_Q[i] for i in range(n)])
    Y = np.zeros(n)
    Y0 = np.zeros(n)
    Y1 = np.zeros(n)
    T1 = np.ones(n).reshape(-1, 1)
    T0 = np.zeros(n).reshape(-1, 1)
    # w = 1
    for i in range(n):
        Y[i] = T[i] * beta_cons + np.dot(beta_x.T, x_[i, :]) + np.dot(beta_x_T.T, x_[i, :] * T[i]) + alpha * (u[i]) * (
                2 * T[i] - 1) + w * u[i]
        Y0[i] = T0[i] * beta_cons + np.dot(beta_x.T, x_[i, :]) + np.dot(beta_x_T.T, x_[i, :] * T0[i]) + alpha * (
        u[i]) * (
                        2 * T0[i] - 1) + w * u[i]
        Y1[i] = T1[i] * beta_cons + np.dot(beta_x.T, x_[i, :]) + np.dot(beta_x_T.T, x_[i, :] * T1[i]) + alpha * (
        u[i]) * (
                        2 * T1[i] - 1) + w * u[i]

    # add random noise
    T = T.flatten()
    eps = np.random.randn(n)
    Y += 1 * eps
    Y0 += 1 * eps
    Y1 += 1 * eps
    return [x_, u, T, Y, Y0, Y1, true_Q_obs, q0]

class SYN_LOG(data.Dataset):
    def __init__(self, dgp_params, split, seed):

        [x_full, u, T_, Y_, Y0_, Y1_, true_Q_obs, q0] = generate_log_data(**dgp_params)

        # Train test split
        np.random.seed(seed)
        random.seed(seed)
        train_ind, test_ind = model_selection.train_test_split(list(range(len(Y_))), test_size=0.9)
        # train_ind, valid_ind = model_selection.train_test_split(train_ind, test_size=0.3)
        x_train = x_full[train_ind, :]
        t_train = T_[train_ind]
        y0_train = Y0_[train_ind]
        y1_train = Y1_[train_ind]
        y_train = Y_[train_ind]
        u_train = u[train_ind]

        # x_valid = x_full[valid_ind, :]
        # t_valid = T_[valid_ind]
        # y0_valid = Y0_[valid_ind]
        # y1_valid = Y1_[valid_ind]
        # y_valid = Y_[valid_ind]
        # u_valid = u[valid_ind]

        x_test = x_full[test_ind, :]
        t_test = T_[test_ind]
        y_test = Y_[test_ind]
        y0_test = Y0_[test_ind]
        y1_test = Y1_[test_ind]
        u_test = u[test_ind]

        self.split = split
        # Set x, y, and t values
        if self.split == "test":
            self.x = x_test
            self.t = t_test
            self.y = y_test
            self.y0 = y0_test
            self.y1 = y1_test
            self.u = u_test
        elif split == "train":
            self.x = x_train
            self.t = t_train
            self.y = y_train
            self.y0 = y0_train
            self.y1 = y1_train
            self.u = u_train
        # elif split == "valid":
        #     self.x = x_valid
        #     self.t = t_valid
        #     self.y = y_valid
        #     self.y0 = y0_valid
        #     self.y1 = y1_valid
        #     self.u = u_valid
        else:
            raise NotImplementedError("Not a valid dataset split")


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.x[idx]).float()
        targets_t = torch.from_numpy(np.asarray(self.t[idx])).float()
        targets = torch.from_numpy(np.asarray(self.y[idx])) .float()
        return inputs, targets_t, idx, targets

def generate_log_data_mod(mu_x, n, beta_cons, beta_x, beta_x_T):
    # generate n datapoints from the same multivariate normal distribution
    d = len(mu_x)
    xi = np.random.rand(n)
    u = (xi > 0.5)
    eta_ = 2.5
    alpha_ = -2
    w_ = 1.5
    beta_t = np.asarray([1.5, 1, 1.5, 1, 0.5, 0])
    beta = np.asarray([0, .5, 0.5, 0, 0, 0])
    beta_T_conf = np.asarray([0, .75, .5, 0, 1, 0])
    mu_x_ = np.asarray([1, 0.5, 1, 0, 1])
    x = np.zeros([n, d])

    for i in range(n):
        x[i, :] = np.random.multivariate_normal(mean=mu_x_ * (2 * u[i] - 1), cov=np.eye(d))
    x_ = np.hstack([x, np.ones([n, 1])])
    # generate propensities
    true_Q = REAL_PROP_LOG_mod(x_, u)
    T = np.array(np.random.uniform(size=n) < true_Q).astype(int).flatten()
    T = T.reshape([n, 1]);
    T_sgned = np.asarray([1 if T[i] == 1 else -1 for i in range(n)]).flatten()
    nominal_propensities_pos = logistic_pol_asgn(beta_T_conf, x_)

    q0 = np.asarray([nominal_propensities_pos[i] if T[i] == 1 else 1 - nominal_propensities_pos[i] for i in range(n)])
    true_Q_obs = np.asarray([true_Q[i] if T[i] == 1 else 1 - true_Q[i] for i in range(n)])
    Y = np.zeros(n)
    Y0 = np.zeros(n)
    Y1 = np.zeros(n)
    T1 = np.ones(n).reshape(-1, 1)
    T0 = np.zeros(n).reshape(-1, 1)
    # w = 1
    for i in range(n):
        Y[i] = np.dot(beta.T, x_[i, :]) + np.dot(beta_t.T, x_[i, :] * T[i]) + 0.5 * alpha_ * (u[i]) * T[i] + eta_ + w_ * u[i]

        Y0[i] = np.dot(beta.T, x_[i, :]) + np.dot(beta_t.T, x_[i, :] * T0[i]) + 0.5 * alpha_ * (u[i]) * T0[i] + eta_ + w_ * \
               u[i]
        # Y0[i] = T0[i] * beta_cons + np.dot(beta_x.T, x_[i, :]) + np.dot(beta_x_T.T, x_[i, :] * T0[i]) + alpha * (
        # u[i]) * (
        #                 2 * T0[i] - 1) + w * u[i]
        # Y1[i] = T1[i] * beta_cons + np.dot(beta_x.T, x_[i, :]) + np.dot(beta_x_T.T, x_[i, :] * T1[i]) + alpha * (
        # u[i]) * (
        #                 2 * T1[i] - 1) + w * u[i]
        Y1[i] = np.dot(beta.T, x_[i, :]) + np.dot(beta_t.T, x_[i, :] * T1[i]) + 0.5 * alpha_ * (u[i]) * T1[i] + eta_ + w_ * \
               u[i]

    # add random noise
    T = T.flatten()
    eps = np.random.randn(n)
    Y += 1 * eps
    Y0 += 1 * eps
    Y1 += 1 * eps
    return [x_, u, T, Y, Y0, Y1, true_Q_obs, q0]

class SYN_LOG_mod(data.Dataset):
    def __init__(self, dgp_params, split, seed):

        [x_full, u, T_, Y_, Y0_, Y1_, true_Q_obs, q0] = generate_log_data_mod(**dgp_params)

        # Train test split
        np.random.seed(seed)
        random.seed(seed)
        train_ind, test_ind = model_selection.train_test_split(list(range(len(Y_))), test_size=0.9)
        # train_ind, valid_ind = model_selection.train_test_split(train_ind, test_size=0.3)
        x_train = x_full[train_ind, :]
        t_train = T_[train_ind]
        y0_train = Y0_[train_ind]
        y1_train = Y1_[train_ind]
        y_train = Y_[train_ind]
        u_train = u[train_ind]

        # x_valid = x_full[valid_ind, :]
        # t_valid = T_[valid_ind]
        # y0_valid = Y0_[valid_ind]
        # y1_valid = Y1_[valid_ind]
        # y_valid = Y_[valid_ind]
        # u_valid = u[valid_ind]

        x_test = x_full[test_ind, :]
        t_test = T_[test_ind]
        y_test = Y_[test_ind]
        y0_test = Y0_[test_ind]
        y1_test = Y1_[test_ind]
        u_test = u[test_ind]

        self.split = split
        # Set x, y, and t values
        if self.split == "test":
            self.x = x_test
            self.t = t_test
            self.y = y_test
            self.y0 = y0_test
            self.y1 = y1_test
            self.u = u_test
        elif split == "train":
            self.x = x_train
            self.t = t_train
            self.y = y_train
            self.y0 = y0_train
            self.y1 = y1_train
            self.u = u_train
        # elif split == "valid":
        #     self.x = x_valid
        #     self.t = t_valid
        #     self.y = y_valid
        #     self.y0 = y0_valid
        #     self.y1 = y1_valid
        #     self.u = u_valid
        else:
            raise NotImplementedError("Not a valid dataset split")


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.x[idx]).float()
        targets_t = torch.from_numpy(np.asarray(self.t[idx])).float()
        targets = torch.from_numpy(np.asarray(self.y[idx])) .float()
        return inputs, targets_t, idx, targets


if __name__ == '__main__':
    trial = 28



