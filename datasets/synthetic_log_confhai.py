import torch
import pyreadr
import requests
import numpy as np

from pathlib import Path

from torch.utils import data

from sklearn import preprocessing
from sklearn import model_selection

from models.crlogit.data_scenarios import *
from models.confhai.data import generate_log_data_pl, generate_log_data_heloc


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

        x_, u, T, Y, true_Q_obs, q0, Y_all, q0_all, hid, T_h = generate_log_data_pl(**dgp_params)

        self.x = x_
        self.t = T
        self.y = Y
        self.y0 = Y_all[:, 0]
        self.y1 = Y_all[:, 1]
        self.u = u

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.x[idx]).float()
        targets_t = torch.from_numpy(np.asarray(self.t[idx])).float()
        targets = torch.from_numpy(np.asarray(self.y[idx])).float()
        return inputs, targets_t, idx, targets

class HELOC_LOG(torch.utils.data.Dataset):
    def __init__(self, x, t, y, y_all, q0_all, hid=None, T_h=None, u=None):
        self.x = x
        self.t = t
        self.y = y
        self.y_all = y_all
        self.y0 = y_all[:, 0]
        self.y1 = y_all[:, 1]
        self.q0_all = q0_all
        self.hid = hid
        self.T_h = T_h
        self.u = u

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]).float(),
            torch.from_numpy(self.q0_all[idx]).float(),
            torch.tensor(int(self.t[idx])),
            torch.tensor(float(self.y[idx])),

        )