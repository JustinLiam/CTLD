import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import Tensor
import pandas as pd
from models import *
from utils import *
from data import *
import argparse

# python3 test_person.py --nepoch 1000 --lr 1e-2 --gamma 1.5 --C 0 --nrep 10
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default = 'heloc')
parser.add_argument("--nepoch", type=int, default = 1000)
parser.add_argument("--lr", type=float, default = 1e-2)
parser.add_argument('--wgamma', nargs='+', type=float, default = [1.0, 1.0, 1.0])
parser.add_argument("--gamma", nargs='+', type=float, default = [2.5, 2.5, 2.5])
parser.add_argument("--C", type=float, default = 0.1)
parser.add_argument("--nrep", type=int, default = 10)
parser.add_argument("--folder", type=str, default = 'exp1')
parser.add_argument("--hidd", type=int, default = 2)

args = parser.parse_args()
nepoch = args.nepoch
lr = args.lr
gamma = np.array(args.gamma)
c = args.C
nrep = args.nrep
folder_name = args.folder
wgamma = np.array(args.wgamma)
dataset = args.dataset
hidd = args.hidd

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

path = os.path.join('./log',folder_name)
if not os.path.exists(path):
    os.mkdir(path)
    print(f"Folder {path} created successfully.")
else:
    print("Folder already exists.")


n = 2000
ntest = 10000
# parameters
rho = np.asarray([1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0, 1 / np.sqrt(3)])  # normalize to unit 0.5
rho = rho / (np.dot(rho, rho) * 2)

beta_cons = 2.5
beta_x = np.asarray([0, .5, -0.5, 0, 0, 0]) # beta_0
beta_x_T = np.asarray([-1.5, 1, -1.5, 1., 0.5, 0]) # beta_treat
beta_T = np.asarray([0, .75, -.5, 0, -1, 0, 0]) # beta
beta_T_conf = np.asarray([0, .75, -.5, 0, -1, 0])
mu_x = np.asarray([-1, .5, -1, 0, -1]);

alpha = -2
w = 1.5

# true specified human model
Gamma = wgamma
# used HBM
usedGamma = gamma

result = pd.DataFrame(columns = ['Human','AO','ConfAO','HAI','ConfHAI','ConfHAIPerson'])
for i in range(nrep):
    seed_everything(i)
    if dataset == 'synthetic':
        x_, u, T, Y, true_Q_obs, q0, Y_all, q0_all, hid, T_h = generate_log_data_pl(mu_x, n, beta_cons, beta_x, beta_x_T, beta_T_conf, Gamma, alpha,w)
        x_test, u_test, T_test, Y_test, true_Q_obs_test, q0_test, Y_all_test, q0_all_test, hid_test, T_h_test = generate_log_data_pl(mu_x, ntest, beta_cons, beta_x, beta_x_T, beta_T_conf, Gamma, alpha,w)
    elif dataset == 'heloc':
        df = pd.read_csv('./heloc_dataset_v1 (1).csv')
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

        x_, u, T, Y, true_Q_obs, q0, Y_all, q0_all, hid, T_h = generate_log_data_heloc(df, feature_cols, credit_col, n,
                                                                                       Gamma, seed=i)
        x_test, u_test, T_test, Y_test, true_Q_obs_test, q0_test, Y_all_test, q0_all_test, hid_test, T_h_test = generate_log_data_heloc(
            df, feature_cols, credit_col, ntest, Gamma, seed=1000 + i)

    print(((T==1)/true_Q_obs).mean())
    print(((T==0)/true_Q_obs).mean())
    print(((T_test==1)/true_Q_obs_test).mean())
    print(((T_test==0)/true_Q_obs_test).mean())

    d = x_.shape[1]  # dimension of x
    batchsize = x_.shape[0]
    dataset = TensorDataset(Tensor(x_).to(device),
                            Tensor(q0_all).to(device),
                            Tensor(T).to(device),
                            Tensor(Y).to(device),
                            )
    loader = DataLoader(dataset, batch_size = batchsize)

    # train a policy
    model = train_ips(d, 2, loader, nepoch, lr = lr, hidd= hidd)
    # test performance
    risk1 = test_reward(model, x_test, Y_all_test, control= True)
    print(f'model only risk is {risk1}')

    # train conf policy
    model, con_ind = train_confips(d, 2, loader, nepoch, lr = lr, gamma = usedGamma[hid], pre_model= None, hidd= hidd)
    risk2 = test_reward(model, x_test, Y_all_test, control= True, con_ind= con_ind)
    print(f'model only conf risk is {risk2}')

    import copy
    modelconf = copy.deepcopy(model)

    model, router = train_hai(modelconf, d, 2, loader, nepoch, lr = lr, C = c, hidd= hidd)
    risk3 = test_hai(model, router, x_test, Y_all_test, T_test, C = c, control= True)
    print(f'human ai risk is {risk3}')

    model, router, con_ind = train_confhai(modelconf, router, d, 2, loader, nepoch, lr = lr, gamma = usedGamma[hid], C = c, hidd= hidd)
    risk4 = test_hai(model, router, x_test, Y_all_test, T_test, C = c, control= True, con_ind= con_ind)
    print(f'human ai conf risk is {risk4}')

    dataset = TensorDataset(Tensor(x_).to(device),
                            Tensor(q0_all).to(device),
                            Tensor(T).to(device),
                            Tensor(Y).to(device),
                            Tensor(hid).to(device),
                            )
    loader = DataLoader(dataset, batch_size = batchsize)

    seed_everything(i+81)
    modelconf1 = copy.deepcopy(modelconf)
    model, router, con_ind = train_confhai_person(modelconf, router, d, 2, loader, num_epochs=nepoch, lr = lr, gamma = usedGamma[hid], nump = len(Gamma), C = c)
    risk5 = test_hai_person(model, router, x_test, Y_all_test, T_h_test, T_test, hid_test, nump=len(Gamma), C = c, control= True, con_ind= con_ind)
    print(f'human ai conf person risk is {risk5}')

    result = pd.concat([result, pd.DataFrame([{
        'Human': Y_test.mean() - Y_all_test[:, 0].mean() + c,
        'AO': risk1,
        'ConfAO': risk2,
        'HAI': risk3,
        'ConfHAI': risk4,
        'ConfHAIPerson': risk5
    }])], ignore_index=True)

    result.to_csv(f'{path}/result_real_{args}.csv')
    print(result.mean())
    print(result.std() / np.sqrt(nrep))
