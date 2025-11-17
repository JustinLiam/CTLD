import numpy as np 
import pandas as pd 
import pdb 
import torch 
import random 
from scipy import stats 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def logistic_pol_asgn(theta, x):
    n = x.shape[0]
    theta = theta.flatten()
    if len(theta) == 1:
        logit = np.multiply(x, theta).flatten()
    else:
        logit = np.dot(x, theta).flatten()
    LOGIT_TERM_POS = np.ones(n)*1.0 / ( np.ones(n) + np.exp( -logit ))
    return LOGIT_TERM_POS

def real_risk(T, beta_cons, beta_x, beta_x_T, x, u, alpha, w):
    n = len(T)
    risk = np.zeros(n)
    for i in range(len(T)):
        #risk[i] = T[i] * beta_cons + np.dot(beta_x.T, x[i, :]) + np.dot(beta_x_T.T, x[i, :] * T[i]) + alpha * (u[i]) * ((2 * T[i] - 1)) + w * (u[i])
        #risk[i] = (T[i]+1) * beta_cons + np.dot(beta_x.T, x[i, :]) + 0.5 * np.dot(beta_x_T.T, x[i, :] * (T[i]+1)) + alpha * u[i] * T[i] + w * (u[i])
        #risk[i] = beta_cons * T[i] + np.dot(beta_x.T, x[i, :]) + np.dot(beta_x_T.T, x[i, :] * T[i]) + alpha * u[i] * (T[i]-0.5)*2 + w * (u[i])
        #Y_all[i,0] = beta_cons * 0 + np.dot(beta_x.T, x_[i, :]) + np.dot(beta_x_T.T, x_[i, :] * (0)) + alpha * u[i] * (-1) + w * (u[i])
        #Y_all[i,1] = beta_cons * 1 + np.dot(beta_x.T, x_[i, :]) + np.dot(beta_x_T.T, x_[i, :] * (1)) + alpha * u[i] * 1 + w * (u[i])   
        risk[i] = T[i] * beta_cons + np.dot(beta_x.T, x[i, :]) + np.dot(beta_x_T.T, x[i, :] * T[i]) + alpha * (u[i]) * ((2 * T[i] - 1)) + w * (u[i])
    return risk


def return_CATE_optimal_assignment(x, u, beta_cons, beta_x, beta_x_T, alpha,w):
    n = x.shape[0]
    risk_T_1 = real_risk(np.ones(n), beta_cons, beta_x, beta_x_T, x, u,alpha,w)
    risk_T_0 = real_risk(np.zeros(n), beta_cons, beta_x, beta_x_T,x, u,alpha,w)
    opt_T = [1 if risk_T_1[k] < risk_T_0[k] else 0 for k in range(n)]
    return opt_T

def REAL_PROP_LOG(x, u, beta_T_conf, beta_cons, beta_x, beta_x_T, Gamma = 3,alpha=-2,w=1.5):
    nominal_ = logistic_pol_asgn(beta_T_conf, x)
    #nominal_ = logistic_pol_asgn(beta_x, x)
    #     return nominal_
    # set u = I[ Y(T)\mid x > Y(-T) \mid x ]
    a_bnd, b_bnd = get_bnds(nominal_, Gamma) # propensity's bound 
    #a_bnd, b_bnd = get_bnds1(nominal_, Gamma)
    q_lo = 1 / b_bnd;
    q_hi = 1 / a_bnd
    opt_T = return_CATE_optimal_assignment(x, u, beta_cons, beta_x, beta_x_T, alpha,w)
    q_real = np.asarray([q_hi[i] if opt_T[i] == 1 else q_lo[i] for i in range(len(u))])
    #pdb.set_trace()
    #q_real = np.asarray([q_lo[i] if opt_T[i] == 1 else q_hi[i] for i in range(len(u))])
    return q_real


def real_risk_prob(prob_1, x, u):
    n = len(u)
    prob_1 = np.asarray(prob_1)
    return prob_1 * real_risk_(np.ones(n), x, u) + (1 - prob_1) * real_risk_(np.zeros(n), x, u)

def get_bnds(est_Q,LogGamma):
    n = len(est_Q)
    if type(LogGamma) is np.array and len(LogGamma) > 1:
        p_hi = est_Q
        p_lo = est_Q
        for i in range(n):
            p_hi[i] = np.multiply(np.exp(LogGamma[i]), est_Q[i] ) / (1 - est_Q[i] + np.multiply(np.exp(LogGamma[i]), est_Q[i] ))
            p_lo[i] = np.multiply(np.exp(-LogGamma[i]), est_Q[i] ) / (1 - est_Q[i] + np.multiply(np.exp(-LogGamma[i]), est_Q[i] ))
    else:
        p_hi = np.multiply(np.exp(LogGamma), est_Q ) / (np.ones(n) - est_Q + np.multiply(np.exp(LogGamma), est_Q ))
        p_lo = np.multiply(np.exp(-LogGamma), est_Q ) / (np.ones(n) - est_Q + np.multiply(np.exp(-LogGamma), est_Q ))
    assert (p_lo <= p_hi).all()
    a_bnd = 1/p_hi;
    b_bnd = 1/p_lo
    return [ a_bnd, b_bnd ]

def get_bnds1(wtilde,LogGamma):
    Gamma = np.exp(LogGamma)
    a_bnd = 1 + (wtilde - 1) / Gamma
    b_bnd = 1 + (wtilde - 1) * Gamma
    return [ a_bnd, b_bnd ]

def get_bnds2(wtilde,T,LogGamma):
    Gamma = np.exp(LogGamma)
    a_bnd = 1 + (wtilde - 1) / Gamma
    b_bnd = 1 + (wtilde - 1) * Gamma

    a_bnd[T==0] = (1 + 1 / (wtilde - 1) / Gamma)[T==0]
    b_bnd[T==0] = (1 + Gamma / (wtilde - 1) )[T==0]
    return [ a_bnd, b_bnd ]


def generate_log_data_pl(mu_x, n, beta_cons, beta_x, beta_x_T, beta_T_conf, Gamma = [0,1,2], alpha = -2, w=1.5):
    # human behavior model 
    # risk 
    # generate n datapoints from the same multivariate normal distribution
    num_person = len(Gamma)
    # random select person 
    hid = np.array([np.random.choice(num_person) for i in range(n)])
    d = len(mu_x)
    u = (np.random.rand(n) > 0.5)  # noise described in the paper 
    x = np.zeros([n, d])
    for i in range(n):
        x[i, :] = np.random.multivariate_normal(mean=mu_x * (2 * u[i] - 1), cov=np.eye(d))
        #x[i, :] = np.random.multivariate_normal(mean=mu_x, cov=np.eye(d))
    x_ = np.hstack([x, np.ones([n, 1])])
    #x_ = x

    # generate propensities
    true_Q_h = np.zeros([n,len(Gamma)])
    T_h = np.zeros([n,len(Gamma)])
    for id, gam in enumerate(Gamma):
        true_Q_h[:,id] = REAL_PROP_LOG(x_, u, beta_T_conf, beta_cons, beta_x, beta_x_T, gam, alpha,w)
        T_h[:,id] = np.array(np.random.uniform(size=n) < true_Q_h[:,id]).astype(int).flatten()   

    true_Q = true_Q_h[range(true_Q_h.shape[0]),hid]

    T = T_h[range(T_h.shape[0]),hid]
    T = T.reshape([n, 1]).astype(int)
    T_sgned = np.asarray([1 if T[i] == 1 else -1 for i in range(n)]).flatten()
    clf = LogisticRegression();
    clf.fit(x, T)
    propensities = clf.predict_proba(x)
    nominal_propensities_pos = propensities[:, 1]

    nominal_propensities_pos = logistic_pol_asgn(beta_T_conf, x_)

    true_Q_obs = np.asarray([true_Q[i] if T[i] == 1 else 1 - true_Q[i] for i in range(n)])

    q0_all = np.zeros((n,2)) 
    q0_all[:,1] = nominal_propensities_pos
    q0_all[:,0] = 1 - q0_all[:,1]
    q0 = np.asarray([nominal_propensities_pos[i] if T[i] == 1 else 1 - nominal_propensities_pos[i] for i in range(n)])

    Y_all = np.zeros((n,2))
    for i in range(n):
        Y_all[i,0] = 0 * beta_cons + np.dot(beta_x.T, x_[i, :]) + np.dot(beta_x_T.T, x_[i, :] * 0) + alpha * (u[i]) * ((2 * 0 - 1)) + w * (u[i])
        Y_all[i,1] = 1 * beta_cons + np.dot(beta_x.T, x_[i, :]) + np.dot(beta_x_T.T, x_[i, :] * 1) + alpha * (u[i]) * ((2 * 1 - 1)) + w * (u[i])
     
    # add random noise
    T = T.flatten()
    Y_all += 2*np.random.normal(size=(n,2))  
    Y = Y_all[range(len(T)),T]
    return [x, u, T, Y, true_Q_obs, q0, Y_all, q0_all, hid, T_h]  



def generate_log_data_heloc(
    df,          # pandas dataframe for HELOC (features+credit label)
    feature_cols,  # list of feature columns to use
    credit_col,    # 'good' or 'bad' indicator, 0=good, 1=bad
    n,           # total sample size to draw (train or test)
    Gamma = [0.1, 0.1, 1.0],  # log(Gamma) list for each "expert"
    seed=None    # optional: reproducibility
):
    if seed is not None:
        np.random.seed(seed)
    num_person = len(Gamma)
    # 随机分配专家
    hid = np.random.choice(num_person, size=n)
    # 随机抽样n条（可加stratify保证比例一致）
    df_ = df.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    x = df_[feature_cols].values.astype(float)
    # 标准化
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-7)
    # 信用标签
    credit = df_[credit_col].values.astype(int)  # 0=good, 1=bad
    # expert effect: 用u作为隐藏混杂因素
    u = np.random.randint(0, 2, size=n)   # binary unmeasured confounder, as in synthetic

    # ---- Step1: 用logistic policy生成 nominal propensity ----
    # 用于模拟专家决策分配
    x_ = np.concatenate([x, np.ones((n,1))], axis=1)  # 若policy有截距
    # 可加beta参数，但此处直接用LR学得参数
    nominal_policy = LogisticRegression(max_iter=500).fit(x, credit) # 用"good/bad"学policy参数
    policy_score = nominal_policy.predict_proba(x)[:,1]  # approve概率

    # ---- Step2: 生成各个expert下的treatment概率 true_Q_h, 和对应的action T_h ----
    true_Q_h = np.zeros((n, num_person))
    T_h = np.zeros((n, num_person))
    for id, gam in enumerate(Gamma):
        # Gamma为log-scale，实际exp(gam)
        prob = sigmoid( np.log(policy_score + 1e-8) * np.exp(gam) )
        true_Q_h[:, id] = prob
        T_h[:, id] = (np.random.uniform(size=n) < prob).astype(int)
    # 每个样本随机分配一个专家行为
    true_Q = true_Q_h[np.arange(n), hid]
    T = T_h[np.arange(n), hid].astype(int)

    # ---- Step3: 拟合propensity score（用实际T作为标签） ----
    prop_clf = LogisticRegression(max_iter=500)
    prop_clf.fit(x, T)
    prop_score = prop_clf.predict_proba(x)[:, 1]
    # 两种policy分配概率
    q0_all = np.zeros((n,2))
    q0_all[:,1] = prop_score
    q0_all[:,0] = 1 - prop_score
    q0 = np.where(T==1, prop_score, 1-prop_score)
    true_Q_obs = np.where(T==1, true_Q, 1-true_Q)

    # ---- Step4: Y_all生成（对每个action=0/1和good/bad credit）----
    Y_all = np.zeros((n,2))
    for i in range(n):
        # action=0: reject，无论信用好坏 N(0,1)
        Y_all[i,0] = np.random.normal(0,1)
        # action=1: approve，信用好 N(-2,1)，信用差 N(2,1)
        if credit[i] == 0:
            Y_all[i,1] = np.random.normal(-2,1)
        else:
            Y_all[i,1] = np.random.normal(2,1)
    # 得到每个样本实际action下的Y
    Y = Y_all[np.arange(n), T]

    return [x, u, T, Y, true_Q_obs, q0, Y_all, q0_all, hid, T_h]



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from geopy.exc import GeocoderUnavailable
import time 
from random import randint

