import numpy as np
import gurobipy as gp
from unconfoundedness_fns import *
from subgrad import *
# from greedy_partitioning_serv import *
import pickle
from datetime import datetime



def catch_predict_index_error(rf,x,quantile):
    try:
        return np.asscalar(rf.predict(x.reshape(1,-1),quantile))
    except IndexError:
        return np.asscalar(rf.predict(x.reshape(1,-1)))


def sharp_worst_case_plugin(th, POL_PROB_1, x, y, a_bnd, b_bnd, Y_quantiles):
    # in space of weights rather than inverse fq 
    n = x.shape[0];
    assert all(len(arr) == n for arr in [x,y, Y_quantiles])
    # If last column is all ones, don't augment
    if (x[:,-1] == np.ones(n)).all():
        x_aug = x
    else: # otherwise augment data
        x_aug = np.hstack([x, np.ones([n,1])]);
    worst_case_weights = np.asarray([ b_bnd[i] if y[i] > Y_quantiles[i] else a_bnd[i] for i in range(n)  ]).flatten()
    pi_1 = POL_PROB_1(th, x_aug).flatten();
    pi_t = np.asarray( [pi_1[i] if t[i] == 1 else 1- pi_1[i] for i in range(n)] )
    loss = np.mean( y*pi_t*worst_case_weights  )
    return loss


def sharp_pol_eval_plugin(th, N_RNDS, WGHTS_, GRAD_, POL_GRAD, POL_PROB_1, BASELINE_POL, x, t01, fq, y,
 a_, b_, gamma,Y_tau_quantiles,Y_1minustau_quantiles): 
    '''wrapper to plug in solution 
    '''
    n = x.shape[0]; 
    t = get_sgn_0_1(t01)
    t_levels = np.unique(t01)
    assert all(len(arr) == n for arr in [x,t01,y,a_,b_])
    # Check if x data contains an intercept: only retain for backwards compatibility:
    # If last column is all ones, don't augment
    if (x[:,-1] == np.ones(n)).all():
        x_aug = x
    else: # otherwise augment data
        x_aug = np.hstack([x, np.ones([n,1])]);
    pi_1 = POL_PROB_1(th, x_aug).flatten();
    p_1 = BASELINE_POL(pi_1);

    Y_quantiles = np.asarray([ Y_tau_quantiles[i] if np.sign((pi_1[i] - p_1[i])*t[i])>0 else Y_1minustau_quantiles[i] for i in range(n)  ]).flatten()
    worst_case_weights = np.asarray([ b_bnd[i] if y[i]*np.sign((pi_1[i] - p_1[i])*t[i]) > np.sign((pi_1[i] - p_1[i])*t[i])*Y_quantiles[i] else a_bnd[i] for i in range(n)  ]).flatten()
    loss = np.mean( y*pi_t*worst_case_weights )

    return loss

def sharp_worst_case_plugin(th, POL_PROB_1, x, t, fq, y, a_bnd, b_bnd, Y_quantiles,logging=False,step_schedule=0.5):
    # in space of weights rather than inverse fq 
    n = x.shape[0];
    assert all(len(arr) == n for arr in [x,t,fq,y])
    # If last column is all ones, don't augment
    if (x[:,-1] == np.ones(n)).all():
        x_aug = x
    else: # otherwise augment data
        x_aug = np.hstack([x, np.ones([n,1])]);
    worst_case_weights = np.asarray([ b_bnd[i] if y[i] > Y_quantiles[i] else a_bnd[i] for i in range(n)  ]).flatten()
    pi_1 = POL_PROB_1(th, x_aug).flatten();
    pi_t = np.asarray( [pi_1[i] if t[i] == 1 else 1- pi_1[i] for i in range(n)] )
    loss = np.mean( y*pi_t*worst_case_weights  )
    return loss


def sharp_worst_case_quantile_balancing(gamma, Y, pi, a_, b_, fq, Y_quantiles,  qr, quiet=True):
    # solve within one treatment subproblem
    wm = 1/fq; wm_sum=wm.sum(); n = len(Y)
    # assume estimated propensities are probs of observing T_i
    y = Y*pi; weights = np.zeros(n);
    m = gp.Model()
    if quiet: m.setParam("OutputFlag", 0)
    w = [m.addVar(obj = -yy, lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    m.update()
    # m.addConstr(gp.quicksum(w)==1)
    m.addConstr(gp.LinExpr(pi*Y_quantiles*1.0/n,w) == gp.quicksum(pi*Y_quantiles)*1.0/n)
    for i in range(len(y)):
        m.addConstr(w[i] <= b_[i])
        m.addConstr(w[i] >= a_[i])
    m.optimize()
    wghts = np.asarray([ ww.X for ww in w ]) # would like to have failsafe for not being able to optimize
    return [-m.ObjVal,wghts]



def sharp_worst_case_quantile_balancing_normalized(gamma, Y, pi, a_, b_, fq, Y_quantiles,   qr, quiet=True):
    # solve within one treatment subproblem
    wm = 1/fq; wm_sum=wm.sum(); n = len(Y)
    wm = wm/wm_sum # normalize propensities
    # assume estimated propensities are probs of observing T_i
    y = Y*pi; weights = np.zeros(n);
    m = gp.Model()
    if quiet: m.setParam("OutputFlag", 0)
    t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    w = [m.addVar(obj = -yy, lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    d = [m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    m.update()
    m.addConstr(gp.quicksum(w)==1)
    m.addConstr(gp.quicksum(d)<=gamma*t)
    m.addConstr(gp.LinExpr(pi*Y_quantiles,w) == gp.quicksum(pi*Y_quantiles)*1.0/n)
    for i in range(len(y)):
        m.addConstr(w[i] <= b_[i] * t/wm_sum)
        m.addConstr(w[i] >= a_[i] * t/wm_sum)
        m.addConstr(d[i] >=   w[i] - t*wm[i])
        m.addConstr(d[i] >= - w[i] + t*wm[i])
    m.optimize()
    wghts = np.asarray([ ww.X for ww in w ]) # would like to have failsafe for not being able to optimize
    return [-m.ObjVal,wghts,t.X/wm_sum]