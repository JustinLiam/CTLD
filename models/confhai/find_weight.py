import numpy as np


''' Given  Y (unsorted), lower bound 'a_', upper bound 'b_' on weights, and possible index list sub_ind,
return Lambda (min problem value), weights, sum(weights)
'''
def find_opt_weights_short(Y, a_, b_, sub_ind=[]):
    if len(sub_ind)>0:
        print (sub_ind)
        a_=a_[sub_ind]; b_=b_[sub_ind]; Y = Y[sub_ind]
    sort_inds = np.lexsort((a_,Y)); a_=a_[sort_inds]; Y = Y[sort_inds]; b_=b_[sort_inds]
    n_plus = len(Y); weights = np.zeros(n_plus); prev_val = -np.inf; k=1; val = sum(np.multiply(b_, Y)) /sum(b_)
    while (val > prev_val) and (k < n_plus+1):
        denom = 0; num = 0; prev_val = val; num = 1.0*sum(np.multiply(a_[0:k], Y[0:k])) + sum(np.multiply(b_[k:], Y[k:]))
        denom = sum(a_[0:k])+sum(b_[k:]); val = num / denom; k+=1;
    lda_opt = prev_val; k_star = k-1
    sort_inds_a = sort_inds[0:k_star]; sort_inds_b = sort_inds[k_star:]
    weights[sort_inds_a] = a_[0:k_star]; weights[sort_inds_b] = b_[k_star:]
    return [lda_opt, weights, sum(weights)]


def rnd_k_val(k_,a_,b_,Y):
    k= int(np.floor(k_)) # floor or round?
    return (1.0*np.sum(np.multiply(a_[0:k], Y[0:k])) + np.sum(np.multiply(b_[k:], Y[k:])))/(np.sum(a_[0:k])+np.sum(b_[k:]))


''' Given  Y (unsorted), lower bound 'a_', upper bound 'b_' on weights, and possible index list sub_ind,
use ternary search to find the optimal value.
Possibly off by one error but it shouldn't matter.
return Lambda (min problem value), weights, sum(weights)
'''
def find_opt_weights_shorter(Y, a_, b_, sub_ind=[]):
    if len(sub_ind)>0:
        a_=a_[sub_ind]; b_=b_[sub_ind]; Y = Y[sub_ind]
    # sort_inds = np.argsort(Y);
    sort_inds = np.lexsort((b_-a_,Y));
    a_=a_[sort_inds]; Y = Y[sort_inds]; b_=b_[sort_inds]
    n_plus = len(Y); weights = np.zeros(n_plus); prev_val = -np.inf; k=1; val = np.sum(np.multiply(b_, Y)) /sum(b_)
    left = 0; right = n_plus-1;keepgoing=True
    while keepgoing:
        #left and right are the current bounds; the maximum is between them
#         print (left,right)
#         print (rnd_k_val(leftThird,a_,b_,Y), rnd_k_val(rightThird,a_,b_,Y))
        if abs(right - left) < 2.1: # separation in index space
            k = np.floor((left + right)/2)
            keepgoing=False
        leftThird = left + (right - left)/3
        rightThird = right - (right - left)/3
        if rnd_k_val(leftThird,a_,b_,Y) < rnd_k_val(rightThird,a_,b_,Y):
            left = leftThird
        else:
            right = rightThird
    k_star=int(k); k=int(k)
    lda_opt = (1.0*np.sum(np.multiply(a_[0:k], Y[0:k])) + np.sum(np.multiply(b_[k:], Y[k:])))/(np.sum(a_[0:k])+np.sum(b_[k:]))
    sort_inds_a = sort_inds[0:k_star]; sort_inds_b = sort_inds[k_star:]
    weights[sort_inds_a] = a_[0:k_star]; weights[sort_inds_b] = b_[k_star:]
    return [lda_opt, weights, np.sum(weights)]