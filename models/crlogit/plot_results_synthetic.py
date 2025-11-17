#!/usr/bin/env python
# coding: utf-8

# In[4]:


# get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import numpy as np 
import pickle


# In[5]:


def plot_risks_series(gams, risks, clr, lbl, mrkr, N_REPS):
    plt.plot(gams, np.mean(risks,axis=1), color=clr,label=lbl, marker=mrkr,)
#     plt.scatter(gams, np.mean(risks,axis=1),color=clr )
    plt.fill_between(gams, np.mean(risks,axis=1)-np.std(risks,axis=1)/np.sqrt(N_REPS), 
                     np.mean(risks,axis=1)+np.std(risks,axis=1)/np.sqrt(N_REPS), 
                    color=clr, alpha=0.1)


# In[17]:



ENNS = [4000, 10000, 20000, 50000]
markers=['o', 'v', '>', '<', '^', 's', 'p', 'x', 'P']


GAMS = [0.025,0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
colors = [ 'b', 'g', 'r', 'm', 'b','purple', 'brown',  'c', ]
pltlabels = ['robust marginal', 'ipw', 'robust sharp plug-in']
risks_syn = pickle.load(open("results/risks_sharpness_test_full_control_baseline_N_"+str(N)+".pkl","rb"))
total_risks = np.zeros([len(ENNS)]+list(np.asarray(risks_syn).shape))
for N_ind,N in enumerate(ENNS):
    risks_syn = pickle.load(open("results/risks_sharpness_test_full_control_baseline_N_"+str(N)+".pkl","rb"))
    fn = 'syn_sharp_n'+str(N)+'_ctrl'
    methods = ['ogd-interval', 'ipw', 'sharp-plugin']
    risks_syn = np.asarray(risks_syn)
    N_REPS = risks_syn.shape[0]
    RISKS = risks_syn
    total_risks[N_ind,:,:,:] = risks_syn
    fig = plt.figure(figsize=(8,2.5))
    [plt.plot(GAMS, np.mean(RISKS[i,:,:],axis=0),label=pltlabels[i],color=colors[i],marker=markers[i]) for i in range(len(methods))]
    [plt.fill_between(GAMS, np.mean(RISKS[i,:,:],axis=0)-np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS)
                      , np.mean(RISKS[i,:,:],axis=0)+np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS),alpha = 0.1, color=colors[i]) for i in range(len(methods))]
    plt.legend(loc = 3)
    plt.xscale('log')
    plt.ylabel('policy regret \n against control')
    plt.xlabel(r'$\log(\Gamma)$ uncertainty parameter')
    plt.savefig('syn-global-sharp-'+fn+'.pdf', bbox_inches='tight')
    pickle.dump(fig,file(fn+'.pkl','w'))
    plt.show()


# In[10]:




[plt.plot(GAMS, np.mean(RISKS[i,:,:],axis=0),label=pltlabels[i],color=colors[i],marker=markers[i]) for i in range(len(methods))]
[plt.fill_between(GAMS, np.mean(RISKS[i,:,:],axis=0)-np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS)
                  , np.mean(RISKS[i,:,:],axis=0)+np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS),alpha = 0.1, color=colors[i]) for i in range(len(methods))]

plt.legend(loc = 3)


# # plt.title('Tree-based policies')
# opt_tree_risks_ = pickle.load(open('synth-global-tree-opttree-50-depth4.pkl','rb'))
# opt_tree_risks = opt_tree_risks_['RISKS-opt-tree']
# plot_risks_series(GAMS,RISKS_opttree[:-1,:,1], 'brown', "Opt. Tree", markers[7], opt_tree_risks.shape[1]) 
# # plot_risks_series(GAMS,RISKS_opttree[:,:,0], 'orange', "Greedy", markers[3], opt_tree_risks.shape[1]) 
plt.xscale('log')
# plt.xlabel(r'$\log(\Gamma)$ uncertainty parameter')
plt.ylabel('policy regret \n against control')
# plt.legend(fontsize=7)#,bbox_to_anchor=(1,1))
plt.xlabel(r'$\log(\Gamma)$ uncertainty parameter')

plt.savefig('syn-global-sharp-'+fn+'.pdf', bbox_inches='tight')
# plt.savefig('../../robust-policy-improvement-paper/figs/syn-global-sharp.pdf', bbox_inches='tight')

# plt.savefig('../../../robust-policy-improvement-paper/figs/syn-global-sharp.pdf', bbox_inches='tight')
pickle.dump(fig,file(fn+'.pkl','w'))
plt.show()
print plt.ylim()


# In[28]:


markers=['o', 'v', '>', '<', '^', 's', 'p', 'x', 'P']
N_REPS = 50


GAMS = [0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,2,3,4,5]
colors = [ 'b', 'g', 'r', 'm', 'b','purple', 'brown',  'c', ]
pltlabels = ['CRLogit', 'CRLogit L1 0.5', 'CRLogit L1 0.25']
fig = plt.figure(figsize=(8,3))
markers=['o', 'v', '>', '<', '^', 's', 'p', 'x', 'P']
plt.plot(GAMS, np.mean(RISK_0)*np.ones(len(GAMS)) , label = 'IPW' ,color = 'purple')
plt.fill_between(GAMS, np.mean(RISK_0)*np.ones(len(GAMS)) + np.std(RISK_0)*np.ones(len(GAMS))/np.sqrt(N_REPS),np.mean(RISK_0)*np.ones(len(GAMS)) - np.std(RISK_0)*np.ones(len(GAMS))/np.sqrt(N_REPS), color = 'purple',alpha=0.4)
plt.plot(GAMS, np.mean(cf_risks)*np.ones(len(GAMS)) , linestyle='--', label = 'GRF'  ,color = 'orange' )
plt.fill_between(GAMS, np.mean(cf_risks)*np.ones(len(GAMS)) + np.std(cf_risks)*np.ones(len(GAMS))/np.sqrt(N_REPS),np.mean(cf_risks)*np.ones(len(GAMS)) - np.std(cf_risks)*np.ones(len(GAMS))/np.sqrt(N_REPS), color = 'orange',alpha=0.4)


[plt.plot(GAMS, np.mean(RISKS[i,:,:],axis=0),label=pltlabels[i],color=colors[i],marker=markers[i]) for i in range(len(methods))]
[plt.fill_between(GAMS, np.mean(RISKS[i,:,:],axis=0)-np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS)
                  , np.mean(RISKS[i,:,:],axis=0)+np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS),alpha = 0.1, color=colors[i]) for i in range(len(methods))]

plt.legend(loc = 3)


# # plt.title('Tree-based policies')
# opt_tree_risks_ = pickle.load(open('synth-global-tree-opttree-50-depth4.pkl','rb'))
# opt_tree_risks = opt_tree_risks_['RISKS-opt-tree']
# plot_risks_series(GAMS,RISKS_opttree[:-1,:,1], 'brown', "Opt. Tree", markers[7], opt_tree_risks.shape[1]) 
# # plot_risks_series(GAMS,RISKS_opttree[:,:,0], 'orange', "Greedy", markers[3], opt_tree_risks.shape[1]) 
plt.xscale('log')
# plt.xlabel(r'$\log(\Gamma)$ uncertainty parameter')
plt.ylabel('policy regret \n against control',fontsize=15)
# plt.legend(fontsize=7)#,bbox_to_anchor=(1,1))
plt.xlabel(r'$\log(\Gamma)$ uncertainty parameter',fontsize=15)
plt.tight_layout()

plt.savefig('syn-global-sharp.pdf', bbox_inches='tight')
# plt.savefig('../../robust-policy-improvement-paper/figs/syn-global-sharp.pdf', bbox_inches='tight')

# plt.savefig('../../../robust-policy-improvement-paper/figs/syn-global-sharp.pdf', bbox_inches='tight')
pickle.dump(fig,file('SYN-sharp.pkl','w'))
plt.show()
print plt.ylim()


# In[27]:


markers=['o', 'v', '>', '<', '^', 's', 'p', 'x', 'P']
N_REPS = 50


GAMS = [0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,2,3,4,5]
colors = [ 'b', 'g', 'r', 'm', 'b','purple', 'brown',  'c', ]
pltlabels = ['CRLogit', 'CRLogit L1 0.5', 'CRLogit L1 0.25']
fig = plt.figure(figsize=(8,3))
markers=['o', 'v', '>', '<', '^', 's', 'p', 'x', 'P']
plt.plot(GAMS, np.mean(RISK_0)*np.ones(len(GAMS)) , label = 'IPW' ,color = 'purple')
plt.fill_between(GAMS, np.mean(RISK_0)*np.ones(len(GAMS)) + np.std(RISK_0)*np.ones(len(GAMS))/np.sqrt(N_REPS),np.mean(RISK_0)*np.ones(len(GAMS)) - np.std(RISK_0)*np.ones(len(GAMS))/np.sqrt(N_REPS), color = 'purple',alpha=0.4)
plt.plot(GAMS, np.mean(cf_risks)*np.ones(len(GAMS)) , linestyle='--', label = 'GRF'  ,color = 'orange' )
plt.fill_between(GAMS, np.mean(cf_risks)*np.ones(len(GAMS)) + np.std(cf_risks)*np.ones(len(GAMS))/np.sqrt(N_REPS),np.mean(cf_risks)*np.ones(len(GAMS)) - np.std(cf_risks)*np.ones(len(GAMS))/np.sqrt(N_REPS), color = 'orange',alpha=0.4)


# [plt.plot(GAMS, np.mean(RISKS[i,:,:],axis=0),label=pltlabels[i],color=colors[i],marker=markers[i]) for i in range(len(methods))]
# [plt.fill_between(GAMS, np.mean(RISKS[i,:,:],axis=0)-np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS)
#                   , np.mean(RISKS[i,:,:],axis=0)+np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS),alpha = 0.1, color=colors[i]) for i in range(len(methods))]

plt.legend(loc = 3)

plt.ylim(-0.5688200782006659, 0.4235951823786645)
# # plt.title('Tree-based policies')
# opt_tree_risks_ = pickle.load(open('synth-global-tree-opttree-50-depth4.pkl','rb'))
# opt_tree_risks = opt_tree_risks_['RISKS-opt-tree']
# plot_risks_series(GAMS,RISKS_opttree[:-1,:,1], 'brown', "Opt. Tree", markers[7], opt_tree_risks.shape[1]) 
# # plot_risks_series(GAMS,RISKS_opttree[:,:,0], 'orange', "Greedy", markers[3], opt_tree_risks.shape[1]) 
plt.xscale('log')
# plt.xlabel(r'$\log(\Gamma)$ uncertainty parameter')
plt.ylabel('policy regret \n against control',fontsize=15)
# plt.legend(fontsize=7)#,bbox_to_anchor=(1,1))
plt.xlabel(r'$\log(\Gamma)$ uncertainty parameter',fontsize=15)
plt.tight_layout()
plt.savefig('syn-global-sharp-baseline.pdf', bbox_inches='tight')
# plt.savefig('../../../robust-policy-improvement-paper/figs/syn-global-sharp.pdf', bbox_inches='tight')
pickle.dump(fig,file('SYN-sharp-baseline.pkl','w'))
plt.show()


# In[17]:


np.mean(risks_syn_trees,axis=1)[0,:]


# In[21]:


plt.plot(GAMS, np.mean(risks_syn_trees,axis=1)[0,:])


# In[22]:


risks_syn.shape
RISKS = np.zeros([4,50,20])
RISKS[0:3,:,:] = risks_syn
RISKS[3,:,:] = risks_syn_trees[1,:,:]


# In[18]:


np.vstack([ risks_syn, risks_syn_trees]) 


# In[36]:


risks_syn_trees = pickle.load(open("risks_joblib_test_opt_tree_mt_50reps.pkl","rb"))
risks_syn_trees = np.asarray(risks_syn_trees)


# In[37]:


# risks_syn_trees = pickle.load(open("risks_joblib_test_opt_tree_moretime.pkl","rb"))
risks_syn_trees = np.asarray(risks_syn_trees)
RISKS = risks_syn_trees
N_REPS = 50

methods = ['ipw', 'trees']
markers=['o', 'v', '>', '<', '^', 's', 'p', 'x', 'P']
GAMS = [0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,2,3,4,5]
colors = [ 'b', 'g', 'r', 'm', 'b','purple', 'brown',  'c', ]
pltlabels = ['CRLogit', 'CRLogit L1 0.5', 'CRLogit L1 0.25']
fig = plt.figure(figsize=(8,2.5))
markers=['o', 'v', '>', '<', '^', 's', 'p', 'x', 'P']
[plt.plot(GAMS, np.mean(RISKS[i,:,:],axis=0),label=pltlabels[i],color=colors[i],marker=markers[i]) for i in range(len(methods))]
[plt.fill_between(GAMS, np.mean(RISKS[i,:,:],axis=0)-np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS)
                  , np.mean(RISKS[i,:,:],axis=0)+np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS),alpha = 0.1, color=colors[i]) for i in range(len(methods))]

plt.xscale('log')
plt.legend(loc = 4)
pickle.dump(fig,file('SYN-sharp-tree.pkl','w'))



# IST

# ## Plot, adding trees

# In[36]:


RISKS.shape


# In[27]:





# In[44]:


markers=['o', 'v', '>', '<', '^', 's', 'p', 'x', 'P']
GAMS = [0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,2,3,4,5]
colors = [ 'b', 'g', 'r', 'm', 'purple', 'brown',  'c', ]
pltlabels = ['CRLogit', 'CRLogit L1 0.5', 'CRLogit L1 0.25', 'Opt.Trees']
fig = plt.figure(figsize=(8,2.5))
markers=['o', 'v', '>', '<', '^', 's', 'p', 'x', 'P']
i=0
N_REPS =RISKS.shape[1]
[plt.plot(GAMS, np.mean(RISKS[i,:,:],axis=0),label=pltlabels[i],color=colors[i],marker=markers[i]) for i in range(len(methods))]
[plt.fill_between(GAMS, np.mean(RISKS[i,:,:],axis=0)-np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS)
                  , np.mean(RISKS[i,:,:],axis=0)+np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS),alpha = 0.1, color=colors[i]) for i in range(len(methods))]

plt.xscale('log')
plt.legend(loc = 4)


# risks_syn_trees = pickle.load(open("risks_joblib_test_opt_tree_mt_50reps.pkl","rb"))
# risks_syn_trees = np.asarray(risks_syn_trees)
# N_REPS =50
# i=0
# plt.plot(GAMS, np.mean(risks_syn_trees[i,:,:],axis=0),label=pltlabels[i],color=colors[i+3],marker=markers[i])
# plt.fill_between(GAMS, np.mean(risks_syn_trees[i,:,:],axis=0)-np.std(risks_syn_trees[i,:,:],axis=0)/np.sqrt(N_REPS), np.mean(risks_syn_trees[i,:,:],axis=0)+np.std(risks_syn_trees[i,:,:],axis=0)/np.sqrt(N_REPS),alpha = 0.1, color=colors[i+3]) 


# i = 1
# plt.plot(GAMS, np.mean(risks_syn_trees[i,:,:],axis=0),label=pltlabels[i],color=colors[i+3],marker=markers[i])
# plt.fill_between(GAMS, np.mean(risks_syn_trees[i,:,:],axis=0)-np.std(risks_syn_trees[i,:,:],axis=0)/np.sqrt(N_REPS), np.mean(risks_syn_trees[i,:,:],axis=0)+np.std(risks_syn_trees[i,:,:],axis=0)/np.sqrt(N_REPS),alpha = 0.1, color=colors[i+3]) 


plt.xscale('log')
plt.xlabel(r'$\log(\Gamma)$ uncertainty parameter')
plt.ylabel('policy regret against control')
# plt.legend(fontsize=7)#,bbox_to_anchor=(1,1))
plt.savefig('../../robust-policy-improvement-paper/figs/syn-global-sharp.pdf', bbox_inches='tight')
# plt.savefig('../../../robust-policy-improvement-paper/figs/syn-global-sharp.pdf', bbox_inches='tight')
pickle.dump(fig,file('SYN-sharp.pkl','w'))
plt.show()


# In[ ]:


risks_IST.shape


# In[ ]:


risks_IST = pickle.load(open("results/risks_test_IST.pkl","rb"))

methods = ['IPW', 'ogd-interval']
risks_IST = np.asarray(risks_IST)
RISKS = risks_IST
N_REPS = risks_IST.shape[1]


# In[ ]:


GAMS = [0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,1.1,1.3,1.5,2]
markers=['o', 'v', '>', '<', '^', 's', 'p', 'x', 'P']
colors = [ 'b', 'g', 'r', 'm', 'b','purple', 'brown',  'c', ]
# pltlabels = ['CRLogit', 'CRLogit L1 0.5', 'CRLogit L1 0.25']
fig = plt.figure(figsize=(8,2.5))
markers=['o', 'v', '>', '<', '^', 's', 'p', 'x', 'P']
[plt.plot(GAMS, np.mean(RISKS[i,:,:],axis=0),label=pltlabels[i],color=colors[i],marker=markers[i]) for i in range(len(methods))]
[plt.fill_between(GAMS, np.mean(RISKS[i,:,:],axis=0)-np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS)
                  , np.mean(RISKS[i,:,:],axis=0)+np.std(RISKS[i,:,:],axis=0)/np.sqrt(N_REPS),alpha = 0.1, color=colors[i]) for i in range(len(methods))]

plt.xscale('log')
plt.legend(loc = 4)
pickle.dump(fig,file('SYN-IST.pkl','w'))


# In[ ]:




