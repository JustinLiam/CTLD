import numpy as np 
import torch 
import torch.nn.functional as F
from .find_weight import *
from .models import *
from .data import *
import random 
import copy 

def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def train_ips(input_dim, output_dim, loader, num_epochs = 100, lr = 1e-2, hidd = 2):
    model = Net(input_dim, output_dim, hidden = hidd)
    model.to(device)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    epsilon = 1e-8
    for epoch in range(num_epochs):
        l = 0
        for batch, (x, q, t, y) in enumerate(loader):
            x, q, t, y = x.to(device), q.to(device), t.to(device), y.to(device)
            out = model(x.float())
            out = F.softmax(out, 1)
            out = out[range(out.size(0)),t.long()]
            logp = q[range(out.size(0)),t.long()]
            loss = y * out / (logp + epsilon)
            loss = loss.mean()
            # backward
            optimizer.zero_grad()
            loss.backward()
            l += loss.item()
            optimizer.step()
            scheduler.step()
        if (epoch +1) % 100 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'.format(epoch, num_epochs, l / (batch + 1)))
    model.cpu()
    return model


def train_confips(input_dim, output_dim, loader, num_epochs = 100, lr = 1e-2, gamma = 1, pre_model = None, hidd = 2):
    # multiple stat 
    multiple = 3
    best_model = None 
    best_loss = 100000
    for _ in range(multiple):
        #if pre_model:
        #    model = pre_model
        #else:
        model = Net(input_dim, output_dim, hidden = hidd)
        model.to(device)
        if torch.cuda.is_available():
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        #optimizer = torch.optim.SGD(model.parameters(), lr = lr)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        for epoch in range(num_epochs):
            l = 0
            for batch, (x, q, t, y) in enumerate(loader):
                x, q, t, y = x.to(device), q.to(device), t.to(device), y.to(device)
                out = model(x.float())
                out = F.softmax(out, 1)
                out = out[range(out.size(0)),t.long()]
                logp = q[range(out.size(0)),t.long()]
                ctrp = (t == 0) * 1. # control policy that assigns everything 0 
                #ctrp = 0 

                #pdb.set_trace()
                a_, b_ = get_bnds1(1/logp.cpu().numpy(), gamma)
                #a_, b_ = get_bnds2(1/q[:,1].cpu().numpy(), t.cpu().numpy(), gamma)
                #a_, b_ = get_bnds(q[:,1].cpu().numpy(), gamma)
                [lda_opt1, weights1, sw1] = find_opt_weights_shorter(((y*(out-ctrp))[t==1]).detach().cpu().numpy(), a_[t.cpu().numpy()==1], b_[t.cpu().numpy()==1])
                [lda_opt0, weights0, sw0] = find_opt_weights_shorter(((y*(out-ctrp))[t==0]).detach().cpu().numpy(), a_[t.cpu().numpy()==0], b_[t.cpu().numpy()==0])
                
                #pdb.set_trace()
                weight = torch.zeros_like(out)
                weight[t==1] = torch.Tensor(weights1).to(device) *1.0 / sw1
                weight[t==0] = torch.Tensor(weights0).to(device) *1.0 / sw0
                #pdb.set_trace()
                #weight = weight / weight.sum()

                #loss = y * (out) * weight 
                loss = y * (out-ctrp) * weight 

                #loss = loss.mean()
                loss = loss.sum()
                # backward
                optimizer.zero_grad()
                loss.backward()
                l += loss.item()
                optimizer.step()
                #scheduler.step()
            if (epoch +1) % 100 == 0:
                print('Epoch[{}/{}], loss: {:.6f}'.format(epoch, num_epochs, l / (batch + 1)))
        if l < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = l
        if best_loss > 0:
            con_ind = 1
        else:
            con_ind = 0            
    model = best_model
    model.cpu()
    return model, con_ind 

def train_hai(model, input_dim, output_dim, loader, num_epochs = 100, lr = 1e-2, C = 0, hidd = 2):
    model = Net(input_dim, output_dim, hidden = hidd)
    model.to(device)
    router = Net(input_dim, 1, hidden = hidd)
    router.to(device)
    router.fc.bias.data[0] = -5
    if torch.cuda.is_available():
        model.cuda()
        router.cuda()
    optimizer = torch.optim.Adam(list(model.parameters())+list(router.parameters()), lr = lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    for epoch in range(num_epochs):
        l = 0
        for batch, (x, q, t, y) in enumerate(loader):
            x, q, t, y = x.to(device), q.to(device), t.to(device), y.to(device)
            out = model(x.float())
            out = F.softmax(out, 1)

            human = torch.sigmoid(router(x.float())).reshape(-1)
            #human = F.softmax(human, 1)

            out = out[range(out.size(0)),t.long()]
            logp = q[range(out.size(0)),t.long()]
            loss = y * out / logp * (1-human) + human * (y+C)
            loss = loss.mean()
            # backward
            optimizer.zero_grad()
            loss.backward()
            l += loss.item()
            optimizer.step()
            #scheduler.step()
        if (epoch +1) % 100 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'.format(epoch, num_epochs, l / (batch + 1)))
    model.cpu()
    router.cpu()
    return model, router

def train_confhai(model, router, input_dim, output_dim, loader, num_epochs = 100, lr = 1e-2, gamma = 3, C = 0, hidd = 2):
    multiple = 3
    best_model = None 
    best_router = None
    best_loss = 100000
    for seed in range(multiple):
        model = Net(input_dim, output_dim, hidden = hidd)
        model.to(device)
        router = Net(input_dim, 1, hidden = hidd)
        router.to(device)
        router.fc.bias.data[0] = 0 # original code has this line, and set bias to -10
        if torch.cuda.is_available():
            model.cuda()
            router.cuda()
        optimizer = torch.optim.Adam(list(model.parameters())+list(router.parameters()), lr = lr)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        for epoch in range(num_epochs):
            l = 0
            for batch, (x, q, t, y) in enumerate(loader):
                x, q, t, y = x.to(device), q.to(device), t.to(device), y.to(device)
                out = model(x.float())
                out = F.softmax(out, 1)
                ctrp = (t == 0) * 1. # control policy that assigns everything 0 

                human = torch.sigmoid(router(x.float())).reshape(-1)
                #human = F.softmax(human, 1)

                out = out[range(out.size(0)),t.long()]
                logp = q[range(out.size(0)),t.long()]

                #a_, b_ = get_bnds(logp.numpy(), gamma)
                #a_, b_ = get_bnds(q[:,1].cpu().numpy(), gamma)
                a_, b_ = get_bnds1(1/logp.cpu().numpy(), gamma)
                [lda_opt1, weights1, sw1] = find_opt_weights_shorter(((y*out*(1-human)-ctrp*y)[t==1]).detach().cpu().numpy(), a_[t.cpu().numpy()==1], b_[t.cpu().numpy()==1])
                [lda_opt0, weights0, sw0] = find_opt_weights_shorter(((y*out*(1-human)-ctrp*y)[t==0]).detach().cpu().numpy(), a_[t.cpu().numpy()==0], b_[t.cpu().numpy()==0])
                
                weight = torch.zeros_like(out)
                weight[t==1] = torch.Tensor(weights1).to(device) *1.0 / sw1
                weight[t==0] = torch.Tensor(weights0).to(device) *1.0 / sw0
                #weight = weight / weight.sum()

                loss1 = y * (out) * weight * (1-human) - y * ctrp * weight
                loss1 = loss1.sum()

                loss2 = human * (y+C) 

                loss = loss1 + loss2.mean()
                # backward
                optimizer.zero_grad()
                loss.backward()
                l += loss.item()
                optimizer.step()
                #scheduler.step()
            if (epoch +1) % 200 == 0:
                print('Epoch[{}/{}], loss: {:.6f}'.format(epoch, num_epochs, l / (batch + 1)))
        if l < best_loss:
            best_model = copy.deepcopy(model)
            best_router = copy.deepcopy(router)
            best_loss = l
        if best_loss > 0:
            con_ind = 1
        else:
            con_ind = 0
    model = best_model
    router = best_router
    model.cpu()
    router.cpu()
    return model, router, con_ind



def train_confhai_person(model, router, input_dim, output_dim, loader, num_epochs = 100, lr = 1e-2, gamma = 3, nump = 5, C = 0, hidd = 2):
    multiple = 3
    best_model = None 
    best_loss = 100000
    for seed in range(multiple):
        #model = Net(input_dim, output_dim, hidden = hidd)
        router = Net(input_dim, nump+1, hidden = hidd) # first dimension is d(a|x)
        router.to(device)
        #router.fc.bias.data[0] = -5
        #router.fc.bias.data[-1] = 1
        if torch.cuda.is_available():
            model.cuda()
            router.cuda()
        optimizer = torch.optim.Adam(list(model.parameters())+list(router.parameters()), lr = lr)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        #optimizer = torch.optim.Adam(list(router.parameters()), lr = lr)
        for epoch in range(num_epochs):
            l = 0
            for batch, (x, q, t, y, hid) in enumerate(loader):
                x, q, t, y, hid = x.to(device), q.to(device), t.to(device), y.to(device), hid.to(device)
                out = model(x.float())
                out = F.softmax(out, 1)
                ctrp = (t == 0) * 1. # control policy that assigns everything 0 

                #human = torch.sigmoid(router(x.float())).reshape(-1)
                human = F.softmax(router(x.float()), 1)

                out = out[range(out.size(0)),t.long()]
                logp = q[range(out.size(0)),t.long()]

                a_, b_ = get_bnds1(1/logp.cpu().numpy(), gamma)
                #a_, b_ = get_bnds(logp.numpy(), gamma)
                #a_, b_ = get_bnds(q[:,1].cpu().numpy(), gamma)
                #a_ = a_ / human[range(human.size(0)), hid]
                #b_ = b_ / human[range(human.size(0)), hid]
                #a_ = a_ * nump
                #b_ = b_ * nump
                [lda_opt1, weights1, sw1] = find_opt_weights_shorter(((y*out*human[:,0]* nump-ctrp*y* nump)[t==1]).detach().cpu().numpy(), a_[t.cpu().numpy()==1], b_[t.cpu().numpy()==1])
                [lda_opt0, weights0, sw0] = find_opt_weights_shorter(((y*out*human[:,0]* nump-ctrp*y* nump)[t==0]).detach().cpu().numpy(), a_[t.cpu().numpy()==0], b_[t.cpu().numpy()==0])
                #[lda_opt1, weights1, sw1] = find_opt_weights_shorter(((y*out*human[:,0]-ctrp*y)[t==1]).detach().cpu().numpy(), a_[t.cpu().numpy()==1], b_[t.cpu().numpy()==1])
                #[lda_opt0, weights0, sw0] = find_opt_weights_shorter(((y*out*human[:,0]-ctrp*y)[t==0]).detach().cpu().numpy(), a_[t.cpu().numpy()==0], b_[t.cpu().numpy()==0])
                
                weight = torch.zeros_like(out)
                weight[t==1] = torch.Tensor(weights1).to(device) *1.0 / sw1
                weight[t==0] = torch.Tensor(weights0).to(device) *1.0 / sw0
                #weight = weight / weight.sum()

                #loss1 = y * (out) * weight * human[:,0] - y * ctrp * weight
                # human[:,0] prob of selecing algorithm 
                #loss1 = y * human[:,0] * out * weight - y * ctrp * weight * nump
                loss1 = y * human[:,0] * out * weight - y * ctrp * weight 
                loss1 = loss1.sum()

                # human[:,1:] prob of selecing diff humans 
                #import pdb 
                #pdb.set_trace()
                loss2 = (human[range(human.size(0)),hid.long()+1]).reshape(-1) * (y+C) * nump 
                #loss2 = (1-human[:,0]) * (y+C) 

                loss = loss1 + loss2.mean()
                loss = loss 
                # backward
                optimizer.zero_grad()
                loss.backward()
                l += loss.item()
                optimizer.step()
                #scheduler.step()
            if (epoch +1) % 200 == 0:
                print('Epoch[{}/{}], loss: {:.6f}'.format(epoch, num_epochs, l / (batch + 1)))
                #print(human.mean(0))
        if l < best_loss:
            best_model = copy.deepcopy(model)
            best_router = copy.deepcopy(router)
            best_loss = l
        if best_loss > 0:
            con_ind = 1
        else:
            con_ind = 0            
    model = best_model
    router = best_router
    model.cpu()
    router.cpu()
    return model, router, con_ind


@torch.no_grad()
def test_reward(model, x, Y, control = False, con_ind = 0):
    if con_ind == 1:
        return 0 
    model.eval()
    pred = np.argmax((model(torch.Tensor(x)).detach()).numpy(),1)
    reward = Y[range(Y.shape[0]),pred]
    model.train()
    if control:
        return reward.mean() - Y[:,0].mean()
    return reward.mean()

@torch.no_grad()
def test_hai(model, router, x, Y, t, C = 0, control = False, con_ind = 0):
    if con_ind == 1:
        return 0 
    model.eval()
    router.eval()
    pred = np.argmax((model(torch.Tensor(x)).detach()).numpy(),1)
    #human = np.argmax((router(torch.Tensor(x)).detach()).numpy(),1)
    human = (torch.sigmoid(router(torch.Tensor(x))).detach()).numpy() > 0.5
    human = human.reshape(-1)
    print(np.bincount(human))
    pred[human == 1] = t[human == 1]
    reward = Y[range(Y.shape[0]),pred]
    model.train()
    router.train()
    if control:
        return reward.mean() + (human * C).mean() - Y[:,0].mean()
    return reward.mean() + (human * C).mean()

@torch.no_grad()
def test_hai_person(model, router, x, Y, t_h, t, hid, nump=3, C = 0, control = False, con_ind = 0):
    if con_ind == 1:
        return 0 
    model.eval()
    router.eval()
    pred = np.argmax((model(torch.Tensor(x)).detach()).numpy(),1)
    human = np.argmax((router(torch.Tensor(x)).detach()).numpy(),1)
    human = human.reshape(-1)
    print(np.bincount(human))
    for i in range(nump):
        pred[human == i+1] = t_h[range(t_h.shape[0]),i][human == i+1]
    reward = Y[range(Y.shape[0]),pred]
    model.train()
    router.train()
    if control:
        return reward.mean() + ((human>0)* C).mean() - Y[:,0].mean()
    return reward.mean() + ((human>0)* C).mean()

