import copy

import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from datasets import IHDP
from torch.utils.data import DataLoader

eps_cst = 1e-8


def lce_surrogate_loss(outputs, costs):
    '''
        L_{CE} loss implementation for cost-sensitive classification from Mozannar and Sontag (2020)
        learner should minimize this loss, tries to pick class with lowest cost

        outputs: model outputs (g_i), tensor of shape [batch_size, num_classes]
        costs: costs of each class per example (c_i), tensor of shape [batch_size, num_classes]

        return:
        surrogate loss, scalar

        '''
    outputs = F.softmax(outputs, dim=1)
    # maximal cost is per example max cost
    maximal_cost, _ = torch.max(costs, dim=1)
    # make costs = max_costs - costs
    costs = maximal_cost.view(-1, 1) - costs
    # vector of loss per example
    loss_pointwise = -torch.sum(costs * torch.log2(outputs + eps_cst), dim=1)
    # average loss over batch
    loss = torch.sum(loss_pointwise) / outputs.size()[0]
    return loss


def min_cost_loss(preds, costs):

    loss = 0
    for i in range(len(preds)):
        pred = preds[i]
        cost = costs[i][pred]
        loss += cost
    loss = torch.tensor(loss)
    loss.requires_grad = True
    return loss







class LCEModel(pl.LightningModule):
    def __init__(self, pmodel, training_costs, validation_costs=None, lr=5e-4, weight_decay=0.0001,optimizer_name='Adam'):
        super().__init__()
        self.model = copy.deepcopy(pmodel)
        self.training_costs = torch.Tensor(training_costs)
        if validation_costs is not None:
            self.validation_costs = torch.Tensor(validation_costs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, t, idx, y = batch
        logits = self(x)
        _, preds = torch.max(logits, 1)
        if torch.cuda.is_available():
            idx = idx.cuda()
            logits = logits.cuda()
            preds = preds.cuda()
            self.training_costs = self.training_costs.cuda()
        loss = lce_surrogate_loss(outputs=logits, costs=self.training_costs[idx])
        # loss = min_cost_loss(preds=preds, costs=self.training_costs[idx])

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t, idx, y = batch
        logits = self(x)
        _, preds = torch.max(logits, 1)
        if torch.cuda.is_available():
            idx = idx.cuda()
            logits = logits.cuda()
            preds = preds.cuda()
            self.validation_costs = self.validation_costs.cuda()
        loss = lce_surrogate_loss(outputs=logits, costs=self.validation_costs[idx])
        # loss = min_cost_loss(preds=preds, costs=self.training_costs[idx])

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss



    def predict_step(self, batch, batch_idx):
        x, t, idx, y = batch
        test_logits = self.model(x)
        _, test_pred = torch.max(test_logits, 1)
        # test_logits_no_def = test_logits[:, 0:2]
        # _, test_preds_no_def = torch.max(test_logits_no_def, 1)
        return test_pred

    def _update_preds(self, all_preds, batch, batch_preds):
        x, t, idx, y = batch
        for i in range(len(batch_preds)):
            all_preds[idx[i]] = batch_preds[i]
        return all_preds

    # def predict(self, dl):
    #     # torch.set_grad_enabled(False)
    #     # self.model.eval()
    #     # all_preds = [None] * (len(dl) * dl.batch_size)
    #     all_preds = [None] * len(dl.dataset)
    #     with torch.no_grad():
    #         for batch_idx, batch in enumerate(dl):
    #             batch_preds = self.predict_step(batch=batch, batch_idx=batch_idx)
    #             all_preds = self._update_preds(all_preds, batch, batch_preds)
    #     return all_preds

    def predict(self, dl):
        self.model.eval()
        all_preds = np.zeros(len(dl.dataset), dtype=int)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dl):
                batch_preds = self.predict_step(batch=batch, batch_idx=batch_idx)
                if isinstance(batch_preds, torch.Tensor):
                    batch_preds = batch_preds.cpu().numpy()
                _, _, idx, _ = batch
                if torch.is_tensor(idx):
                    idx = idx.cpu().numpy()
                all_preds[idx] = batch_preds
        return all_preds

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay) # try 0.01 afterward

        optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(),
                                                              lr=self.lr,
                                                              weight_decay=self.weight_decay)
        return optimizer
    # return torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=0.01)

    #0.005


class LCEModel_logits(pl.LightningModule):
    def __init__(self, pmodel, training_costs, lr, weight_decay):
        super().__init__()
        self.model = copy.deepcopy(pmodel)
        self.training_costs = torch.Tensor(training_costs)
        self.lr = lr
        self.weight_decay = weight_decay


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, t, idx, y = batch
        logits = self(x)
        if torch.cuda.is_available():
            idx = idx.cuda()
            logits = logits.cuda()
            self.training_costs = self.training_costs.cuda()
        loss = lce_surrogate_loss(outputs=logits, costs=self.training_costs[idx])
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    # def validation_step(self, batch, batch_idx):
    #     x, t, idx, y = batch
    #     logits = self(x)
    #     loss = lce_surrogate_loss(outputs=logits, costs=self.training_costs[idx]) # val costs are needed:(
    #     # logs metrics for each training_step,
    #     # and the average across the epoch, to the progress bar and logger
    #     self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     return loss

    def predict_step(self, batch, batch_idx):
        x, t, idx, y = batch
        test_logits = self.model(x)
        # _, test_pred = torch.max(test_logits, 1)
        # test_logits_no_def = test_logits[:, 0:2]
        # _, test_preds_no_def = torch.max(test_logits_no_def, 1)
        return test_logits

    def _update_preds(self, all_preds, batch, batch_preds):
        x, t, idx, y = batch
        for i in range(len(batch_preds)):
            all_preds[idx[i]] = batch_preds[i]
        return all_preds

    def predict(self, dl):
        # torch.set_grad_enabled(False)
        # self.model.eval()
        all_preds = [None] * (len(dl) * dl.batch_size)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dl):
                batch_preds = self.predict_step(batch=batch, batch_idx=batch_idx)
                all_preds = self._update_preds(all_preds, batch, batch_preds)
        return all_preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    # return torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=0.01)



class LCEModel_optuna(pl.LightningModule):
    def __init__(self, pmodel, training_costs, trial):
        super().__init__()
        self.model = copy.deepcopy(pmodel)
        self.training_costs = torch.Tensor(training_costs)
        self.lr = trial.suggest_loguniform('learning_rate', 1e-5,0.1)
        self.optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW'])
        self.weight_decay = trial.suggest_loguniform('weight_decay', 0, 0.1)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, t, idx, y = batch
        logits = self(x)
        _, preds = torch.max(logits, 1)
        if torch.cuda.is_available():
            idx = idx.cuda()
            logits = logits.cuda()
            preds = preds.cuda()
            self.training_costs = self.training_costs.cuda()
        loss = lce_surrogate_loss(outputs=logits, costs=self.training_costs[idx])
        # loss = min_cost_loss(preds=preds, costs=self.training_costs[idx])

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    # def validation_step(self, batch, batch_idx):
    #     x, t, idx, y = batch
    #     logits = self(x)
    #     loss = lce_surrogate_loss(outputs=logits, costs=self.training_costs[idx]) # val costs are needed:(
    #     # logs metrics for each training_step,
    #     # and the average across the epoch, to the progress bar and logger
    #     self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     return loss

    def predict_step(self, batch, batch_idx):
        x, t, idx, y = batch
        test_logits = self.model(x)
        _, test_pred = torch.max(test_logits, 1)
        # test_logits_no_def = test_logits[:, 0:2]
        # _, test_preds_no_def = torch.max(test_logits_no_def, 1)
        return test_pred

    def _update_preds(self, all_preds, batch, batch_preds):
        x, t, idx, y = batch
        for i in range(len(batch_preds)):
            all_preds[idx[i]] = batch_preds[i]
        return all_preds

    def predict(self, dl):
        # torch.set_grad_enabled(False)
        # self.model.eval()
        all_preds = [None] * (len(dl) * dl.batch_size)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dl):
                batch_preds = self.predict_step(batch=batch, batch_idx=batch_idx)
                all_preds = self._update_preds(all_preds, batch, batch_preds)
        return all_preds

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(),
                                                              lr=self.lr,
                                                              weight_decay=self.weight_decay)
        return optimizer
        # return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)