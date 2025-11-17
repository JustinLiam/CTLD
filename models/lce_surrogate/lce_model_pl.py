import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

eps_cst = 1e-8

def asymmetric_softmax_transformation(logits, defer_penalty=1.0):

    classifier_logits = logits[:, :-1]
    defer_logit = logits[:, -1].view(-1, 1)

    classifier_probs = F.softmax(classifier_logits, dim=-1)

    max_classifier_prob, _ = torch.max(classifier_probs, dim=-1, keepdim=True)

    transformed_defer_score = defer_logit - defer_penalty * max_classifier_prob.detach()

    transformed_logits = torch.cat([classifier_logits, transformed_defer_score], dim=1)

    return transformed_logits


def asymmetric_lce_loss(outputs, costs, defer_penalty=1.0):

    transformed_logits = asymmetric_softmax_transformation(outputs, defer_penalty=defer_penalty)

    log_probs = F.log_softmax(transformed_logits, dim=1)
    loss_pointwise = torch.sum(costs * log_probs, dim=1)
    loss = torch.mean(loss_pointwise)
    return loss

def lce_surrogate_loss(outputs, costs):

    log_probs = F.log_softmax(outputs, dim=1)

    loss_pointwise = torch.sum(costs * log_probs, dim=1)

    loss = torch.mean(loss_pointwise)
    return loss


class LCEModel(L.LightningModule):

    def __init__(self, pmodel, lr=5e-4, weight_decay=1e-5, optimizer_name='AdamW'):
        super().__init__()

        self.save_hyperparameters(ignore=['pmodel'])
        # self.pmodel = copy.deepcopy(pmodel)
        self.pmodel = pmodel

    def forward(self, x):
        return self.pmodel(x)


    def training_step(self, batch, batch_idx):
        x_features, costs = batch
        x_features = x_features.to(self.device)
        costs = costs.to(self.device)
        logits = self(x_features)

        # print(f"[DEBUG] training_step: "
        #       f"model_device={next(self.pmodel.parameters()).device}, "
        #       f"data_device={batch[0].device}, "
        #       f"lightning_module_device={self.device}")
        # print(f"step: model={next(self.pmodel.parameters()).device}, x={x_features.device}, costs={costs.device}")

        transformed_logits = asymmetric_softmax_transformation(logits, defer_penalty=1.0)
        log_probs = F.log_softmax(transformed_logits, dim=1)
        lce_loss = torch.mean(torch.sum(costs * log_probs, dim=1))

        # --- 新增：熵正则化项 ---
        # 在log_probs之后计算probs，避免重复计算softmax
        probs = torch.exp(log_probs)
        # 计算熵 H(p) = - sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=1)

        # entropy_lambda 是一个新的超参数，控制正则化的强度
        # 我们希望最大化熵，所以在loss中减去它
        entropy_lambda = 0.1  # 可以从0.01, 0.1等开始尝试
        total_loss = lce_loss - entropy_lambda * torch.mean(entropy)

        # self.log("train_loss", total_loss, on_epoch=True, prog_bar=True)
        # self.log("lce_loss", lce_loss, on_epoch=True)  # 单独记录，便于观察
        # self.log("entropy", torch.mean(entropy), on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # 验证步骤与训练步骤的逻辑一致
        x_features, costs = batch
        x_features = x_features.to(self.device)
        costs = costs.to(self.device)
        logits = self(x_features)

        # loss = lce_surrogate_loss(outputs=logits, costs=costs)
        loss = asymmetric_lce_loss(outputs=logits, costs=costs, defer_penalty=1.0)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_features = batch[0]
        x_features = x_features.to(self.device)
        logits = self(x_features)
        # _, preds = torch.max(logits, 1)
        transformed_logits = asymmetric_softmax_transformation(logits, defer_penalty=1.0)
        _, preds = torch.max(transformed_logits, 1)
        return preds

    def configure_optimizers(self):

        optimizer = getattr(torch.optim, self.hparams.optimizer_name)(
            self.pmodel.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer