import copy
import torch
import torch.nn.functional as F
import lightning as L

eps_cst = 1e-8


def asymmetric_softmax_transformation(logits: torch.Tensor) -> torch.Tensor:

    action_logits = logits[:, :2]
    psi_action = F.softmax(action_logits, dim=1)

    full_softmax_probs = F.softmax(logits, dim=1)

    defer_prob_from_softmax = full_softmax_probs[:, 2].unsqueeze(1)

    max_action_prob = torch.max(full_softmax_probs[:, :2], dim=1, keepdim=True)[0]

    normalization_factor = 1 - max_action_prob

    psi_defer = defer_prob_from_softmax / (normalization_factor + eps_cst)

    return torch.cat([psi_action, psi_defer], dim=1)


class CTLD_Model(L.LightningModule):

    def __init__(self, pmodel, lr=1e-3, weight_decay=1e-5, optimizer_name='Adam', gamma: str = None):
        super().__init__()

        self.pmodel = pmodel
        self.add_module("pmodel", self.pmodel)
        self.save_hyperparameters(ignore=['pmodel'])
        self.gamma = gamma

    def forward(self, x):
        return self.pmodel(x)


    def _calculate_loss(self, logits, targets):

        pred_probs = asymmetric_softmax_transformation(logits)

        target_action_probs = targets[:, :2]
        target_defer_prob = targets[:, 2]

        pred_action_probs = pred_probs[:, :2]
        pred_defer_prob = pred_probs[:, 2]

        loss_action = -torch.sum(target_action_probs * torch.log(pred_action_probs + eps_cst), dim=1)

        loss_defer = F.binary_cross_entropy(pred_defer_prob, target_defer_prob, reduction='none')

        mean_loss_action = torch.mean(loss_action)
        mean_loss_defer = torch.mean(loss_defer)

        lambda_defer = 0.2

        total_loss = torch.mean(loss_action + lambda_defer *  loss_defer)

        return total_loss, mean_loss_action, mean_loss_defer

    def training_step(self, batch, batch_idx):
        x_features, p_tilde_targets = batch


        logits = self(x_features)

        loss, action_loss, defer_loss = self._calculate_loss(logits, p_tilde_targets)

        return loss

    def validation_step(self, batch, batch_idx):


        x_features, p_tilde_targets = batch
        x_features = x_features.to(self.device)
        p_tilde_targets = p_tilde_targets.to(self.device)

        logits = self(x_features)

        loss, action_loss, defer_loss = self._calculate_loss(logits, p_tilde_targets)


        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False
        )

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        x_features = batch[0]
        x_features = x_features.to(self.device)
        logits = self(x_features)
        _, preds = torch.max(logits, 1)
        return preds

    def configure_optimizers(self):

        optimizer = getattr(torch.optim, self.hparams.optimizer_name)(
            self.pmodel.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
