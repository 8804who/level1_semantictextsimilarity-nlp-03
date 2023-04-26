import torch
import torch.nn as nn


def L1_loss(logits, target):
    loss_func = nn.L1Loss()
    return loss_func(logits, target)


def mse_loss(logits, target):
    loss_func = nn.MSELoss()
    return loss_func(logits, target)


def BCEWithLogitsLoss(logits, target):
    loss_func = nn.BCEWithLogitsLoss()
    return loss_func(logits, target)

    
def ContrastiveLoss(logits, target) :
    margin = 0.1
    square_logits = torch.square(logits)
    margin_square = torch.square(torch.max(margin - logits, torch.zeros_like(logits)))
    loss = torch.mean(target * square_logits + (1 - target) * margin_square)
    return loss

loss_config = {
    "l1": L1_loss,
    "mse": mse_loss,
    "bce": BCEWithLogitsLoss,
    "contrastive": ContrastiveLoss
}
