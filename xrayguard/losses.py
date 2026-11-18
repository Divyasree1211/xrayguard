"""Multi-label losses: BCE, class-weighted BCE, focal loss, asymmetric loss (ASL)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pw", torch.as_tensor(pos_weight, dtype=torch.float32))

    def forward(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pw)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, y):
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        loss = ce * (1 - pt) ** self.gamma
        if self.alpha is not None:
            loss = loss * (self.alpha * y + (1 - self.alpha) * (1 - y))
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Ridnik et al., ICCV 2021."""

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0) if self.clip else p
        lp = y * torch.log(p.clamp(min=self.eps)) * (1 - p) ** self.gp
        ln = (1 - y) * torch.log((1 - p_neg).clamp(min=self.eps)) * p_neg ** self.gn
        return -(lp + ln).mean()


def build_loss(name, train_labels):
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "wbce":
        pos = train_labels.mean(0).clip(1e-4, None)
        return WeightedBCE(((1 - pos) / pos).clip(1, 50))
    if name == "focal":
        return FocalLoss()
    if name == "asl":
        return AsymmetricLoss()
    raise ValueError(f"unknown loss {name}")
