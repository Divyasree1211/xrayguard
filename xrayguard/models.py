"""Classifier backbones with a GAP + dropout + linear head (so class-activation maps are exact and cheap)."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import LABELS, NUM_CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _backbone(name, pretrained):
    w = "DEFAULT" if pretrained else None
    if name == "densenet121":
        m = models.densenet121(weights=w)
        return nn.Sequential(m.features, nn.ReLU(inplace=False)), 1024
    if name == "resnet50":
        m = models.resnet50(weights=w)
        return nn.Sequential(*list(m.children())[:-2]), 2048
    if name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=w)
        return m.features, 1280
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=w)
        return m.features, 768
    raise ValueError(f"unknown backbone {name}")


class CXRNet(nn.Module):
    def __init__(self, backbone="densenet121", n_classes=NUM_CLASSES, dropout=0.2, pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        chest_model = None
        if backbone == "densenet121_chexpert":
            from torchxrayvision.models import DenseNet

            chest_model = DenseNet(
                weights="densenet121-res224-chex" if pretrained else None,
                cache_dir=str(Path(__file__).resolve().parent.parent / "pretrained"),
            )
            self.backbone = nn.Sequential(chest_model.features, nn.ReLU(inplace=False))
            self.dim = chest_model.classifier.in_features
        else:
            self.backbone, self.dim = _backbone(backbone, pretrained)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.dim, n_classes)
        if chest_model is not None and pretrained:
            with torch.no_grad():
                for target, finding in enumerate(LABELS[:n_classes]):
                    if finding in chest_model.pathologies:
                        source = chest_model.pathologies.index(finding)
                        self.fc.weight[target].copy_(chest_model.classifier.weight[source])
                        self.fc.bias[target].copy_(chest_model.classifier.bias[source])
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def normalize(self, x):
        """Convert uint8 or unit-range images to the selected backbone's input scale."""
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if self.backbone_name == "densenet121_chexpert":
            return x.mean(1, keepdim=True) * 2048.0 - 1024.0
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - self.mean) / self.std

    def feature_map(self, x):
        return self.backbone(self.normalize(x))

    def forward(self, x):
        f = self.feature_map(x)
        return self.fc(self.drop(f.mean((2, 3))))

    def forward_all(self, x):
        """Logits, pooled features and feature map in one pass (used for CAM + MC dropout)."""
        f = self.feature_map(x)
        pooled = f.mean((2, 3))
        return self.fc(pooled), pooled, f

    def mc_logits(self, pooled, n=30):
        """MC-dropout on the head only: the backbone runs once, the head is sampled n times (cheap)."""
        p = pooled.float()
        keep = 1.0 - self.drop.p
        if self.drop.p <= 0:
            return self.fc(p).unsqueeze(0)
        masks = (torch.rand(n, *p.shape, device=p.device) < keep).float() / keep
        return torch.einsum("nbd,cd->nbc", masks * p.unsqueeze(0), self.fc.weight.float()) + self.fc.bias.float()

    def cam(self, fmap, size=None):
        """Class activation maps for all classes: (B, C, h, w), upsampled to `size` if given, min-max scaled per map.
        Standard CAM (Zhou et al., 2016): raw weighted sum, then min-max scaling. No ReLU before scaling, so a map whose
        values are all negative still has a defined peak (with ReLU it would collapse to zero and give no location)."""
        cam = torch.einsum("bdhw,cd->bchw", fmap.float(), self.fc.weight.float())
        if size is not None:
            cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
        b, c = cam.shape[:2]
        flat = cam.view(b, c, -1)
        mn, mx = flat.min(-1, keepdim=True)[0], flat.max(-1, keepdim=True)[0]
        return ((flat - mn) / (mx - mn).clamp(min=1e-8)).view_as(cam)


def build_model(cfg, pretrained=None):
    m = cfg["model"]
    return CXRNet(m["backbone"], NUM_CLASSES, m["dropout"], m["pretrained"] if pretrained is None else pretrained)


def load_model(cfg, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = ck.get("cfg", {}).get("model", cfg["model"])
    model = CXRNet(mcfg["backbone"], NUM_CLASSES, mcfg["dropout"], pretrained=False)
    state = ck.get("ema") or ck["model"]
    model.load_state_dict(state)
    return model.to(device).eval(), ck
