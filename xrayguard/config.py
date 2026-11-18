"""Configuration: defaults + YAML file + command-line overrides (--set a.b=value)."""
import copy
import os

import yaml

# NIH ChestX-ray14 label order (do not change: checkpoints depend on it)
LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(LABELS)

DEFAULTS = {
    "seed": 42,
    "data": {
        # Kaggle: the NIH dataset is mounted under /kaggle/input/<slug>. Files are searched recursively.
        "nih_root": "/kaggle/input/data",
        "work_dir": "/kaggle/working/xrayguard",
        "cache_dir": None,           # None -> <work_dir>/cache
        "cache_size": 256,           # images are resized once to this square size (uint8 memmap)
        "val_frac": 0.10,            # patient-wise validation split carved from train_val_list.txt
        "use_cache": True,
        "clahe": False,              # contrast-limited histogram equalisation (ablation flag)
    },
    "model": {
        "backbone": "densenet121",   # densenet121 | resnet50 | efficientnet_v2_s | convnext_tiny
        "pretrained": True,
        "dropout": 0.2,              # head dropout; also used for MC-dropout uncertainty
    },
    "train": {
        "img_size": 224,
        "epochs": 8,
        "batch_size": 64,
        "lr": 2.0e-4,
        "weight_decay": 1.0e-4,
        "warmup_frac": 0.05,
        "loss": "bce",               # bce | wbce | focal | asl
        "ema": 0.999,                # 0 disables EMA weights
        "amp": True,
        "num_workers": 4,
        "patience": 3,               # early stopping on validation mean AUROC
        "multi_gpu": True,           # use DataParallel when >1 GPU (Kaggle T4 x2)
        "max_train_batches": None,   # debug: limit batches per epoch
        "rot_deg": 7.0,
        "crop_scale_min": 0.80,
    },
    "calib": {
        "precision_target": 0.60,    # "present" (definite) statements need >= this precision on validation
        "min_support": 20,           # minimum predicted positives for the precision-targeted threshold
        "mc_samples": 30,
        "uncert_quantile": 0.90,     # MC-std above this validation quantile downgrades present -> possible
    },
    "evidence": {
        "cam_thresh": 0.5,
        "bilateral_frac": 0.30,
    },
    "report": {
        "llm_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "llm_batch": 8,
        "llm_max_new_tokens": 220,
        "max_regen": 1,              # LLM regenerations with verifier feedback before deterministic correction
        "allow_extent": False,       # if False, size/severity words are never supported (CAM area is not a size measure)
    },
    "eval": {
        "n_boot": 500,
        "report_n": 2000,            # test images used for report-level evaluation
        "llm_n": 500,                # test images used for LLM report generation (slower)
        "batch_size": 128,
    },
    "iu": {
        "root": "/kaggle/input/chest-xrays-indiana-university",
        "n": None,
    },
}


def _deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    d = cfg["data"]
    if not d.get("cache_dir"):
        d["cache_dir"] = os.path.join(d["work_dir"], "cache")
    os.makedirs(d["work_dir"], exist_ok=True)
    return cfg


def wpath(cfg, name):
    """Path inside the working directory."""
    return os.path.join(cfg["data"]["work_dir"], name)
