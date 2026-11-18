"""GPU training: AMP, channels-last, AdamW + warmup-cosine, EMA weights, AUROC-based model selection,
early stopping, full resume."""
import copy
import json
import math
import os
import random
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import LABELS, wpath
from .data import CXRDataset, load_manifest, open_cache, worker_init
from .losses import build_loss
from .metrics import mean_auroc
from .models import build_model


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def amp_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


def make_loader(ds, bs, shuffle, workers, drop_last=False):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      drop_last=drop_last, worker_init_fn=worker_init, persistent_workers=workers > 0)


@torch.no_grad()
def predict_logits(model, loader, dev, amp=True, desc="predict"):
    model.eval()
    out, ys, ids = [], [], []
    use_amp = amp and dev == "cuda"
    for x, y, i in tqdm(loader, desc=desc, leave=False):
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=amp_dtype(), enabled=use_amp):
            out.append(model(x).float().cpu())
        ys.append(y)
        ids.append(i)
    return torch.cat(out).numpy(), torch.cat(ys).numpy(), torch.cat(ids).numpy()


def lr_lambda(total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def save_checkpoint(state, path):
    """Replace a checkpoint only after the new file has finished writing."""
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), suffix=".tmp", delete=False) as file:
        temporary = file.name
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def train(cfg, resume=False, tag=None):
    import pandas as pd
    seed_all(cfg["seed"])
    t = cfg["train"]
    dev = device()
    tag = tag or f"{cfg['model']['backbone']}_{t['loss']}"
    run_dir = wpath(cfg, f"runs/{tag}")
    os.makedirs(run_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)

    df = load_manifest(cfg)
    cache = open_cache(cfg)
    if cache is None:
        print("[train] WARNING: no image cache found -> decoding PNGs on the fly (slow). Run `python run.py cache`.")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    dl_tr = make_loader(CXRDataset(tr, cfg, True, cache), t["batch_size"], True, t["num_workers"], drop_last=True)
    dl_va = make_loader(CXRDataset(va, cfg, False, cache), t["batch_size"] * 2, False, t["num_workers"])
    print(f"[train] device={dev} train={len(tr)} val={len(va)} steps/epoch={len(dl_tr)} run={run_dir}")

    model = build_model(cfg).to(dev).to(memory_format=torch.channels_last)
    criterion = build_loss(t["loss"], tr[LABELS].values.astype(np.float32)).to(dev)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=t["lr"])
    steps_per_epoch = len(dl_tr) if not t.get("max_train_batches") else min(len(dl_tr), t["max_train_batches"])
    total = t["epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, int(t["warmup_frac"] * total)))
    use_amp = t["amp"] and dev == "cuda"
    adt = amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and adt == torch.float16)
    ema = EMA(model, t["ema"]) if t.get("ema") else None

    start, best, bad_epochs, history = 0, -1.0, 0, []
    last_p = os.path.join(run_dir, "last.pt")
    if resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        if ema and ck.get("ema"):
            ema.shadow.load_state_dict(ck["ema"])
        start, best, bad_epochs, history = ck["epoch"] + 1, ck["best"], ck["bad_epochs"], ck["history"]
        print(f"[train] resumed from epoch {start} (best val AUROC {best:.4f})")

    net = nn.DataParallel(model) if (t["multi_gpu"] and torch.cuda.device_count() > 1) else model
    for ep in range(start, t["epochs"]):
        model.train()
        t0, run_loss, nb = time.time(), 0.0, 0
        pbar = tqdm(dl_tr, desc=f"epoch {ep + 1}/{t['epochs']}", total=steps_per_epoch)
        for k, (x, y, _) in enumerate(pbar):
            if t.get("max_train_batches") and k >= t["max_train_batches"]:
                break
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=adt, enabled=use_amp):
                logits = net(x)
            loss = criterion(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model)
            run_loss += loss.item()
            nb += 1
            if k % 50 == 0:
                pbar.set_postfix(loss=f"{run_loss / nb:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        eval_model = ema.shadow if ema else model
        lg, yy, _ = predict_logits(eval_model, dl_va, dev, t["amp"], desc="val")
        val_loss = float(nn.functional.binary_cross_entropy_with_logits(torch.tensor(lg), torch.tensor(yy)))
        auc, per = mean_auroc(lg, yy)
        rec = dict(epoch=ep + 1, train_loss=run_loss / max(nb, 1), val_bce=val_loss, val_mean_auroc=auc,
                   minutes=(time.time() - t0) / 60, **{f"auc_{k}": v for k, v in per.items()})
        history.append(rec)
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        print(f"epoch {ep + 1}: train loss {rec['train_loss']:.4f} | val BCE {val_loss:.4f} | "
              f"val mean AUROC {auc:.4f} | {rec['minutes']:.1f} min")
        if auc > best:
            best, bad_epochs = auc, 0
            save_checkpoint({"model": model.state_dict(), "ema": ema.shadow.state_dict() if ema else None,
                        "cfg": cfg, "epoch": ep, "val_auroc": per, "val_mean_auroc": auc, "labels": LABELS},
                       os.path.join(run_dir, "best.pt"))
            print(f"  saved best (val mean AUROC {auc:.4f})")
        else:
            bad_epochs += 1
        save_checkpoint({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "ema": ema.shadow.state_dict() if ema else None,
                    "epoch": ep, "best": best, "bad_epochs": bad_epochs, "history": history, "cfg": cfg}, last_p)
        if bad_epochs >= t["patience"]:
            print(f"[train] early stop: no AUROC improvement for {bad_epochs} epochs")
            break
    # make this run the active model for the later stages
    with open(wpath(cfg, "active_run.txt"), "w") as f:
        f.write(tag)
    print(f"[train] done. best val mean AUROC {best:.4f}. active run -> {tag}")
    return run_dir


def active_run_dir(cfg, tag=None):
    if tag is None:
        p = wpath(cfg, "active_run.txt")
        if not os.path.exists(p):
            raise FileNotFoundError("No trained model found - run:  python run.py train")
        tag = open(p).read().strip()
    return wpath(cfg, f"runs/{tag}")
