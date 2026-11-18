"""Shared inference: load the trained model + calibration, run splits, analyse single images."""
import json
import os

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import LABELS
from .data import CXRDataset, eval_transform, load_manifest, open_cache, to_gray_uint8
from .evidence import EvidenceEngine, fit_calibration, load_calibration, overlay
from .models import load_model
from .train import active_run_dir, amp_dtype, device, make_loader


def load_trained(cfg, tag=None):
    run_dir = active_run_dir(cfg, tag)
    dev = device()
    model, ck = load_model(cfg, os.path.join(run_dir, "best.pt"), dev)
    return model, ck, run_dir, dev


@torch.no_grad()
def forward_split(model, df, cfg, dev, desc):
    """logits + pooled features for a dataframe (deterministic, no augmentation)."""
    cache = open_cache(cfg)
    dl = make_loader(CXRDataset(df, cfg, False, cache), cfg["eval"]["batch_size"], False, cfg["train"]["num_workers"])
    L, P, Y, I = [], [], [], []
    use_amp = cfg["train"]["amp"] and dev == "cuda"
    for x, y, i in tqdm(dl, desc=desc, leave=False):
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=amp_dtype(), enabled=use_amp):
            lg, pooled, _ = model.forward_all(x)
        L.append(lg.float().cpu())
        P.append(pooled.float().cpu())
        Y.append(y)
        I.append(i)
    return (torch.cat(L).numpy(), torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(I).numpy())


def calibrate(cfg, tag=None):
    model, ck, run_dir, dev = load_trained(cfg, tag)
    df = load_manifest(cfg)
    lg, pooled, y, idx = forward_split(model, df[df.split == "val"], cfg, dev, "val")
    res = fit_calibration(cfg, lg, y, pooled, model, dev)
    res["val_mean_auroc"] = ck.get("val_mean_auroc")
    json.dump(res, open(os.path.join(run_dir, "calib.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(run_dir, "val_outputs.npz"), logits=lg, pooled=pooled, y=y, idx=idx)
    print(f"[calibrate] ECE raw {res['ece_raw']:.4f} -> global-T {res['ece_global_T']:.4f} -> per-class-T "
          f"{res['ece_classwise_T']:.4f}")
    print(f"[calibrate] findings that never reach the {cfg['calib']['precision_target']:.0%} precision target "
          f"(always reported as 'possible'): {res['never_definite']}")
    for n in LABELS:
        t = res["thresholds"][n]
        print(f"  {n:<20} T={res['temperature'][LABELS.index(n)]:.2f}  mid={t['mid']:.3f}  high={t['high']:.3f}  "
              f"std_cut={t['std_cut']:.4f}")
    return res


def predict_split(cfg, split="test", tag=None):
    """Save calibrated probabilities + MC-dropout std for a split (input to all evaluations)."""
    from .evidence import calibrated_probs, mc_std
    model, ck, run_dir, dev = load_trained(cfg, tag)
    calib = load_calibration(run_dir)
    df = load_manifest(cfg)
    lg, pooled, y, idx = forward_split(model, df[df.split == split], cfg, dev, split)
    p = calibrated_probs(lg, calib)
    std = mc_std(model, pooled, calib["temperature"], cfg["calib"]["mc_samples"], dev)
    out = os.path.join(run_dir, f"{split}_outputs.npz")
    np.savez_compressed(out, logits=lg, prob=p, std=std, y=y, idx=idx)
    print(f"[predict] {split}: {len(idx)} images -> {out}")
    return out


class XRayGuard:
    """End-to-end analysis of one image (used by the CLI `analyze` command and the Gradio demo)."""

    def __init__(self, cfg, tag=None, llm=False):
        self.cfg = cfg
        self.model, _, self.run_dir, self.dev = load_trained(cfg, tag)
        self.calib = load_calibration(self.run_dir)
        self.engine = EvidenceEngine(self.model, self.calib, cfg, self.dev)
        self.llm = None
        if llm:
            from .report import LLMGenerator
            self.llm = LLMGenerator(cfg["report"]["llm_model"], self.dev, cfg["report"]["llm_max_new_tokens"])

    def analyze(self, image, include_visuals=False):
        from .report import generate_verified, template_report
        from .verifier import correct_report, split_report, verify_report
        import cv2
        g = to_gray_uint8(image)
        g = cv2.resize(g, (self.cfg["data"]["cache_size"],) * 2, interpolation=cv2.INTER_AREA)
        if self.cfg["data"].get("clahe"):
            from .data import apply_clahe
            g = apply_clahe(g)
        g = eval_transform(g, self.cfg["train"]["img_size"])
        x = torch.from_numpy(g)[None, None]
        out = self.engine.run(x)
        ev = out["evidence"][0]
        shown = [n for n, d in ev.items() if d["tier"] != "absent"]
        res = dict(evidence=ev, overlay=overlay(g, out["cams"][0], shown))
        if include_visuals:
            res["processed_image"] = overlay(g, None, [])
            res["finding_overlays"] = {n: overlay(g, out["cams"][0], [n]) for n in shown}
        if self.llm is not None:
            r = generate_verified([ev], self.llm, self.cfg, verify_report, correct_report, split_report)[0]
            df, di = split_report(r["draft"])
            res.update(draft=r["draft"], issues_first=verify_report(df, di, ev, self.cfg), findings=r["findings"],
                       impression=r["impression"], log=r["log"], remaining=r["remaining"], generator="llm+verifier")
        else:
            f, i = template_report(ev)
            res.update(draft=f"FINDINGS: {f} IMPRESSION: {i}", issues_first=verify_report(f, i, ev, self.cfg),
                       findings=f, impression=i, log=[], remaining=[], generator="template")
        return res
