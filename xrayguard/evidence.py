"""Calibration (per-class temperature), assertion thresholds, MC-dropout uncertainty, CAM localisation and the
structured evidence layer that the report generator and the verifier both read."""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .config import LABELS, NUM_CLASSES
from .metrics import classwise_ece, ece, f1_threshold, precision_threshold, sigmoid

ABSENT, POSSIBLE, PRESENT = 0, 1, 2
TIER_NAME = {ABSENT: "absent", POSSIBLE: "possible", PRESENT: "present"}
TIER_CONF = {ABSENT: "Low", POSSIBLE: "Moderate", PRESENT: "High"}

# findings for which a side (left/right/bilateral) is meaningful, and those that also get a lung zone
LATERAL = {"Atelectasis", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
           "Consolidation", "Fibrosis", "Pleural_Thickening"}
ZONED = {"Atelectasis", "Infiltration", "Mass", "Nodule", "Pneumonia", "Consolidation", "Fibrosis"}
FIXED_REGION = {"Cardiomegaly": "cardiac silhouette", "Hernia": "lower mediastinum / hiatus",
                "Edema": "both lungs", "Emphysema": "both lungs"}


# ----------------------------------------------------------------------------- calibration
def fit_temperatures(logits, y, iters=200):
    """Per-class temperature scaling (Guo et al., 2017) fitted by minimising validation BCE."""
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    log_t = torch.zeros(lg.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lg / log_t.exp(), yt, reduction="none").mean(0).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().detach().numpy().clip(0.05, 20.0)


def fit_global_temperature(logits, y):
    t = fit_temperatures(logits.reshape(-1, 1), y.reshape(-1, 1))
    return float(t[0])


def mc_std(model, pooled, temps, n, device, batch=2048):
    """Std of calibrated probabilities over MC-dropout samples of the head."""
    out = []
    T = torch.tensor(temps, dtype=torch.float32, device=device)
    with torch.no_grad():
        for s in range(0, len(pooled), batch):
            p = torch.as_tensor(pooled[s:s + batch], device=device)
            samples = torch.sigmoid(model.mc_logits(p, n) / T)
            out.append(samples.std(0).cpu())
    return torch.cat(out).numpy()


def fit_calibration(cfg, logits, y, pooled, model, device):
    c = cfg["calib"]
    temps = fit_temperatures(logits, y)
    t_glob = fit_global_temperature(logits, y)
    p_raw, p_cal = sigmoid(logits), sigmoid(logits / temps)
    std = mc_std(model, pooled, temps, c["mc_samples"], device)
    th = {}
    for j, n in enumerate(LABELS):
        t_mid = f1_threshold(p_cal[:, j], y[:, j])
        t_high = max(t_mid, precision_threshold(p_cal[:, j], y[:, j], c["precision_target"], c["min_support"]))
        sel = p_cal[:, j] >= t_mid
        ref = std[sel, j] if sel.sum() >= 10 else std[:, j]
        th[n] = dict(mid=float(t_mid), high=float(t_high), std_cut=float(np.quantile(ref, c["uncert_quantile"])))
    res = dict(
        temperature=[float(t) for t in temps], global_temperature=t_glob, thresholds=th,
        ece_raw=ece(p_raw, y), ece_global_T=ece(sigmoid(logits / t_glob), y), ece_classwise_T=ece(p_cal, y),
        classwise_ece_raw=classwise_ece(p_raw, y), classwise_ece_T=classwise_ece(p_cal, y),
        brier_raw=float(((p_raw - y) ** 2).mean()), brier_T=float(((p_cal - y) ** 2).mean()),
        never_definite=[n for n in LABELS if th[n]["high"] > 1.0],
    )
    return res


def load_calibration(run_dir):
    p = os.path.join(run_dir, "calib.json")
    if not os.path.exists(p):
        raise FileNotFoundError("calib.json missing - run:  python run.py calibrate")
    return json.load(open(p))


def calibrated_probs(logits, calib):
    return sigmoid(logits / np.asarray(calib["temperature"]))


def assign_tiers(p, std, calib, use_uncertainty=True):
    """Three-way decision per finding. present: p >= high (precision-targeted) and MC-std below the cut;
    possible: p >= mid (F1-optimal); absent otherwise."""
    th = calib["thresholds"]
    mid = np.array([th[n]["mid"] for n in LABELS])
    high = np.array([th[n]["high"] for n in LABELS])
    cut = np.array([th[n]["std_cut"] for n in LABELS])
    tier = np.zeros(p.shape, dtype=np.int8)
    tier[p >= mid] = POSSIBLE
    pres = p >= high
    if use_uncertainty and std is not None:
        pres &= std <= cut
    tier[pres] = PRESENT
    return tier


# ----------------------------------------------------------------------------- localisation
def side_of(mass_left_img, mass_right_img, bilateral_frac):
    """Radiographs are displayed with the patient's right on the image left."""
    tot = mass_left_img + mass_right_img
    if tot <= 0:
        return None
    frac_left = mass_left_img / tot
    if min(frac_left, 1 - frac_left) > bilateral_frac:
        return "bilateral"
    return "right" if frac_left > 0.5 else "left"


def zone_of(cy_rel):
    return "upper" if cy_rel < 1 / 3 else ("mid" if cy_rel < 2 / 3 else "lower")


def cam_region(cam, finding, cam_thresh=0.5, bilateral_frac=0.3):
    """Turn a [0,1] CAM into side / zone / area / peak location."""
    H, W = cam.shape
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    mask = cam >= cam_thresh
    area = float(mask.mean())
    out = dict(side=None, zone=None, area=area, peak=(int(peak[0]), int(peak[1])), region=FIXED_REGION.get(finding))
    if finding in LATERAL and mask.any():
        m = cam * mask
        out["side"] = side_of(float(m[:, : W // 2].sum()), float(m[:, W // 2:].sum()), bilateral_frac)
        ys = np.nonzero(mask)[0]
        if finding in ZONED:
            out["zone"] = zone_of(float(ys.mean()) / H)
        out["region"] = " ".join(x for x in [out["side"], out["zone"] and f"{out['zone']} zone"] if x) or None
    return out


def box_region(x, y, w, h, img_w=1024, img_h=1024, bilateral_frac=0.3):
    """Side/zone of a radiologist bounding box, using the same rules as cam_region (for agreement tests)."""
    xl, xr = max(0, x), min(img_w, x + w)
    left_part = max(0.0, min(xr, img_w / 2) - xl)
    right_part = max(0.0, xr - max(xl, img_w / 2))
    return dict(side=side_of(left_part, right_part, bilateral_frac), zone=zone_of((y + h / 2) / img_h))


# ----------------------------------------------------------------------------- evidence
class EvidenceEngine:
    """Image(s) -> calibrated probabilities, MC uncertainty, tiers, CAMs and region descriptors."""

    def __init__(self, model, calib, cfg, device):
        self.model, self.calib, self.cfg, self.dev = model, calib, cfg, device
        self.T = torch.tensor(calib["temperature"], dtype=torch.float32, device=device)

    @torch.no_grad()
    def run(self, x, need_cams=True, use_uncertainty=True, cam_size=None, all_regions=False):
        """x: uint8 tensor (B,1,H,W). Returns dict of numpy arrays + list of per-image evidence dicts."""
        x = x.to(self.dev)
        logits, pooled, fmap = self.model.forward_all(x)
        logits = logits.float()
        p = torch.sigmoid(logits / self.T).cpu().numpy()
        samples = torch.sigmoid(self.model.mc_logits(pooled, self.cfg["calib"]["mc_samples"]) / self.T)
        std = samples.std(0).cpu().numpy()
        tier = assign_tiers(p, std, self.calib, use_uncertainty)
        cams = None
        if need_cams:
            cams = self.model.cam(fmap, size=cam_size or tuple(x.shape[-2:])).cpu().numpy()
        evid = [self.evidence(p[b], std[b], tier[b], None if cams is None else cams[b], all_regions)
                for b in range(len(p))]
        return dict(logits=logits.cpu().numpy(), prob=p, std=std, tier=tier, cams=cams, evidence=evid)

    def evidence(self, p, std, tier, cams, all_regions=False):
        e = self.cfg["evidence"]
        ev = {}
        for j, n in enumerate(LABELS):
            d = dict(prob=float(p[j]), std=float(std[j]), tier=TIER_NAME[int(tier[j])],
                     confidence=TIER_CONF[int(tier[j])], side=None, zone=None, region=None, area=None)
            if cams is not None and (tier[j] > ABSENT or all_regions):
                d.update(cam_region(cams[j], n, e["cam_thresh"], e["bilateral_frac"]))
                d.pop("peak", None)
            ev[n] = d
        return ev


def evidence_table(ev, include_absent=False):
    """Human/LLM-readable evidence layer (the 'evidence representation' stage)."""
    rows = ["FINDING              PROB   CONFIDENCE  STATUS    LOCATION"]
    for n, d in sorted(ev.items(), key=lambda kv: -kv[1]["prob"]):
        if d["tier"] == "absent" and not include_absent:
            continue
        rows.append(f"{n:<20} {d['prob']:.2f}   {d['confidence']:<10}  {d['tier']:<8}  {d['region'] or '-'}")
    if len(rows) == 1:
        rows.append("(no finding above the reporting threshold)")
    return "\n".join(rows)


def overlay(gray, cams, names):
    """Max-CAM heat-map over the image for the listed findings."""
    import cv2
    base = np.stack([gray] * 3, -1).astype(np.float32)
    if cams is None or not names:
        return base.astype(np.uint8)
    idx = [LABELS.index(n) for n in names]
    cam = cams[idx].max(0)
    if cam.shape != gray.shape:
        cam = cv2.resize(cam, gray.shape[::-1])
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    a = (cam[..., None] * 0.5).astype(np.float32)
    return (base * (1 - a) + heat * a).astype(np.uint8)
