"""Evaluation for the paper.

eval-cls     classification (AUROC with patient-level bootstrap CIs), calibration, subgroups, selective prediction,
             statement-tier reliability (PPV of 'present', NPV of 'absent')
eval-loc     localisation on the 984 NIH radiologist boxes (pointing game, IoU, side/zone agreement) + baselines
eval-report  report-level hallucination analysis against NIH labels for several generators (+ optional LLM)
stress-test  error-injection test of the verifier/corrector (detection recall, false alarms, repair rate)
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import LABELS, wpath
from .data import CXRDataset, load_manifest, open_cache
from .evidence import (ABSENT, LATERAL, POSSIBLE, PRESENT, TIER_NAME, ZONED, EvidenceEngine, assign_tiers, box_region,
                       cam_region, load_calibration, overlay)
from .metrics import (binary_stats, classwise_ece, ece, mean_auroc, patient_bootstrap, per_class_auprc,
                      per_class_auroc, reliability_curve, sigmoid)
from .pipeline import load_trained
from .report import finding_sentence, NAME, template_report
from .train import make_loader
from .verifier import (ERROR_TYPES, WARNING_TYPES, correct_report, errors_only, split_report, split_sentences,
                       text_to_labels, verify_report)


def results_dir(cfg):
    d = wpath(cfg, "results")
    os.makedirs(d, exist_ok=True)
    return d


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[saved] {path}")


def _load_outputs(run_dir, split):
    p = os.path.join(run_dir, f"{split}_outputs.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing - run:  python run.py predict --split {split}")
    return dict(np.load(p))


# ============================================================================= classification
def eval_cls(cfg, tag=None):
    from .train import active_run_dir
    run_dir = active_run_dir(cfg, tag)
    calib = load_calibration(run_dir)
    o = _load_outputs(run_dir, "test")
    df = load_manifest(cfg).set_index("idx").loc[o["idx"]].reset_index()
    y, p, std, lg = o["y"], o["prob"], o["std"], o["logits"]
    groups = df.patient.values
    nb = cfg["eval"]["n_boot"]
    res = {"n_images": int(len(y)), "n_patients": int(len(np.unique(groups))), "run": os.path.basename(run_dir)}

    mean_auc, per = mean_auroc(p, y)
    res["mean_auroc"] = mean_auc
    res["mean_auroc_ci"] = patient_bootstrap(lambda ix: mean_auroc(p[ix], y[ix])[0], groups, nb)
    res["per_class"] = {}
    auprc = per_class_auprc(p, y)
    mids = np.array([calib["thresholds"][n]["mid"] for n in LABELS])
    for j, n in enumerate(LABELS):
        ci = patient_bootstrap(
            lambda ix, j=j, n=n: per_class_auroc(
                p[ix, j:j + 1], y[ix, j:j + 1], names=[n]
            ).get(n, float("nan")),
            groups, nb, seed=j + 1,
        )
        st = binary_stats((p[:, j] >= mids[j]).astype(int), y[:, j].astype(int))
        res["per_class"][n] = dict(auroc=per.get(n), auroc_ci=ci, auprc=auprc.get(n), prevalence=float(y[:, j].mean()),
                                   n_pos=int(y[:, j].sum()), **{k: st[k] for k in ("precision", "recall",
                                                                                    "specificity", "f1")})
    res["macro_f1_at_mid"] = float(np.mean([v["f1"] for v in res["per_class"].values()]))

    # calibration on the held-out test set
    pr = sigmoid(lg)
    res["calibration"] = dict(ece_raw=ece(pr, y), ece_T=ece(p, y), classwise_ece_raw=classwise_ece(pr, y),
                              classwise_ece_T=classwise_ece(p, y), brier_raw=float(((pr - y) ** 2).mean()),
                              brier_T=float(((p - y) ** 2).mean()),
                              reliability_raw=reliability_curve(pr, y), reliability_T=reliability_curve(p, y))

    # subgroups (sex, age, view position)
    sub = {}
    age_bin = pd.cut(df.age, [0, 40, 60, 200], right=False, labels=["<40", "40-59", ">=60"])
    for name, mask in [("sex=M", df.sex == "M"), ("sex=F", df.sex == "F"), ("view=PA", df.view == "PA"),
                       ("view=AP", df.view == "AP")] + [(f"age{b}", age_bin == b) for b in ["<40", "40-59", ">=60"]]:
        m = mask.values
        if m.sum() > 100:
            a, _ = mean_auroc(p[m], y[m])
            sub[name] = dict(n=int(m.sum()), mean_auroc=a,
                             ci=patient_bootstrap(lambda ix, m=m: mean_auroc(p[m][ix], y[m][ix])[0], groups[m],
                                                  max(100, nb // 5)))
    res["subgroups"] = sub

    # selective prediction: drop the most uncertain (image, finding) pairs per class
    covs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    rng = np.random.RandomState(0)
    sel = {"coverage": covs, "mc_std": [], "random": []}
    for c in covs:
        a_u, a_r = [], []
        for j in range(len(LABELS)):
            k = max(10, int(round(c * len(y))))
            keep_u = np.argsort(std[:, j])[:k]
            keep_r = rng.permutation(len(y))[:k]
            for keep, acc in ((keep_u, a_u), (keep_r, a_r)):
                yy = y[keep, j]
                if 0 < yy.sum() < len(yy):
                    from sklearn.metrics import roc_auc_score
                    acc.append(roc_auc_score(yy, p[keep, j]))
        sel["mc_std"].append(float(np.mean(a_u)))
        sel["random"].append(float(np.mean(a_r)))
    res["selective"] = sel

    # reliability of the three statement tiers (what the report will assert / hedge / deny)
    tiers = assign_tiers(p, std, calib, True)
    tiers_nomc = assign_tiers(p, std, calib, False)
    res["tiers"] = {}
    for name, t in (("with_mc", tiers), ("without_mc", tiers_nomc)):
        pos_pres = y[t == PRESENT]
        pos_poss = y[t == POSSIBLE]
        neg = y[t == ABSENT]
        per_cls = {}
        for j, n in enumerate(LABELS):
            tp = t[:, j]
            per_cls[n] = dict(ppv_present=float(y[tp == PRESENT, j].mean()) if (tp == PRESENT).any() else None,
                              n_present=int((tp == PRESENT).sum()),
                              ppv_possible=float(y[tp == POSSIBLE, j].mean()) if (tp == POSSIBLE).any() else None,
                              n_possible=int((tp == POSSIBLE).sum()),
                              npv_absent=float(1 - y[tp == ABSENT, j].mean()) if (tp == ABSENT).any() else None,
                              recall_present_or_possible=float(((tp > 0) & (y[:, j] == 1)).sum() / max(1, y[:, j].sum())))
        res["tiers"][name] = dict(ppv_present=float(pos_pres.mean()) if len(pos_pres) else None,
                                  ppv_possible=float(pos_poss.mean()) if len(pos_poss) else None,
                                  npv_absent=float(1 - neg.mean()), n_present=int(len(pos_pres)),
                                  n_possible=int(len(pos_poss)), per_class=per_cls)
    _dump(res, os.path.join(results_dir(cfg), "cls_metrics.json"))
    print(f"[eval-cls] mean AUROC {mean_auc:.4f}  95% CI {res['mean_auroc_ci'][0]:.4f}-{res['mean_auroc_ci'][1]:.4f}")
    print(f"[eval-cls] test ECE raw {res['calibration']['ece_raw']:.4f} -> calibrated {res['calibration']['ece_T']:.4f}")
    t = res["tiers"]["with_mc"]
    print(f"[eval-cls] PPV(present)={t['ppv_present']}  PPV(possible)={t['ppv_possible']}  NPV(absent)={t['npv_absent']:.4f}")
    return res


# ============================================================================= localisation
def eval_loc(cfg, tag=None, n_examples=8):
    bpath = wpath(cfg, "bbox.csv")
    if not os.path.exists(bpath):
        raise FileNotFoundError("bbox.csv missing (BBox_List_2017.csv not found when building the manifest)")
    model, _, run_dir, dev = load_trained(cfg, tag)
    calib = load_calibration(run_dir)
    eng = EvidenceEngine(model, calib, cfg, dev)
    bb = pd.read_csv(bpath)
    bb = bb[bb.split != "train"].copy()           # never evaluate on images the model was trained on
    man = load_manifest(cfg).set_index("idx")
    S = cfg["train"]["img_size"]
    groups = bb.groupby(["idx", "label"])
    keys = list(groups.groups.keys())
    imgs = sorted({k[0] for k in keys})
    ds = CXRDataset(man.loc[imgs].reset_index(), cfg, False, open_cache(cfg))
    cams, tiers = {}, {}
    dl = make_loader(ds, 32, False, cfg["train"]["num_workers"])
    for x, _, ii in tqdm(dl, desc="CAMs (bbox images)", leave=False):
        out = eng.run(x, need_cams=True, cam_size=(S, S))
        for b, i in enumerate(ii.numpy()):
            cams[int(i)] = out["cams"][b].astype(np.float16)
            tiers[int(i)] = out["tier"][b]

    rows, examples = [], []
    rng = np.random.RandomState(0)
    for (i, lab), g in groups:
        j = LABELS.index(lab)
        ow, oh = man.loc[i, "orig_w"], man.loc[i, "orig_h"]
        sx, sy = S / ow, S / oh
        cam = cams[i][j].astype(np.float32)
        box_mask = np.zeros((S, S), bool)
        for _, r in g.iterrows():
            x0, y0 = int(max(0, r.x * sx)), int(max(0, r.y * sy))
            x1, y1 = int(min(S, np.ceil((r.x + r.w) * sx))), int(min(S, np.ceil((r.y + r.h) * sy)))
            box_mask[y0:y1, x0:x1] = True
        py, px = np.unravel_index(int(np.argmax(cam)), cam.shape)
        mask = cam >= cfg["evidence"]["cam_thresh"]
        inter = (mask & box_mask).sum()
        iou = inter / max(1, (mask | box_mask).sum())
        iobb = inter / max(1, mask.sum())
        rr = g.iloc[0]
        reg_cam = cam_region(cam, lab, cfg["evidence"]["cam_thresh"], cfg["evidence"]["bilateral_frac"])
        # side/zone of the union of boxes
        bx0, by0 = g.x.min(), g.y.min()
        bx1, by1 = (g.x + g.w).max(), (g.y + g.h).max()
        reg_box = box_region(bx0, by0, bx1 - bx0, by1 - by0, ow, oh, cfg["evidence"]["bilateral_frac"])
        rows.append(dict(idx=i, label=lab, split=rr.split, hit=bool(box_mask[py, px]), iou=float(iou), iobb=float(iobb),
                         center_hit=bool(box_mask[S // 2, S // 2]), box_frac=float(box_mask.mean()),
                         detected=bool(tiers[i][j] > ABSENT), cam_side=reg_cam["side"], box_side=reg_box["side"],
                         cam_zone=reg_cam["zone"], box_zone=reg_box["zone"]))
        if len(examples) < n_examples and tiers[i][j] > ABSENT and box_mask[py, px] and rng.rand() < 0.5:
            examples.append((i, lab, [tuple(v) for v in g[["x", "y", "w", "h"]].values * [sx, sy, sx, sy]]))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(results_dir(cfg), "loc_rows.csv"), index=False)

    def summ(d):
        lat = d[d.label.isin(LATERAL) & d.box_side.notna() & d.cam_side.notna()]
        zon = d[d.label.isin(ZONED) & d.box_zone.notna() & d.cam_zone.notna()]
        maj = lat.box_side.value_counts(normalize=True).max() if len(lat) else None
        return dict(n=int(len(d)), pointing=float(d.hit.mean()), iou=float(d.iou.mean()),
                    acc_iou_01=float((d.iou >= 0.1).mean()), acc_iou_025=float((d.iou >= 0.25).mean()),
                    acc_iobb_025=float((d.iobb >= 0.25).mean()), centre_baseline=float(d.center_hit.mean()),
                    random_baseline=float(d.box_frac.mean()),
                    side_agreement=float((lat.cam_side == lat.box_side).mean()) if len(lat) else None,
                    side_majority_baseline=float(maj) if maj is not None else None,
                    zone_agreement=float((zon.cam_zone == zon.box_zone).mean()) if len(zon) else None, n_side=len(lat),
                    n_zone=len(zon))

    res = {"overall": summ(R), "detected_only": summ(R[R.detected]), "per_class": {}}
    for lab, d in R.groupby("label"):
        res["per_class"][lab] = summ(d)
    _dump(res, os.path.join(results_dir(cfg), "loc_metrics.json"))
    o = res["overall"]
    print(f"[eval-loc] pointing game {o['pointing']:.3f} (centre baseline {o['centre_baseline']:.3f}, random "
          f"{o['random_baseline']:.3f}); side agreement {o['side_agreement']}")

    # example overlays for the paper
    import cv2
    exdir = os.path.join(results_dir(cfg), "cam_examples")
    os.makedirs(exdir, exist_ok=True)
    cache = open_cache(cfg)
    for k, (i, lab, boxes) in enumerate(examples):
        g = ds._image(list(ds.idx).index(i))
        g = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA)
        ov = overlay(g, cams[i].astype(np.float32), [lab])
        for (x, y, w, h) in boxes:
            cv2.rectangle(ov, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(exdir, f"{k:02d}_{lab}.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    return res


# ============================================================================= report-level evaluation
def _variant_evidence(ev, tiers_row, regions):
    """Copy of an evidence dict with different tiers (regions are taken from the all-regions evidence)."""
    out = {}
    for j, n in enumerate(LABELS):
        d = dict(ev[n])
        d["tier"] = TIER_NAME[int(tiers_row[j])]
        r = regions[n]
        d.update(side=r["side"], zone=r["zone"], region=r["region"])
        out[n] = d
    return out


def score_reports(texts, Y, evidences=None, cfg=None):
    """Statement-level clinical correctness of reports against image labels (+ evidence consistency)."""
    n = len(texts)
    lab = [text_to_labels(t) for t in texts]
    D = np.array([[l[c] == "definite" for c in LABELS] for l in lab])
    H = np.array([[l[c] == "hedged" for c in LABELS] for l in lab])
    N = np.array([[l[c] == "negative" for c in LABELS] for l in lab])
    P = D | H
    Y = Y.astype(bool)
    tp, fp, fn = (P & Y).sum(), (P & ~Y).sum(), (~P & Y).sum()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    macro = []
    for j in range(len(LABELS)):
        a, b, c = (P[:, j] & Y[:, j]).sum(), (P[:, j] & ~Y[:, j]).sum(), (~P[:, j] & Y[:, j]).sum()
        if Y[:, j].sum() > 0:
            macro.append(2 * a / max(1, 2 * a + b + c))
    res = dict(
        n_reports=n, words_per_report=float(np.mean([len(t.split()) for t in texts])),
        finding_statements_per_report=float(P.sum() / n),
        definite_per_report=float(D.sum() / n), hedged_per_report=float(H.sum() / n),
        precision=float(prec), recall=float(rec), micro_f1=float(2 * prec * rec / max(1e-9, prec + rec)),
        macro_f1=float(np.mean(macro)) if macro else None,
        # None (not 0) when a generator makes no positive statements at all, so silence never looks like accuracy
        hallucination_rate=float(fp / P.sum()) if P.sum() else None,          # positive statements not in the labels
        definite_hallucination_rate=float((D & ~Y).sum() / D.sum()) if D.sum() else None,  # false *assertions*
        hedged_ppv=float((H & Y).sum() / max(1, H.sum())) if H.sum() else None,
        reports_with_false_assertion=float(((D & ~Y).sum(1) > 0).mean()),
        negation_error_rate=float((N & Y).sum() / max(1, N.sum())) if N.sum() else None,
        omission_rate=float(fn / max(1, Y.sum())),
    )
    if evidences is not None:
        cnt = {t: 0 for t in ERROR_TYPES + WARNING_TYPES}
        bad_reports = 0
        for t, ev in zip(texts, evidences):
            f, i = split_report(t)
            iss = verify_report(f, i, ev, cfg)
            for x in iss:
                cnt[x["type"]] += 1
            bad_reports += bool(errors_only(iss))
        res["verifier_errors_per_100"] = {k: 100 * v / n for k, v in cnt.items()}
        res["reports_with_verifier_error"] = bad_reports / n
    return res


def eval_report(cfg, tag=None, use_llm=False):
    model, _, run_dir, dev = load_trained(cfg, tag)
    calib = load_calibration(run_dir)
    eng = EvidenceEngine(model, calib, cfg, dev)
    man = load_manifest(cfg)
    te = man[man.split == "test"].sample(n=min(cfg["eval"]["report_n"], (man.split == "test").sum()),
                                         random_state=cfg["seed"]).reset_index(drop=True)
    dl = make_loader(CXRDataset(te, cfg, False, open_cache(cfg)), 64, False, cfg["train"]["num_workers"])
    evid, logits, probs, stds = [], [], [], []
    for x, _, _ in tqdm(dl, desc="evidence", leave=False):
        out = eng.run(x, need_cams=True, cam_size=(112, 112), all_regions=True)
        evid += out["evidence"]
        logits.append(out["logits"])
        probs.append(out["prob"])
        stds.append(out["std"])
    lg, p, std = np.concatenate(logits), np.concatenate(probs), np.concatenate(stds)
    Y = te[LABELS].values
    mids = np.array([calib["thresholds"][n]["mid"] for n in LABELS])
    variants = {
        "naive_0.5": np.where(sigmoid(lg) >= 0.5, PRESENT, ABSENT),             # uncalibrated, no hedging
        "f1_definite": np.where(p >= mids, PRESENT, ABSENT),                    # calibrated F1 threshold, no hedging
        "tiered_no_mc": assign_tiers(p, std, calib, False),                     # + precision-targeted hedging
        "tiered_mc (ours)": assign_tiers(p, std, calib, True),                  # + MC-dropout downgrade
    }
    res, examples = {}, {}
    # every report is verified against OUR evidence layer (calibrated tiers + MC), so the 'verifier-flagged'
    # column shows how many statements of a baseline the evidence would not support
    ev_ours = [_variant_evidence(e, variants["tiered_mc (ours)"][k], e) for k, e in enumerate(evid)]
    fmt = lambda v: "  --  " if v is None else f"{v:.3f}"
    for name, T in variants.items():
        evs = [_variant_evidence(e, T[k], e) for k, e in enumerate(evid)]
        texts = []
        for e in evs:
            f, i = template_report(e)
            texts.append(f"FINDINGS: {f} IMPRESSION: {i}")
        res[name] = score_reports(texts, Y, ev_ours, cfg)
        examples[name] = texts[:30]
        print(f"[eval-report] {name:<18} halluc {fmt(res[name]['hallucination_rate'])}  false-assertion "
              f"{fmt(res[name]['definite_hallucination_rate'])}  micro-F1 {res[name]['micro_f1']:.3f}  "
              f"flagged {res[name]['reports_with_verifier_error']:.3f}")

    if use_llm:
        from .report import LLMGenerator, generate_verified
        llm = LLMGenerator(cfg["report"]["llm_model"], dev, cfg["report"]["llm_max_new_tokens"])
        n = min(cfg["eval"]["llm_n"], len(ev_ours))
        bs = cfg["report"]["llm_batch"]
        outs = []
        for s in tqdm(range(0, n, bs), desc="LLM reports"):
            outs += generate_verified(ev_ours[s:min(s + bs, n)], llm, cfg, verify_report, correct_report, split_report)
        drafts = [o["draft"] for o in outs]
        regen = [o["regenerated"] for o in outs]
        final = [f"FINDINGS: {o['findings']} IMPRESSION: {o['impression']}" for o in outs]
        sub = ev_ours[:n]
        res["llm_raw"] = score_reports(drafts, Y[:n], sub, cfg)
        res["llm_regenerated"] = score_reports(regen, Y[:n], sub, cfg)
        res["llm_verified (ours)"] = score_reports(final, Y[:n], sub, cfg)
        tmpl = ["FINDINGS: {} IMPRESSION: {}".format(*template_report(e)) for e in sub]
        res["template_same_subset"] = score_reports(tmpl, Y[:n], sub, cfg)
        res["llm_model"] = cfg["report"]["llm_model"]
        res["llm_mean_regenerations"] = float(np.mean([o["regen"] for o in outs]))
        res["llm_remaining_errors_after_correction"] = float(np.mean([len(o["remaining"]) > 0 for o in outs]))
        from .evidence import evidence_table
        examples["llm"] = [dict(evidence_table=evidence_table(e),
                                draft=o["draft"], final=f, log=o["log"]) for e, o, f in zip(sub[:30], outs, final)]
        # annotation sheet for an optional human (clinician) review of verifier decisions
        rows = []
        for k, (e, o) in enumerate(zip(sub, outs)):
            f, i = split_report(o["draft"])
            for iss in errors_only(verify_report(f, i, e, cfg)):
                rows.append(dict(image=te.image[k], sentence=iss["sentence"], flag=iss["type"], detail=iss["detail"],
                                 annotator_agrees="", comment=""))
        pd.DataFrame(rows).to_csv(os.path.join(results_dir(cfg), "verifier_annotation_sheet.csv"), index=False)
        for name in ("llm_raw", "llm_verified (ours)"):
            print(f"[eval-report] {name:<18} halluc {fmt(res[name]['hallucination_rate'])}  verifier-flagged reports "
                  f"{res[name]['reports_with_verifier_error']:.3f}")

    res["n_images"] = len(te)
    _dump(res, os.path.join(results_dir(cfg), "report_metrics.json"))
    _dump(examples, os.path.join(results_dir(cfg), "report_examples.json"))
    with open(os.path.join(results_dir(cfg), "report_evidence_sample.json"), "w") as f:
        json.dump(ev_ours[:500], f)
    return res


# ============================================================================= verifier stress test
PARAPHRASE = {
    "UNSUPPORTED_FINDING": ["There is {art}{loc}{name}.", "{Name} is noted in the {side} lung.",
                            "Findings are consistent with {name}."],
    "UNSUPPORTED_DEVICE": ["An endotracheal tube is in satisfactory position.", "A right-sided central venous catheter "
                           "terminates in the SVC.", "Cardiac pacemaker leads are noted."],
    "PRIOR_REFERENCE": ["Unchanged compared with the prior study.", "Stable appearance since the previous radiograph.",
                        "Interval improvement in aeration."],
    "UNSUPPORTED_MEASUREMENT": ["The transverse cardiac diameter measures 15 cm.",
                                "A 12 mm opacity is projected over the left apex."],
    "UNVERIFIABLE_FINDING": ["There is an old healed rib fracture.", "A calcified granuloma is seen.",
                             "Mild degenerative changes of the thoracic spine."],
    "GLOBAL_CONTRADICTION": ["No acute cardiopulmonary abnormality.", "The lungs are clear."],
}


def _inject(f, i, ev, kind, rng):
    """Return (findings, impression, expected_label) with one injected error, or None if not applicable."""
    fs, is_ = split_sentences(f), split_sentences(i)
    present = [n for n, d in ev.items() if d["tier"] == "present"]
    possible = [n for n, d in ev.items() if d["tier"] == "possible"]
    absent = [n for n, d in ev.items() if d["tier"] == "absent"]

    def sent_idx(sents, n):
        from .verifier import parse_sentence
        return [k for k, s in enumerate(sents) if any(m["label"] == n and not m["negated"] for m in parse_sentence(s))]

    if kind == "UNSUPPORTED_NEGATIVE":
        if not absent:
            return None
        n = rng.choice(absent)
        fs.append(f"No {NAME[n]}.")
        return " ".join(fs), " ".join(is_), n
    if kind == "UNSUPPORTED_FINDING":
        n = rng.choice(absent)
        side = rng.choice(["right", "left"])
        tpl = rng.choice(PARAPHRASE[kind])
        loc = f"{side} " if n in LATERAL else ""
        s = tpl.format(art="a " if n in ("Effusion", "Mass", "Nodule", "Pneumothorax", "Hernia") else "", loc=loc,
                       name=NAME[n], Name=NAME[n].capitalize(), side=side)
        if "{side}" in tpl and n not in LATERAL:
            s = f"Findings are consistent with {NAME[n]}."
        fs.insert(rng.randint(0, len(fs) + 1), s)
        return " ".join(fs), " ".join(is_), n
    if kind == "LATERALITY":
        cand = [n for n in present + possible if n in LATERAL and ev[n]["side"] in ("left", "right")]
        if not cand:
            return None
        n = rng.choice(cand)
        ks = sent_idx(fs, n)
        if not ks:
            return None
        sw = {"left": "right", "right": "left"}
        fs[ks[0]] = fs[ks[0]].replace(ev[n]["side"], sw[ev[n]["side"]], 1)
        return " ".join(fs), " ".join(is_), n
    if kind == "OVERCONFIDENT":
        if not possible:
            return None
        n = rng.choice(possible)
        ks = sent_idx(fs, n)
        if not ks:
            return None
        fs[ks[0]] = fs[ks[0]].replace("possible ", "")
        return " ".join(fs), " ".join(is_), n
    if kind == "CONTRADICTION":
        if not present:
            return None
        n = rng.choice(present)
        neg = "Heart size is within normal limits." if n == "Cardiomegaly" else f"No {NAME[n]}."
        fs = [s for k, s in enumerate(fs) if k not in sent_idx(fs, n)] + [neg]
        is_ = [s for k, s in enumerate(is_) if k not in sent_idx(is_, n)] or ["No acute findings."]
        return " ".join(fs), " ".join(is_), n
    if kind == "OMISSION":
        if not present:
            return None
        n = rng.choice(present)
        fs = [s for k, s in enumerate(fs) if k not in sent_idx(fs, n)]
        is_ = [s for k, s in enumerate(is_) if k not in sent_idx(is_, n)]
        return " ".join(fs), " ".join(is_), n
    if kind == "UNSUPPORTED_SEVERITY":
        cand = present + possible
        if not cand:
            return None
        n = rng.choice(cand)
        ks = sent_idx(fs, n)
        if not ks:
            return None
        word = rng.choice(["large", "moderate", "small", "severe"])
        s = fs[ks[0]]
        target = NAME[n].split()[0] if n != "Cardiomegaly" else "enlarged"
        if target not in s:
            return None
        fs[ks[0]] = s.replace(target, f"{word} {target}", 1)
        return " ".join(fs), " ".join(is_), n
    if kind == "GLOBAL_CONTRADICTION":
        if not present:
            return None
        is_.append(rng.choice(PARAPHRASE[kind]))
        return " ".join(fs), " ".join(is_), None
    # appended out-of-scope sentences
    fs.append(rng.choice(PARAPHRASE[kind]))
    return " ".join(fs), " ".join(is_), None


def stress_test(cfg, evidences=None, n=None):
    """Inject one error of each type into evidence-consistent reports and measure detection and repair."""
    if evidences is None:
        p = os.path.join(results_dir(cfg), "report_evidence_sample.json")
        if not os.path.exists(p):
            raise FileNotFoundError("run `python run.py eval-report` first (it saves the evidence sample)")
        evidences = json.load(open(p))
    if n:
        evidences = evidences[:n]
    rng = random.Random(cfg["seed"])
    kinds = ["UNSUPPORTED_FINDING", "LATERALITY", "OVERCONFIDENT", "CONTRADICTION", "OMISSION", "UNSUPPORTED_SEVERITY",
             "GLOBAL_CONTRADICTION", "PRIOR_REFERENCE", "UNSUPPORTED_DEVICE", "UNSUPPORTED_MEASUREMENT",
             "UNVERIFIABLE_FINDING", "UNSUPPORTED_NEGATIVE"]
    stats = {k: dict(n=0, detected=0, other_flags=0, repaired=0, faithful=0) for k in kinds}
    clean_false_alarm = 0
    for ev in evidences:
        f, i = template_report(ev)
        clean_false_alarm += bool(errors_only(verify_report(f, i, ev, cfg)))
        clean_labels = text_to_labels(f"{f} IMPRESSION: {i}")
        for k in kinds:
            inj = _inject(f, i, ev, k, rng)
            if inj is None:
                continue
            F, I, lab = inj
            iss = errors_only(verify_report(F, I, ev, cfg))
            hit = any(x["type"] == k and (lab is None or x["label"] == lab) for x in iss)
            s = stats[k]
            s["n"] += 1
            s["detected"] += hit
            s["other_flags"] += any(x["type"] != k for x in iss)
            fixed = correct_report(F, I, ev, cfg)
            s["repaired"] += not fixed["remaining"]
            s["faithful"] += text_to_labels(f"{fixed['findings']} IMPRESSION: {fixed['impression']}") == clean_labels
    out = {"n_reports": len(evidences), "clean_false_alarm_rate": clean_false_alarm / max(1, len(evidences)),
           "per_type": {}}
    for k, s in stats.items():
        if s["n"]:
            out["per_type"][k] = dict(n=s["n"], detection_recall=s["detected"] / s["n"],
                                      collateral_flags=s["other_flags"] / s["n"], repair_rate=s["repaired"] / s["n"],
                                      label_faithful_after_repair=s["faithful"] / s["n"])
    tot = sum(s["n"] for s in stats.values())
    out["overall"] = dict(n=tot, detection_recall=sum(s["detected"] for s in stats.values()) / max(1, tot),
                          repair_rate=sum(s["repaired"] for s in stats.values()) / max(1, tot),
                          label_faithful_after_repair=sum(s["faithful"] for s in stats.values()) / max(1, tot))
    _dump(out, os.path.join(results_dir(cfg), "stress_test.json"))
    print(f"[stress-test] {tot} injected errors: detection {out['overall']['detection_recall']:.3f}, repair "
          f"{out['overall']['repair_rate']:.3f}, clean false-alarm {out['clean_false_alarm_rate']:.3f}")
    return out
