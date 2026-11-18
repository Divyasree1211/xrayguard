"""External (cross-dataset) evaluation on the Indiana University chest X-ray collection (IU X-Ray / OpenI).

NIH ChestX-ray14 has no free-text reports, so reference-based text metrics (BLEU, ROUGE-L, METEOR, BERTScore) can only be
computed on a dataset that has them. IU X-Ray is public (Kaggle: raddar/chest-xrays-indiana-university) and its images were
never seen in training, so it also measures generalisation to another hospital. Reference labels are extracted from the
reference reports with the same rule-based labeller, so they are weak labels.
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import LABELS
from .data import eval_transform
from .evaluate import _dump, results_dir, score_reports
from .evidence import EvidenceEngine, load_calibration
from .metrics import per_class_auroc
from .pipeline import load_trained
from .report import template_report
from .verifier import text_to_labels


def _find(root, name):
    h = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not h:
        raise FileNotFoundError(f"{name} not found under {root} (set iu.root)")
    return h[0]


def load_iu(root, n=None):
    rep = pd.read_csv(_find(root, "indiana_reports.csv"))
    proj = pd.read_csv(_find(root, "indiana_projections.csv"))
    proj = proj[proj.projection.str.lower().str.startswith("frontal")].drop_duplicates("uid")
    df = proj.merge(rep, on="uid")
    df["reference"] = (df.findings.fillna("") + " " + df.impression.fillna("")).str.strip()
    df = df[df.reference.str.len() > 10].reset_index(drop=True)
    files = {os.path.basename(p): p for p in glob.glob(os.path.join(root, "**", "*.png"), recursive=True)}
    df["path"] = df.filename.map(files)
    df = df[df.path.notna()].reset_index(drop=True)
    return df.head(n) if n else df


def text_metrics(hyps, refs):
    out = {}
    try:
        import nltk
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
        th = [h.lower().split() for h in hyps]
        tr = [[r.lower().split()] for r in refs]
        sm = SmoothingFunction().method1
        for k in (1, 2, 3, 4):
            out[f"BLEU-{k}"] = corpus_bleu(tr, th, weights=tuple([1 / k] * k + [0] * (4 - k)), smoothing_function=sm)
        try:
            from nltk.translate.meteor_score import meteor_score
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            out["METEOR"] = float(np.mean([meteor_score([r.lower().split()], h.lower().split())
                                           for h, r in zip(hyps, refs)]))
        except Exception as e:
            print("METEOR skipped:", e)
    except ImportError:
        print("nltk not installed - BLEU/METEOR skipped")
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        out["ROUGE-L"] = float(np.mean([sc.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]))
    except ImportError:
        print("rouge-score not installed - ROUGE-L skipped")
    try:
        from bert_score import score as bscore
        out["BERTScore-F1"] = float(bscore(hyps, refs, lang="en", verbose=False)[2].mean())
    except Exception as e:
        print("BERTScore skipped:", e)
    return out


def eval_iu(cfg, tag=None, use_llm=False):
    model, _, run_dir, dev = load_trained(cfg, tag)
    calib = load_calibration(run_dir)
    eng = EvidenceEngine(model, calib, cfg, dev)
    df = load_iu(cfg["iu"]["root"], cfg["iu"].get("n"))
    S, C = cfg["train"]["img_size"], cfg["data"]["cache_size"]
    evid, probs = [], []
    bs = 32
    for s in tqdm(range(0, len(df), bs), desc="IU X-Ray"):
        xs = []
        for p in df.path[s:s + bs]:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(g, (C, C), interpolation=cv2.INTER_AREA)
            xs.append(eval_transform(g, S))
        out = eng.run(torch.from_numpy(np.stack(xs))[:, None], cam_size=(112, 112))
        evid += out["evidence"]
        probs.append(out["prob"])
    P = np.concatenate(probs)
    ref_lab = [text_to_labels(r) for r in df.reference]
    Y = np.array([[l[c] in ("definite", "hedged") for c in LABELS] for l in ref_lab]).astype(int)
    hyps = [" ".join(template_report(e)) for e in evid]
    res = {"n": len(df), "note": "reference labels extracted from IU reports with the rule-based labeller",
           "classifier_auroc_vs_report_labels": per_class_auroc(P, Y)}
    a = res["classifier_auroc_vs_report_labels"]
    res["classifier_mean_auroc"] = float(np.mean(list(a.values()))) if a else None
    res["template"] = dict(**text_metrics(hyps, list(df.reference)), **score_reports(hyps, Y, evid, cfg))
    if use_llm:
        from .report import LLMGenerator, generate_verified
        from .verifier import correct_report, split_report, verify_report
        llm = LLMGenerator(cfg["report"]["llm_model"], dev, cfg["report"]["llm_max_new_tokens"])
        outs = []
        for s in tqdm(range(0, len(evid), cfg["report"]["llm_batch"]), desc="LLM (IU)"):
            outs += generate_verified(evid[s:s + cfg["report"]["llm_batch"]], llm, cfg, verify_report, correct_report,
                                      split_report)
        raw = [o["draft"] for o in outs]
        fin = [f"{o['findings']} {o['impression']}" for o in outs]
        res["llm_raw"] = dict(**text_metrics(raw, list(df.reference)), **score_reports(raw, Y, evid, cfg))
        res["llm_verified"] = dict(**text_metrics(fin, list(df.reference)), **score_reports(fin, Y, evid, cfg))
    _dump(res, os.path.join(results_dir(cfg), "iu_metrics.json"))
    return res
