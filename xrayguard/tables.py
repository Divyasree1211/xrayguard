"""Turn results/*.json into paper tables (LaTeX + Markdown) and figures. Writes into <work_dir>/paper_assets/."""
import glob
import json
import os

import numpy as np
import pandas as pd

from .config import LABELS, wpath


def _j(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _f(x, d=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _md(df):
    import re
    s = df.to_markdown(index=False)
    for a, b in (("$\\downarrow$", "↓"), ("$\\geq$", "≥"), ("$<$", "<"), ("\\%", "%"), ("\\#", "#")):
        s = s.replace(a, b)
    return re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", s)


def _latex(df, caption, label, path, col_fmt=None, wide=False):
    body = df.to_latex(index=False, escape=False, column_format=col_fmt or "l" + "c" * (len(df.columns) - 1))
    env = "table*" if wide else "table"
    with open(path, "w") as f:
        width = "\\textwidth" if wide else "\\columnwidth"
        f.write(f"\\begin{{{env}}}[t]\n\\centering\n\\small\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
                f"\\begin{{adjustbox}}{{max width={width}}}\n" + body + f"\\end{{adjustbox}}\n\\end{{{env}}}\n")


def build(cfg):
    R = wpath(cfg, "results")
    out = wpath(cfg, "paper_assets")
    os.makedirs(out, exist_ok=True)
    md = []

    cls = _j(os.path.join(R, "cls_metrics.json"))
    if cls:
        rows = []
        for n in LABELS:
            v = cls["per_class"][n]
            rows.append({"Finding": n.replace("_", " "), "Prev.": _f(v["prevalence"] * 100, 1) + "\\%",
                         "AUROC (95\\% CI)": f"{_f(v['auroc'])} ({_f(v['auroc_ci'][0])}--{_f(v['auroc_ci'][1])})",
                         "AUPRC": _f(v["auprc"]), "F1": _f(v["f1"])})
        rows.append({"Finding": "\\textbf{Mean}", "Prev.": "",
                     "AUROC (95\\% CI)": f"\\textbf{{{_f(cls['mean_auroc'])}}} ({_f(cls['mean_auroc_ci'][0])}--"
                                         f"{_f(cls['mean_auroc_ci'][1])})", "AUPRC": "", "F1": _f(cls["macro_f1_at_mid"])})
        t = pd.DataFrame(rows)
        _latex(t, f"Per-finding detection on the official NIH ChestX-ray14 test split "
                  f"({cls['n_images']:,} images, {cls['n_patients']:,} patients). CIs: patient-level bootstrap.",
               "tab:cls", os.path.join(out, "tab_cls.tex"))
        md.append("## Detection\n" + _md(t))

        c = cls["calibration"]
        tr = cls["tiers"]
        t2 = pd.DataFrame([
            {"Quantity": "ECE (pooled)", "Uncalibrated": _f(c["ece_raw"], 4), "Calibrated": _f(c["ece_T"], 4)},
            {"Quantity": "ECE (class-wise mean)", "Uncalibrated": _f(c["classwise_ece_raw"], 4),
             "Calibrated": _f(c["classwise_ece_T"], 4)},
            {"Quantity": "Brier score", "Uncalibrated": _f(c["brier_raw"], 4), "Calibrated": _f(c["brier_T"], 4)}])
        _latex(t2, "Calibration on the test split before and after per-finding temperature scaling.", "tab:calib",
               os.path.join(out, "tab_calib.tex"))
        t3 = pd.DataFrame([{"Decision rule": k.replace("_", " "), "PPV(present)": _f(v["ppv_present"]),
                            "\\#present": v["n_present"], "PPV(possible)": _f(v["ppv_possible"]),
                            "\\#possible": v["n_possible"], "NPV(absent)": _f(v["npv_absent"], 4)}
                           for k, v in tr.items()])
        _latex(t3, "Reliability of the three statement tiers on the test split (pooled over findings).", "tab:tiers",
               os.path.join(out, "tab_tiers.tex"))
        sg = pd.DataFrame([{"Subgroup": k.replace(">=", "$\\geq$").replace("<", "$<$"), "n": v["n"],
                            "Mean AUROC (95\\% CI)": f"{_f(v['mean_auroc'])} ({_f(v['ci'][0])}--{_f(v['ci'][1])})"}
                           for k, v in cls["subgroups"].items()])
        _latex(sg, "Subgroup performance (sex, age, view position).", "tab:subgroups",
               os.path.join(out, "tab_subgroups.tex"))
        md += ["## Calibration\n" + _md(t2), "## Tiers\n" + _md(t3),
               "## Subgroups\n" + _md(sg)]

    loc = _j(os.path.join(R, "loc_metrics.json"))
    if loc:
        rows = []
        for k, v in list(loc["per_class"].items()) + [("\\textbf{All}", loc["overall"])]:
            rows.append({"Finding": k.replace("_", " "), "n": v["n"], "Pointing": _f(v["pointing"]),
                         "Centre base.": _f(v["centre_baseline"]), "IoU": _f(v["iou"]), "Acc@IoBB$\\geq$0.25":
                             _f(v["acc_iobb_025"]), "Side agr.": _f(v["side_agreement"]), "Zone agr.": _f(v["zone_agreement"])})
        t = pd.DataFrame(rows)
        _latex(t, "Localisation against NIH radiologist boxes (images outside the training split).", "tab:loc",
               os.path.join(out, "tab_loc.tex"), wide=True)
        md.append("## Localisation\n" + _md(t))

    rep = _j(os.path.join(R, "report_metrics.json"))
    if rep:
        rows = []
        for k, v in rep.items():
            if not isinstance(v, dict) or "micro_f1" not in v:
                continue
            rows.append({"Generator": k.replace("_", " "), "Halluc.$\\downarrow$": _f(v["hallucination_rate"]),
                         "False assert.$\\downarrow$": _f(v["definite_hallucination_rate"]),
                         "Reports w/ false assert.$\\downarrow$": _f(v["reports_with_false_assertion"]),
                         "Omission$\\downarrow$": _f(v["omission_rate"]), "Micro-F1": _f(v["micro_f1"]),
                         "Verifier-flagged$\\downarrow$": _f(v.get("reports_with_verifier_error"))})
        t = pd.DataFrame(rows)
        _latex(t, f"Report-level clinical correctness against NIH labels ({rep.get('n_images')} test images; LLM rows "
                  "on a subset).", "tab:report", os.path.join(out, "tab_report.tex"), wide=True)
        md.append("## Reports\n" + _md(t))

    st = _j(os.path.join(R, "stress_test.json"))
    if st:
        rows = [{"Injected error": k.replace("_", " ").lower(), "n": v["n"], "Detection": _f(v["detection_recall"]),
                 "Repair": _f(v["repair_rate"]), "Faithful": _f(v["label_faithful_after_repair"])}
                for k, v in st["per_type"].items()]
        rows.append({"Injected error": "\\textbf{overall}", "n": st["overall"]["n"],
                     "Detection": _f(st["overall"]["detection_recall"]), "Repair": _f(st["overall"]["repair_rate"]),
                     "Faithful": _f(st["overall"]["label_faithful_after_repair"])})
        t = pd.DataFrame(rows)
        _latex(t, f"Verifier stress test: one injected error per report ({st['n_reports']} base reports; false-alarm "
                  f"rate on clean reports {_f(st['clean_false_alarm_rate'])}).", "tab:stress",
               os.path.join(out, "tab_stress.tex"))
        md.append("## Stress test\n" + _md(t))

    iu = _j(os.path.join(R, "iu_metrics.json"))
    if iu:
        rows = []
        for k in ("template", "llm_raw", "llm_verified"):
            if k in iu:
                v = iu[k]
                rows.append({"Generator": k.replace("_", " "), **{m: _f(v.get(m)) for m in
                             ("BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "BERTScore-F1")},
                             "Label micro-F1": _f(v["micro_f1"]), "Halluc.": _f(v["hallucination_rate"])})
        t = pd.DataFrame(rows)
        _latex(t, f"External evaluation on IU X-Ray ({iu['n']} frontal studies, unseen hospital). Classifier mean AUROC "
                  f"against report-derived labels: {_f(iu.get('classifier_mean_auroc'))}.", "tab:iu",
               os.path.join(out, "tab_iu.tex"), wide=True)
        md.append("## IU X-Ray\n" + _md(t))

    abl = sorted(glob.glob(wpath(cfg, "runs/*/history.csv")))
    if len(abl) > 1:
        rows = []
        for h in abl:
            d = pd.read_csv(h)
            rows.append({"Run": os.path.basename(os.path.dirname(h)).replace("_", " "), "Epochs": len(d),
                         "Best val AUROC": _f(d.val_mean_auroc.max())})
        t = pd.DataFrame(rows)
        _latex(t, "Backbone / loss ablation (validation mean AUROC).", "tab:ablation", os.path.join(out, "tab_ablation.tex"))
        md.append("## Ablation (validation)\n" + _md(t))

    with open(os.path.join(out, "results_summary.md"), "w") as f:
        f.write("\n\n".join(md))
    figures(cfg, out, cls)
    print(f"[tables] wrote LaTeX tables, figures and results_summary.md to {out}")


def figures(cfg, out, cls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 200})
    if cls:
        fig, ax = plt.subplots(figsize=(3.3, 3.1))
        for key, lab, mk in (("reliability_raw", "uncalibrated", "o"), ("reliability_T", "temperature-scaled", "s")):
            r = np.array(cls["calibration"][key])
            ax.plot(r[:, 0], r[:, 1], marker=mk, ms=3, label=lab)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlabel="predicted probability", ylabel="observed frequency", xlim=(0, 1), ylim=(0, 1))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_reliability.pdf"))
        fig.savefig(os.path.join(out, "fig_reliability.png"))
        plt.close(fig)

        s = cls["selective"]
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        ax.plot(s["coverage"], s["mc_std"], "o-", ms=3, label="reject by MC-dropout std")
        ax.plot(s["coverage"], s["random"], "s--", ms=3, label="random rejection")
        ax.invert_xaxis()
        ax.set(xlabel="coverage", ylabel="mean AUROC on retained")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_selective.pdf"))
        fig.savefig(os.path.join(out, "fig_selective.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.8, 2.4))
        names = LABELS
        a = [cls["per_class"][n]["auroc"] for n in names]
        lo = [a[k] - cls["per_class"][n]["auroc_ci"][0] for k, n in enumerate(names)]
        hi = [cls["per_class"][n]["auroc_ci"][1] - a[k] for k, n in enumerate(names)]
        ax.bar(range(len(names)), a, yerr=[lo, hi], capsize=2, color="#4C72B0")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=40, ha="right")
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("test AUROC")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_auroc.pdf"))
        fig.savefig(os.path.join(out, "fig_auroc.png"))
        plt.close(fig)

    hist = sorted(glob.glob(wpath(cfg, "runs/*/history.csv")))
    if hist:
        fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
        for h in hist:
            d = pd.read_csv(h)
            name = os.path.basename(os.path.dirname(h))
            ax[0].plot(d.epoch, d.train_loss, label=f"{name} train")
            ax[0].plot(d.epoch, d.val_bce, "--", label=f"{name} val")
            ax[1].plot(d.epoch, d.val_mean_auroc, "o-", ms=3, label=name)
        ax[0].set(xlabel="epoch", ylabel="loss")
        ax[1].set(xlabel="epoch", ylabel="val mean AUROC")
        ax[1].legend(frameon=False, fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_training.pdf"))
        fig.savefig(os.path.join(out, "fig_training.png"))
        plt.close(fig)

    ex = sorted(glob.glob(wpath(cfg, "results/cam_examples/*.png")))[:8]
    if ex:
        import cv2
        k = len(ex)
        fig, axs = plt.subplots(1, k, figsize=(1.7 * k, 1.9))
        axs = np.atleast_1d(axs)
        for a, p in zip(axs, ex):
            a.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
            a.set_title(os.path.basename(p)[3:-4].replace("_", " "), fontsize=7)
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_cam_examples.pdf"))
        fig.savefig(os.path.join(out, "fig_cam_examples.png"))
        plt.close(fig)
