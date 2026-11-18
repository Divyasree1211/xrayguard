"""Gradio demo: X-ray -> verified report + evidence panel + heat-map + verifier log."""
import pandas as pd

from .pipeline import XRayGuard


def launch(cfg, tag=None, use_llm=False, share=True):
    import gradio as gr
    pipe = XRayGuard(cfg, tag, llm=use_llm)
    mark = {"present": "✓", "possible": "?", "absent": "✗"}

    def predict(img):
        if img is None:
            return "", None, None, ""
        r = pipe.analyze(img)
        rows = [{"Finding": k.replace("_", " "), "Probability": f"{v['prob'] * 100:.0f}%",
                 "Uncertainty (±)": f"{v['std'] * 100:.1f}%", "Confidence": v["confidence"],
                 "Location": v["region"] or "-", "": mark[v["tier"]]}
                for k, v in sorted(r["evidence"].items(), key=lambda kv: -kv[1]["prob"])]
        report = f"FINDINGS:\n{r['findings']}\n\nIMPRESSION:\n{r['impression']}\n\n[generator: {r['generator']}]"
        flags = [f"{i['type']}: {i['detail']}" for i in r["issues_first"] if not i["type"].startswith("OMISSION_P")]
        log = "Verifier flags on first draft:\n" + ("\n".join(flags) or "none") + "\n\nCorrections:\n" + \
              ("\n".join(r["log"]) or "none")
        return report, pd.DataFrame(rows), r["overlay"], log

    gr.Interface(
        fn=predict, inputs=gr.Image(type="numpy", label="Chest X-ray (frontal PA/AP)"),
        outputs=[gr.Textbox(label="Verified report", lines=9), gr.Dataframe(label="Evidence panel"),
                 gr.Image(label="Localisation heat-map (present/possible findings)"),
                 gr.Textbox(label="Clinical consistency verifier", lines=9)],
        title="XRayGuard - evidence-grounded, uncertainty-aware chest X-ray reporting",
        description="Research prototype trained on NIH ChestX-ray14. NOT for clinical use.",
    ).launch(share=share)
