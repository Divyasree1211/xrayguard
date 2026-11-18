"""Serve the standalone XRayGuard webpage and its local inference endpoint."""
import argparse
import base64
import csv
import hashlib
import json
import math
from io import BytesIO
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request, send_file


UPLOAD_LIMIT = 10 * 1024 * 1024
PROJECT = Path(__file__).resolve().parent


def create_app(work_dir, tag=None, config=None):
    app = Flask(__name__, static_folder=str(PROJECT / "web"), static_url_path="/assets")
    app.config["MAX_CONTENT_LENGTH"] = UPLOAD_LIMIT
    inference_lock = Lock()

    def run_name():
        marker = work_dir / "active_run.txt"
        return tag or (marker.read_text(encoding="utf-8").strip() if marker.is_file() else "")

    @lru_cache(maxsize=1)
    def load_pipeline(name):
        from xrayguard.config import load_config
        from xrayguard.pipeline import XRayGuard

        cfg = load_config(config, [f"data.work_dir={json.dumps(str(work_dir))}"])
        return XRayGuard(cfg, tag=name)

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/results")
    def results_page():
        return app.send_static_file("results.html")

    @app.get("/about")
    def about_page():
        return app.send_static_file("about.html")

    @app.get("/api/status")
    def status():
        name = run_name()
        ready = bool(name) and all((work_dir / "runs" / name / f).is_file() for f in ("best.pt", "calib.json"))
        metadata_path = work_dir / "runs" / name / "config.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        return jsonify(model_run=name, files_ready=ready, generator="Evidence-constrained template",
                       clinical_validation="Not established", backbone=metadata.get("model", {}).get("backbone"),
                       uncertainty_samples=metadata.get("calib", {}).get("mc_samples"), findings_count=14)

    def demo_catalog():
        path = PROJECT / "demo_samples/catalog.json"
        return json.loads(path.read_text()) if path.is_file() else []

    @app.get("/api/examples")
    def examples():
        return jsonify(examples=[{k: v for k, v in row.items() if k != "file"} for row in demo_catalog()])

    @app.get("/api/examples/<example_id>/image")
    def example_image(example_id):
        example = next((row for row in demo_catalog() if row["id"] == example_id), None)
        if example is None:
            return jsonify(error="Example not found."), 404
        folder = (PROJECT / "demo_samples").resolve()
        path = (folder / example["file"]).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            return jsonify(error="Example image unavailable."), 404
        return send_file(path, mimetype="image/png")

    @app.get("/api/results")
    def results():
        name = run_name()
        directory = work_dir / "runs" / name / "evaluation"
        path = directory / "cls_metrics.json"
        if not path.is_file():
            return jsonify(error="Evaluation results for this model are unavailable."), 404
        metrics = json.loads(path.read_text())
        if metrics.get("run") != name:
            return jsonify(error="Evaluation model does not match the active model."), 409
        def read_optional(path):
            return json.loads(path.read_text()) if path.is_file() else None
        def finite(value):
            if isinstance(value, dict):
                return {k: finite(v) for k, v in value.items()}
            if isinstance(value, list):
                return [finite(v) for v in value]
            return None if isinstance(value, float) and not math.isfinite(value) else value
        history_path = work_dir / "runs" / name / "history.csv"
        history = []
        if history_path.is_file():
            with history_path.open(newline="") as history_file:
                history = [{key: float(row[key]) for key in ("epoch", "val_mean_auroc", "train_loss", "val_bce", "minutes")}
                           for row in csv.DictReader(history_file)]
        return jsonify(finite(dict(model_run=name, classification=metrics,
            original_report_metrics=read_optional(directory / "report_metrics.json"),
            current_rule_stress_test=read_optional(work_dir / "review/results/stress_test.json"),
            selection=read_optional(work_dir / "selection_summary.json"),
            training_history=history,
            report_metrics_version="Original exported report generator, before the negative-wording update",
            expert_review_completed=False, localisation_evaluation_completed=(directory / "loc_metrics.json").is_file())))

    @app.post("/api/verify")
    def verify_text():
        if request.content_length and request.content_length > 20000:
            return jsonify(error="Keep the wording check below 20 KB."), 413
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Provide report text and evidence from an analysis."), 400
        findings, impression, evidence = (payload.get(k) for k in ("findings", "impression", "evidence"))
        from xrayguard.config import LABELS
        if (not isinstance(findings, str) or not isinstance(impression, str) or
                len(findings) + len(impression) > 10000 or not isinstance(evidence, dict) or set(evidence) != set(LABELS)):
            return jsonify(error="Provide report text and all 14 finding records from an analysis."), 400
        for values in evidence.values():
            if not isinstance(values, dict) or values.get("tier") not in ("present", "possible", "absent"):
                return jsonify(error="Invalid finding evidence."), 400
            for key in ("prob", "std"):
                value = values.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key == "prob" and value > 1):
                    return jsonify(error="Invalid finding score."), 400
            if values.get("side") not in (None, "left", "right", "bilateral") or values.get("zone") not in (None, "upper", "mid", "lower"):
                return jsonify(error="Invalid finding location."), 400
        from xrayguard.verifier import correct_report, errors_only
        corrected = correct_report(findings, impression, evidence)
        return jsonify(issues=errors_only(corrected["issues_before"]), findings=corrected["findings"],
                       impression=corrected["impression"], corrections=corrected["log"], remaining=corrected["remaining"],
                       scope="Checks wording against supplied model evidence, not clinical truth.")

    @app.errorhandler(413)
    def upload_too_large(_error):
        return jsonify(error="Choose an image no larger than 10 MB."), 413

    @app.post("/api/report")
    def report():
        started = time.perf_counter()
        if request.mimetype not in {"image/png", "image/jpeg"}:
            return jsonify(error="Choose a PNG or JPEG image."), 415

        raw = request.get_data(cache=False)
        if not (raw.startswith(b"\x89PNG\r\n\x1a\n") or raw.startswith(b"\xff\xd8\xff")):
            return jsonify(error="This file could not be read as a PNG or JPEG image."), 400

        name = run_name()
        folder = work_dir / "runs" / name
        if not name or any(not (folder / file).is_file() for file in ("best.pt", "calib.json")):
            return jsonify(error="The trained model is unavailable. Add best.pt and calib.json from the same NIH run and select that run when starting the website."), 503

        try:
            from PIL import Image, UnidentifiedImageError
            try:
                with Image.open(BytesIO(raw)) as header:
                    if header.width * header.height > 4_194_304:
                        return jsonify(error="Choose an image with no more than 4 megapixels (for example, 2048 by 2048)."), 400
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                return jsonify(error="This image could not be opened. Try exporting it again as PNG or JPEG."), 400
            import cv2
            import numpy as np

            cv2.setNumThreads(1)
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return jsonify(error="This image could not be opened. Try exporting it again as PNG or JPEG."), 400

            with inference_lock:
                pipeline = load_pipeline(name)
                # Keep the MC estimate repeatable for identical decoded pixels, while
                # restoring the process RNG state after the request.
                import torch
                sampling_seed = int.from_bytes(hashlib.sha256(image.tobytes() + str(image.shape).encode()).digest()[:4], "big")
                rng_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=rng_devices):
                    torch.manual_seed(sampling_seed)
                    result = pipeline.analyze(image, include_visuals=True)
                evidence = [
                    dict(finding=finding, **values,
                         report_threshold=pipeline.calib["thresholds"][finding]["mid"])
                    for finding, values in sorted(
                        result["evidence"].items(), key=lambda item: -item[1]["prob"]
                    )
                ]

            shown = [row["finding"] for row in evidence if row["tier"] != "absent"]
            def encode_image(rgb):
                success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if not success:
                    raise RuntimeError("Could not encode an analysis image")
                return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")

            heatmap = None
            if shown:
                heatmap = encode_image(result["overlay"])
            finding_heatmaps = {name: encode_image(rgb) for name, rgb in result["finding_overlays"].items()}
            processed_image = encode_image(result["processed_image"])
            from xrayguard.verifier import errors_only, verify_report
            rule_errors = errors_only(verify_report(result["findings"], result["impression"], result["evidence"], pipeline.cfg))
        except ImportError:
            app.logger.exception("Could not import the NIH inference dependencies")
            return jsonify(error="The model dependencies are unavailable. Install the packages in requirements.txt."), 503
        except Exception:
            app.logger.exception("NIH inference failed")
            return jsonify(error="Report generation failed. Check the server log for details."), 500

        return jsonify(
            findings=result["findings"], impression=result["impression"],
            evidence=evidence, heatmap=heatmap, heatmap_findings=shown,
            uncertainty_samples=pipeline.cfg["calib"]["mc_samples"], model_run=name,
            finding_heatmaps=finding_heatmaps, processed_image=processed_image,
            generated_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.perf_counter() - started, 2),
            generator=result.get("generator", "template"), verification=dict(
                status="consistent" if not rule_errors else "review_required", issues=rule_errors,
                corrections=result.get("log", []), scope="Report wording checked against model evidence. Clinical correctness is not established."),
            sampling_seed=sampling_seed,
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=PROJECT / "work")
    parser.add_argument("--tag", help="Run folder name under the work directory's runs folder")
    parser.add_argument("--config", help="YAML config used by the trained run")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    create_app(args.work_dir.resolve(), args.tag, args.config).run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
