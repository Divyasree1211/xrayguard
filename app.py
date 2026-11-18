"""Portable WSGI entry point shared by Docker and local smoke checks."""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from website import create_app

ROOT = Path(__file__).resolve().parent
app = create_app(
    ROOT / "work_nih_10000",
    tag="nih10000_bce_gpu",
    config=str(ROOT / "inference.yaml"),
)


@app.get("/healthz")
def health():
    """Readiness of packaged assets; actual inference is tested separately."""
    run = ROOT / "work_nih_10000/runs/nih10000_bce_gpu"
    ready = all((run / name).is_file() for name in ("best.pt", "calib.json"))
    return {"status": "ok" if ready else "missing_model"}, 200 if ready else 503


@app.after_request
def headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if response.mimetype == "application/json":
        response.headers["Cache-Control"] = "no-store"
    return response
