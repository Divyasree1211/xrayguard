"""Exercise packaged routes and real model inference without any training data."""
import io
import json
import sys
from pathlib import Path

from PIL import Image
from app import app


def main():
    client = app.test_client()
    checked = []
    for route in ("/", "/results", "/about", "/assets/theme.css", "/assets/report.js", "/healthz", "/api/status", "/api/results", "/api/examples"):
        response = client.get(route)
        assert response.status_code == 200, (route, response.status_code)
        checked.append(route)
    assert client.post("/api/report", data=b"bad", content_type="image/png").status_code == 400
    assert client.post("/api/report", data=b"bad", content_type="text/plain").status_code == 415
    large = io.BytesIO()
    Image.new("L", (3000, 2000)).save(large, format="PNG")
    assert client.post("/api/report", data=large.getvalue(), content_type="image/png").status_code == 400
    reports = []
    for example in client.get("/api/examples").json["examples"]:
        image = client.get(f"/api/examples/{example['id']}/image").data
        response = client.post("/api/report", data=image, content_type="image/png")
        assert response.status_code == 200, response.json
        result = response.json
        assert result["model_run"] == "nih10000_bce_gpu"
        assert len(result["evidence"]) == 14
        assert result["processed_image"].startswith("data:image/png;base64,")
        repeated = client.post("/api/report", data=image, content_type="image/png").json
        assert result["evidence"] == repeated["evidence"], "Identical images must give repeatable evidence"
        assert result["impression"] == repeated["impression"]
        reports.append({"example": example["id"], "seconds": result["elapsed_seconds"], "repeatable": True,
                        "flags": result["heatmap_findings"]})
    memory = None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak_rss", "rss", "peak_page", "page", "peak_nonpage", "nonpage", "pagefile", "peak_pagefile")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        memory = {"rss_mb": round(counters.rss / 1024**2, 1), "peak_rss_mb": round(counters.peak_rss / 1024**2, 1), "platform": "Windows; Linux container must be checked separately"}
    else:
        import resource
        memory = {"peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1), "platform": sys.platform}
    result = {"routes": checked, "invalid_upload_checks": 3, "reports": reports, "memory": memory}
    Path("deployment_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
