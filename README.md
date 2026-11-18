# XRayGuard

A capstone research demo that serves a chest X-ray upload webpage and its Python model API from one Docker service. Includes the trained NIH DenseNet-121 checkpoint, calibration, CAM visualizations, uncertainty scores, constrained draft reports, and held-out evaluation charts.

This is a research prototype. Clinical correctness has not been established. Below-threshold findings do not exclude disease; CAM maps are coarse model explanations. The report verifier checks agreement with model evidence, not clinical truth. Use public, de-identified demonstration images.

## Run the complete website

```sh
docker build -t xrayguard .
docker run --rm -p 10000:10000 xrayguard
```

Open http://localhost:10000. No dataset download, GPU, external LLM, or API key is needed. Both frontend assets and backend endpoints share the same host.

## Deploy to Render

Connect this GitHub repository in Render, choose **New > Blueprint**, select this repository, and review the single **Free** Docker web service declared in `render.yaml`. Alternatively create a Docker Web Service with this repository, the Free plan, and health check `/healthz`. The container binds to the platform's `PORT` automatically.

The free service has a limited memory budget and sleeps when idle. Successful remote image inference must be verified after deployment; a successful `/healthz` alone does not establish that model inference fits within the host's RAM. No paid upgrade is authorized by this configuration.

## Included model and evidence

- Model run: `nih10000_bce_gpu`; DenseNet-121, 14 NIH labels, 30 head-dropout samples.
- Experiment split: 8,000 training, 1,000 validation, 1,000 test patients.
- Held-out mean AUROC: 0.8190. This is a ranking metric, not 81.9% diagnostic accuracy.
- `model_integrity.json` records the checkpoint checksum.
- `/results` presents measured detection and reporting limitations, including weak pneumonia sensitivity and original report metrics.
- `/about` explains the architecture and related work.
- Public NIH demonstration images retain their dataset labels, source filenames, and split provenance in `demo_samples/catalog.json`; these are not expert-adjudicated case studies.

The snapshot is prepared from the parent XRayGuard project using `prepare_deployment.py`. Training datasets, caches, virtual environments, unrelated experiments, patient manifests, notebooks, and user documents are excluded. Trained weights and aggregate evaluation files are included so the backend can run immediately.

Uploads are processed in memory; this app does not save uploaded images or report history. The hosting provider still handles network requests and operational logs. Do not upload identifiable patient information to this demo.

Source dataset: [NIH Chest X-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC). Dataset labels are weak labels extracted from reports, not complete reference reports. Third-party library and dataset terms remain applicable; no new license is granted for third-party material by this snapshot.
