# Motion Capture — Hero Overview

What this project does
----------------------
Capture and convert video (live camera, phone stream, or recorded file) into robust 2D pose motion-capture data and validation metrics. The codebase supports:

- single-camera and multi-camera capture
- offline video verification with annotated output (MP4 + CSV/JSON diagnostics)
- stabilized bone-length tracking and temporal smoothing
- quality-scored frames, short-gap interpolation and explicit tracking-loss reporting

Tech stack
----------

- Python 3 (Linux/macOS/Windows)
- OpenCV for image I/O and annotation
- MediaPipe Tasks (Pose Landmarker) for landmark detection
- NumPy / SciPy for numeric processing
- Simple Flask/Vite frontend for dashboards (optional)

Quick demo / screenshots
------------------------

- Example annotated output (recorded run): data/offline_validation_runs/annotated_20260503_132209.mp4
- Example metrics CSV and summary JSON: data/offline_validation_runs/annotated_20260503_132209_metrics.csv and data/offline_validation_runs/summary_20260503_132209.json

Install
-------

From the repo root create and activate a virtual environment, then install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r Motion-capture/requirements.txt
```

Run tests (optional):

```bash
pytest -q Motion-capture/tests
```

How to run (offline verification)
---------------------------------

Process a video file and generate an annotated MP4, CSV and JSON summary:

```bash
# from repo root
python3 Motion-capture/tools/process_video_offline.py \
	--input "/path/to/video.mp4" \
	--output-dir Motion-capture/data/offline_validation_runs
```

Or run the wrapper script to see CLI help:

```bash
python3 Motion-capture/tools/process_video_offline.py --help
```

Validation results (example)
----------------------------

Summary from a recent offline run (example):

- input: WhatsApp Video 2026-05-01 at 1.00.41 PM.mp4
- frames seen: 348, frames annotated: 348
- pose frames: 257 (face/hand frames: 0)
- pose coverage: 73.85%
- mean consistency score: 0.2451
- processing fps: 37.0 (realtime ratio ≈ 1.23)
- mean absolute normalized deviation: 0.2804

Files produced for that run live under `Motion-capture/data/offline_validation_runs/` and include annotated MP4, per-frame CSV, and `summary_*.json` and `quality_report_*.json` files.

Known limitations
-----------------

- Current validation reports and example outputs are pose-only: face and hand tracking are disabled for these uploaded validation runs.
- The pipeline focuses on 2D landmarks and stabilization; 3D reconstruction and full-body physics-ready retargeting are out of scope for now.

Repository tidy / next steps
---------------------------

The repository root currently contains several exploratory scripts (for visualization and experiments). Suggested clean-up (proposed; not applied here):

- Move small utilities and ad-hoc visualizers into `visualization/` or `scripts/`.
- Consolidate validation outputs under `data/offline_validation_runs/` (already used).
- Move user-facing docs into `Motion-capture/docs/` and keep architecture assets under `docs/`.

If you want, I can apply the tidy plan next: move trail/visualizer scripts into `Motion-capture/visualization/`, add a short index doc for each moved script, and update references. Reply with "Proceed with cleanup" to continue.

Where to look next
------------------

- Active app and code: [Motion-capture/](Motion-capture/)
- Offline tools: [Motion-capture/tools/](Motion-capture/tools/)
- Validation outputs: [Motion-capture/data/offline_validation_runs/](Motion-capture/data/offline_validation_runs/)

----
Small note: this README emphasizes the verification/validation workflow and points at recent example outputs so you and collaborators can quickly re-run or inspect annotated videos and metrics.