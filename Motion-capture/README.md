# Motion Capture System

Active implementation for the motion-capture workspace.

This folder contains the live app, launchers, and the metric pipeline that now supports webcam, phone, and offline video verification inputs.

## Current capabilities

- single-camera capture from a webcam, phone/IP stream, or local video file
- multi-camera server/master capture for synchronized 3D reconstruction
- offline annotation export for verification videos
- zero-latency 2D visual tracking decoupled from strictly stabilized 3D physics metrics
- HEAVY pose model complexity enabled by default for maximum accuracy
- per-landmark correction metadata exported in offline validation CSVs
- bone-length stabilization with stateful tracking and world-space preference
- live Tkinter dashboard plus web frontend support

## Quick start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main_gui.py
```

## Camera sources

Use `--camera-source` with the launcher when you want anything other than the default webcam:

```bash
# Webcam
python launch_multi_camera.py --mode single --camera-source 0

# Phone stream
python launch_multi_camera.py --mode single --camera-source http://<PHONE_IP>:8080/video

# Offline video file
python launch_multi_camera.py --mode single --camera-source path/to/video.mp4
```

## Multi-camera mode

```bash
# Server laptop
python launch_multi_camera.py --mode server

# Master laptop
python launch_multi_camera.py --mode master --remote-ip <SERVER_IP>
```

See [docs/SETUP.md](docs/SETUP.md) for the full network and firewall setup.

## Validation workflow

1. Run the live app or the offline verifier.
2. Export an annotated video with `tools/process_video.py` when checking metric stability.
3. Use `tools/validate_session.py` and `tools/compare_angles.py` for database-backed sessions.

### One-command Linux verification

Run the full offline pipeline plus automated quality checks directly with:

```bash
scripts/run_offline_validation_linux.sh "/absolute/path/to/video.mp4"
```

Script path:

- `scripts/run_offline_validation_linux.sh`

It performs:

1. Local venv dependency setup
2. Model file checks/download
3. Python syntax sweep
4. Offline video annotation export
5. Automated quality gate reports (balanced + strict)

Artifacts are written under:

- `data/offline_validation_runs/`

Key outputs:

- `annotated_<timestamp>.mp4`
- `summary_<timestamp>.json`
- `quality_report_<timestamp>.json`
- `quality_report_strict_<timestamp>.json`

### Recent Verification Results

Our latest automated test runs produced the following offline verification artifacts. These validate our zero-latency 2D visualization architecture and strict 3D physical constraints.

- **Video 1 (May 03)**
  - Original: `WhatsApp Video 2026-05-03 at 1.02.28 PM.mp4`
  - Annotated Output: [annotated_20260503_144652.mp4](data/offline_validation_runs/annotated_20260503_144652.mp4)
  - Metrics Data: [annotated_20260503_144652_metrics.csv](data/offline_validation_runs/annotated_20260503_144652_metrics.csv)
  - Generated Charts: [Variance](analysis_results/annotated_20260503_144652_metrics_variance.png), [Jitter](analysis_results/annotated_20260503_144652_metrics_jitter.png), [Gantt](analysis_results/annotated_20260503_144652_metrics_gantt.png)

- **Video 2 (May 01)**
  - Original: `WhatsApp Video 2026-05-01 at 1.00.41 PM.mp4`
  - Annotated Output: [annotated_20260503_144734.mp4](data/offline_validation_runs/annotated_20260503_144734.mp4)
  - Metrics Data: [annotated_20260503_144734_metrics.csv](data/offline_validation_runs/annotated_20260503_144734_metrics.csv)
  - Generated Charts: [Variance](analysis_results/annotated_20260503_144734_metrics_variance.png), [Jitter](analysis_results/annotated_20260503_144734_metrics_jitter.png), [Gantt](analysis_results/annotated_20260503_144734_metrics_gantt.png)

### Frontend upload flow (offline verification)

You can now upload a video from the React dashboard and get back an annotated output video.

1. Start the verification API:
	```bash
	python tools/offline_verify_api.py
	```
2. Start the frontend:
	```bash
	cd frontend
	npm install
	npm run dev
	```
3. Open the dashboard and use the **Offline Video Verify** panel.

## Repository layout

```text
Motion-capture/
├── config.py
├── main_gui.py
├── launch_multi_camera.py
├── CHANGES.md
├── IMPLEMENTATION.md
├── docs/
├── frontend/
├── scripts/
├── src/
├── tests/
└── tools/
```

The detailed architecture and reference material live in [docs/](docs/) and the top-level root docs.