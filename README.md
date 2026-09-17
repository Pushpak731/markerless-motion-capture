# Markerless 3D Motion Capture & Avatar Studio

Real-time multi-camera markerless 3D motion capture on commodity hardware — MediaPipe pose estimation, confidence-weighted 3D reconstruction, kinematic analytics, and live avatar retargeting, backed by a full offline validation suite.

![Architecture](docs/screenshots/architecture-overview.png)

> **Team project** built during an internship — see [Contributors](../../graphs/contributors) for authorship. Core pipeline development and commit history by the internship team; repository hosted on this account.

## Capabilities

### Capture
- Single-camera capture from a **webcam, phone/IP stream, or local video file**
- **Multi-camera server/master** setup for synchronized 3D reconstruction over the network
- Zero-latency 2D visual tracking, decoupled from strictly stabilized 3D physics metrics

### 3D pipeline
- **MediaPipe HEAVY** pose inference for high-accuracy joint tracking
- **Bone-length stabilization** with stateful tracking and world-space preference
- **OneEuro smoothing** — responsive jitter reduction without motion lag
- **Perspective-aware reliability engine** that classifies angle vs. error and raises smart warnings

### Studio & visualization
- Live **Tkinter dashboard** plus a **React/Three.js** web frontend
- **Panda3D desktop 3D Avatar Studio** — hardware-accelerated GLB retargeting to Mixamo-standard rigs, with play/pause and frame-step controls

### Validation suite
- Offline annotation export with per-landmark correction metadata
- **8-chart diagnostics**: bone variance, jitter, symmetry, visibility, FPS, and more
- **11-point automated quality gate** (balanced + strict) for athletic trials
- Automatic `faststart` re-encoding for mobile sharing

## Quick start

```bash
cd Motion-capture
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main_gui.py
```

### Camera sources

Use `--camera-source` with the launcher for anything other than the default webcam:

```bash
# Webcam
python launch_multi_camera.py --mode single --camera-source 0

# Phone stream
python launch_multi_camera.py --mode single --camera-source http://<PHONE_IP>:8080/video

# Offline video file
python launch_multi_camera.py --mode single --camera-source path/to/video.mp4
```

### Multi-camera mode

```bash
# Server laptop
python launch_multi_camera.py --mode server

# Master laptop
python launch_multi_camera.py --mode master --remote-ip <SERVER_IP>
```

See [Motion-capture/docs/SETUP.md](Motion-capture/docs/SETUP.md) for the full network and firewall setup.

## Validation workflow

1. Run the live app or the offline verifier.
2. Export an annotated video with `tools/process_video.py` when checking metric stability.
3. Use `tools/validate_session.py` and `tools/compare_angles.py` for database-backed sessions.

One-command Linux verification (full offline pipeline + automated quality checks):

```bash
scripts/run_offline_validation_linux.sh "/absolute/path/to/video.mp4"
```

This performs venv dependency setup, model file checks/download, a Python syntax sweep, offline video annotation export, and automated quality gate reports (balanced + strict).

## Trial results

Genuine outputs of the offline validation pipeline on two recorded trials:

| Metric | Video 1 (May 01) | Video 2 (May 03) | Status |
|---|---|---|---|
| Pose coverage | 100% | 91% | PASS |
| Bone variance | 0.0003 | 0.0002 | EXCELLENT |
| Limb symmetry | 66.9% (angle) | 4.9% (frontal) | PERSPECTIVE |
| Reliability score | 72.4 / 100 | 48.1 / 100 | RELIABLE |

| Video 1 — dashboard | Video 1 — symmetry analysis |
|---|---|
| ![Dashboard](Motion-capture/analysis_results/fig1_dashboard.png) | ![Symmetry](Motion-capture/analysis_results/fig1_symmetry.png) |

## Project structure

```
Motion-capture/
├── src/           core stabilization logic (calculations, detector, pose corrector)
├── tools/         offline processing (ReliabilityEngine, ProcessVideo, Local3DStudio)
├── frontend/      web-based 3D dashboard (React, Three.js)
├── scripts/       automation + one-command validation
├── tests/         unit and integration tests
└── docs/          SETUP, WORKFLOW_FLOW, DATABASE_SCHEMA, coordinate system spec
```

A technical deep-dive into the pipeline is in [Motion-capture/docs/WORKFLOW_FLOW.md](Motion-capture/docs/WORKFLOW_FLOW.md); the system architecture lives in [docs/architecture.mmd](docs/architecture.mmd) with rendered diagrams in [docs/screenshots/](docs/screenshots/).
