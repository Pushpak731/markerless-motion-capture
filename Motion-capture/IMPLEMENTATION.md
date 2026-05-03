# Implementation Complete: 4-Layer Self-Aware Database

**Date**: 2026-03-02
**Status**: COMPLETE

## Summary

The repository now includes a 4-layer self-aware persistence architecture with offline validation tooling.

- Layer 0: frame timeline (`frames`)
- Layer 1: immutable 2D raw landmarks (`raw_landmarks_2d`)
- Layer 2: triangulated 3D joints (`joints_3d`)
- Layer 3: derived kinematics (`kinematics_3d`, `kinematics_2d`, derivatives)
- Layer 4: validation outputs (`validation_runs`, `validation_angle_comparison`, `validation_artifacts`)

## Integration Points

### Runtime capture path

- `main_gui.py` recording loop continues legacy save:
  - `db.save_synced_frame(...)`
- And now explicitly calls layered persistence:
  - `coordinator.save_frame_to_database(synced_batch, pose_3d)`

### Coordinator DB flow

`MasterCoordinator.save_frame_to_database()` performs:
1. Layer 0 insert via `save_frame_metadata`
2. Layer 1 inserts via `save_raw_landmarks` per camera
3. Layer 2 insert via `save_joints_3d`
4. Layer 3 insert via `save_kinematics_3d`

### Validation toolchain

- `tools/validate_session.py`
  - Recompute angles from Layer 2
  - Compare with Layer 3
  - Save Layer 4 summary and per-angle rows
- `tools/compare_angles.py`
  - Produce CSV / Markdown / JSON reports
- `tools/export_validation_artifacts.py`
  - Export 3D, optional 2D, optional angle comparison + metadata
- `tools/test_validation_system.py`
  - Smoke checks for schema + API

## Typical Workflow (Offline Intelligence)

1. **Process and Stabilize**: Use the automated validation script to process your video with physics-aware constraints.
   - `bash scripts/run_offline_validation_linux.sh "/path/to/video.mp4"`
   - This produces an annotated video, metrics CSV, raw 3D nodes CSV, and a reliability report.
2. **Review Metrics**: Open the `analysis_results/` folder to view the 8-chart diagnostic suite (Variance, Jitter, Symmetry, etc.).
3. **Animate 3D Avatar**: Run the native Panda3D studio to see the motion on a 3D character.
   - `python tools/local_3d_studio.py`
4. **Export for Analysis**: Use `_metrics.csv` for biomechanical analysis in Excel/Pandas.

## Current capture modes

- **Live Studio**: `python main_gui.py` (Full UI for real-time 3D capture).
- **Offline Intelligence**: `bash scripts/run_offline_validation_linux.sh` (Best for high-fidelity 3D verification).
- **Native 3D Viewport**: `python tools/local_3d_studio.py` (Local avatar retargeting trial).

