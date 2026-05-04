# Offline, Webcam, and Mobile Motion Mapping Notes

This note explains how the current motion-capture system maps video input into pose data, metrics, validation artifacts, and 3D avatar motion. It covers the offline video pipeline, webcam integration, phone/mobile stream integration, result chart generation, and known issues.

## 1. High-Level Architecture

The system has three main capture paths:

1. Offline video verification
2. Live single-camera capture from webcam, phone stream, or video file
3. Multi-camera master/server capture for synchronized 3D reconstruction

All paths reuse the same core motion stack:

```text
Input frame
  -> Camera/OpenCV acquisition
  -> MocapDetector
  -> MediaPipe pose inference
  -> FrameQualityAnalyzer, offline path
  -> PoseCorrector
  -> BoneLengthTracker and Calculations
  -> visualization, database, CSV, JSON, or avatar retargeting output
```

Main files:

- `tools/process_video.py`: offline verification pipeline
- `src/camera.py`: webcam, phone stream, file, or URL capture wrapper
- `src/detector.py`: MediaPipe detector and preprocessing
- `src/pose_corrector.py`: landmark smoothing, visibility gating, bone constraints
- `src/calculations.py`: angles, body metrics, bone length tracking
- `src/frame_quality.py`: offline quality scoring and interpolation helpers
- `main_gui.py`: live GUI, source switching, recording, master/server logic
- `src/camera_server.py`: network camera server for multi-camera mode
- `src/master_coordinator.py`: remote frame sync, triangulation, fusion, and layered database save
- `tools/generate_analysis_charts.py`: chart generation for `analysis_results/`
- `tools/local_3d_studio.py`: Panda3D avatar retargeting from raw 3D node CSV

## 2. Offline Video Mapping

Offline processing is handled by `tools/process_video.py`.

The processor opens a video with OpenCV:

```python
capture = cv2.VideoCapture(input_path)
fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
```

For each frame, the pipeline:

1. Reads the next frame from the video.
2. Creates a synthetic timestamp from frame index and FPS.
3. Runs `MocapDetector.process()`.
4. Extracts pose and world landmarks.
5. Scores frame quality with `FrameQualityAnalyzer`.
6. Runs `PoseCorrector`.
7. Computes bone length and consistency metrics with `BoneLengthTracker`.
8. Draws the annotated output video.
9. Writes per-frame metric rows to `_metrics.csv`.
10. Writes per-frame world landmarks to `_raw_3d_nodes.csv`.
11. Produces a summary JSON when requested.

The key loop is:

```python
timestamp_ms = int((frame_idx / fps) * 1000.0)
results = detector.process(frame, timestamp_ms=timestamp_ms)
quality = analyzer.analyze(pose_lm, width, height)
results = corrector.process(results, timestamp_ms=timestamp_ms, frame_quality=quality)
```

Offline mode intentionally disables face and hand detection:

```python
detector = MocapDetector(enable_face=False, enable_hand=False)
```

That makes offline verification body-pose focused and avoids spending compute on unused face/hand landmarks.

## 3. Synthetic Timestamp Creation

Synthetic timestamps are created using:

```python
timestamp_ms = int((frame_idx / fps) * 1000.0)
```

Example at 30 FPS:

| Frame | Timestamp |
|---:|---:|
| 0 | 0 ms |
| 1 | 33 ms |
| 2 | 66 ms |
| 30 | 1000 ms |

These timestamps make offline playback deterministic. The same video produces the same timestamp sequence every run, independent of CPU speed or wall-clock delays.

`MocapDetector` then enforces monotonic timestamps before calling MediaPipe:

```python
if timestamp_ms <= self.last_timestamp_ms:
    timestamp_ms = self.last_timestamp_ms + 1
self.last_timestamp_ms = timestamp_ms
```

This matters because MediaPipe `detect_for_video()` expects strictly increasing timestamps. If timestamps repeat or go backward, MediaPipe tracking can become unstable or reject frames.

One small implementation detail: since `last_timestamp_ms` starts at `0`, the first offline timestamp of `0` becomes `1`. That is harmless and ensures monotonicity.

## 4. MocapDetector

`MocapDetector` is the neural inference layer in `src/detector.py`.

It uses MediaPipe Tasks:

- `PoseLandmarker`
- optional `FaceLandmarker`
- optional `HandLandmarker`

The pose model path is selected from `config.MODEL_PATHS`. The default complexity is currently:

```python
POSE_MODEL_COMPLEXITY = 'FULL'
```

Available pose models:

- `LITE`: fastest
- `FULL`: balanced/default
- `HEAVY`: most accurate but slower

The detector uses `vision.RunningMode.VIDEO`, so timestamps are part of the detector contract.

Before inference, the detector preprocesses frames:

1. Resizes wide frames to `MAX_FRAME_WIDTH`.
2. Applies optional gamma correction.
3. Applies optional face-guided exposure.
4. Applies CLAHE contrast enhancement.
5. Converts BGR to RGB or RGBA for MediaPipe.
6. Runs MediaPipe on CPU or GPU delegate.

Important config values:

```python
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
MAX_FRAME_WIDTH = 1280
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
INFERENCE_BACKEND = 'mps'
```

The detector also has GPU fallback behavior. If the requested GPU delegate fails during model creation, it retries on CPU.

## 5. Why The Offline Output Looks Clear

The clarity comes from a combination of image, timing, and skeleton stabilization decisions.

Image-level reasons:

- The system keeps frames up to `MAX_FRAME_WIDTH = 1280`, preserving more joint-localization detail.
- CLAHE improves local contrast before MediaPipe sees the frame.
- Gamma and exposure controls exist, though default gamma and face exposure are currently conservative.

Timing reasons:

- Offline videos use synthetic timestamps from frame index and FPS.
- MediaPipe receives stable monotonic timestamps.
- Processing speed does not alter the motion timeline.

Skeleton reasons:

- Low-confidence landmarks are gated or held.
- World landmarks are smoothed with One Euro filters.
- Subject-specific bone lengths are calibrated.
- Bone stretch is constrained after calibration.
- Short missing-pose gaps are interpolated.
- Long missing-pose gaps are marked as tracking loss instead of pretending the track is valid.

So the output is clear because the system is not merely drawing raw MediaPipe landmarks. It is building a stable, auditable motion stream.

## 6. PoseCorrector

`PoseCorrector` is the physics/stabilization layer.

It receives MediaPipe results and edits the pose result in place.

Key behavior:

1. Preserves raw landmarks before correction.
2. Leaves normalized 2D display landmarks raw to avoid visual lag.
3. Applies correction mainly to `pose_world_landmarks`.
4. Uses visibility hard gates.
5. Uses One Euro filters per landmark.
6. Calibrates reference bone lengths.
7. Reconstructs or holds unreliable low-confidence landmarks.
8. Applies visibility-weighted bone-length constraints after calibration.

Current key config values:

```python
VISIBILITY_HARD_GATE = 0.5
FILTER_MIN_CUTOFF = 1.0
FILTER_BETA = 0.05
BONE_LENGTH_CALIBRATION_FRAMES = 30
BONE_LENGTH_MIN_CONFIDENCE = 0.50
MIN_VALID_BONES_PER_CALIBRATION_FRAME = 5
```

The skeleton hierarchy is limb-oriented:

| Child | Parent | Segment |
|---:|---:|---|
| 14 | 12 | right shoulder to right elbow |
| 16 | 14 | right elbow to right wrist |
| 13 | 11 | left shoulder to left elbow |
| 15 | 13 | left elbow to left wrist |
| 26 | 24 | right hip to right knee |
| 28 | 26 | right knee to right ankle |
| 25 | 23 | left hip to left knee |
| 27 | 25 | left knee to left ankle |

During calibration, the corrector collects bone lengths only when both joints are sufficiently visible. After enough valid calibration frames, it freezes median reference lengths and activates constraints.

When visibility is low:

- If the parent is reliable and a previous direction exists, the child can be reconstructed from the parent and reference length.
- If reconstruction is not possible, the last filtered position is held.
- If no stable previous state exists, the landmark is marked as rejected/uninitialized.

This prevents low-confidence joints from causing limb snapping or impossible stretching.

## 7. BoneLengthTracker

`BoneLengthTracker` lives in `src/calculations.py`.

It is a stateful metrics and reporting component. It is related to `PoseCorrector`, but it has a different job.

`PoseCorrector` edits world landmarks.

`BoneLengthTracker` measures and reports whether the skeleton is stable.

It:

1. Prefers `world_landmarks` if available.
2. Falls back to normalized pose landmarks if world landmarks are missing.
3. Smooths or carries landmark state.
4. Computes raw body lengths and widths.
5. Learns reference lengths during early frames.
6. Normalizes current lengths against the reference.
7. Computes constrained length values.
8. Updates variance and standard deviation.
9. Computes a weighted bone consistency score.

Tracked metric families include:

- `Length_*`
- `Normalized_*`
- `Constrained_*`
- `Bone_Length_Variance_*`
- `Bone_Length_StdDev_*`
- `World_*`
- `Reference_*`
- `Bone_Consistency_Score`
- `Stable_Frame_Used`

Important tracked segments include:

- upper arm left/right
- lower arm left/right
- upper leg left/right
- lower leg left/right
- shoulder width
- hip width

Current tracker config:

```python
BONE_LENGTH_CALIBRATION_FRAMES = 30
BONE_LENGTH_EMA_ALPHA = 1.0
BONE_LENGTH_MAX_DEVIATION = 0.20
BONE_LENGTH_MIN_CONFIDENCE = 0.50
```

`BONE_LENGTH_EMA_ALPHA = 1.0` means redundant EMA smoothing is effectively disabled in this tracker because `PoseCorrector` is already doing the main stabilization.

## 8. FrameQualityAnalyzer

`FrameQualityAnalyzer` is used in the offline pipeline to decide which frames are trustworthy.

It computes:

- mean visibility
- visible joint ratio
- pose bounding box area ratio
- normalized motion jump from the previous detected frame
- final quality score

It classifies frames as:

- `missing`
- `low_confidence`
- `partial_pose`
- `unstable`
- `valid`

Only valid usable frames are allowed to update bone reference metrics in the offline pipeline. This prevents bad frames from poisoning calibration and stability statistics.

## 9. Missing Frame Handling

Offline processing has explicit gap handling.

If pose is missing for a short gap, up to 5 frames, the system interpolates between the last detected pose and the next detected pose.

If the gap is longer than 5 frames, the system treats it as tracking loss and uses the last stable pose when available.

This produces a more continuous annotated video while still recording the truth in the CSV through `frame_state`:

- `detected`
- `interpolated`
- `tracking_loss`
- `missing`

## 10. Webcam Integration

Webcam integration is based on OpenCV `VideoCapture`.

The wrapper is `src/camera.py`.

`Camera` accepts:

- integer device IDs, for example `0`
- string device IDs, for example `"0"`
- local video paths
- stream URLs

The normalization logic is:

```python
if isinstance(source, str):
    stripped = source.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped
return source
```

For a real webcam device, OpenCV capture properties are set:

```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
self.cap.set(cv2.CAP_PROP_FPS, FPS)
```

The camera runs a background thread that continuously reads the latest frame. The live processing loop then asks for the latest frame instead of blocking on slow capture.

In `main_gui.py`, the live path is:

```text
Camera.read()
  -> optional mirror flip
  -> detector.process(frame)
  -> corrector.process(results)
  -> _compute_stream_metrics()
  -> visualizer.draw_landmarks()
  -> optional database save
  -> GUI update
```

The launcher supports webcam mode:

```bash
python launch_multi_camera.py --mode single --camera-source 0
```

The GUI also supports switching the source at runtime through `apply_selected_source()`.

## 11. Mobile / Phone Integration

Phone integration uses the same OpenCV source mechanism as webcam integration.

The phone is treated as an IP camera or MJPEG stream. Example:

```bash
python launch_multi_camera.py --mode single --camera-source http://<PHONE_IP>:8080/video
```

That URL is passed to:

```python
cv2.VideoCapture(source)
```

So the phone stream becomes just another frame source. After the frame enters the system, the rest of the pipeline is identical to webcam input:

```text
phone stream URL
  -> OpenCV VideoCapture
  -> Camera background thread
  -> MocapDetector
  -> PoseCorrector
  -> Calculations and BoneLengthTracker
  -> GUI, DB, or validation outputs
```

The GUI validates phone mode by requiring a non-empty URL:

```python
if mode == 'phone':
    if not text:
        raise ValueError("Phone stream URL cannot be empty.")
    return text
```

This means the mobile integration is lightweight and practical: no special phone SDK is used. The phone only needs to expose a camera stream URL that OpenCV can read.

## 12. Multi-Camera / Network Integration

The multi-camera system has two roles:

- `server`: remote camera node
- `master`: main node that receives remote frames and combines them with local frames

Commands:

```bash
python launch_multi_camera.py --mode server
python launch_multi_camera.py --mode master --remote-ip <SERVER_IP>
```

The server side uses `CameraServer`.

It:

1. Broadcasts discovery packets.
2. Serializes pose landmarks and world landmarks.
3. JPEG-encodes the display frame.
4. Sends packed frame data over ZeroMQ.
5. Replies to clock sync pings when enabled.

Frame payload includes:

- `schema_version`
- `camera_id`
- `frame_number`
- `timestamp`
- `calibration_id`
- `capture_fps`
- compact 2D landmarks
- serialized MediaPipe results
- GPU/inference timing
- optional JPEG frame

The master side uses `MasterCoordinator`.

It:

1. Connects to the remote camera.
2. Estimates clock offsets using Cristian's Algorithm.
3. Buffers local and remote frames.
4. Applies clock correction to timestamps.
5. Finds synchronized frame batches.
6. Runs triangulation when two views are available.
7. Falls back to monocular estimates when stereo is unavailable.
8. Saves layered database rows when recording.

The live GUI adds the local camera frame into the coordinator buffer manually, because the coordinator receives remote frames automatically but the master machine's own camera is local.

The synchronization strategy uses the newest frame from the slowest camera as the anchor, then finds nearest frames from each camera within the allowed threshold. This avoids a fast camera running far ahead of a slower remote camera.

## 13. Avatar Retargeting

`tools/local_3d_studio.py` reads the offline `_raw_3d_nodes.csv`.

The CSV contains:

- `frame_idx`
- `timestamp_ms`
- root hip-center position
- 33 MediaPipe world landmarks
- visibility for each landmark

The studio maps MediaPipe segments to avatar bones.

Example Mixamo mapping:

| Avatar bone | MediaPipe segment |
|---|---|
| `mixamorigLeftArm` | left shoulder to left elbow |
| `mixamorigLeftForeArm` | left elbow to left wrist |
| `mixamorigRightArm` | right shoulder to right elbow |
| `mixamorigLeftUpLeg` | left hip to left knee |
| `mixamorigSpine` | mid hip to mid shoulder |
| `mixamorigNeck` | mid shoulder to nose |

The retargeting process:

1. Load the GLB avatar with Panda3D.
2. Resolve available skeleton bone names.
3. Capture a clean human rest pose from the CSV.
4. Capture the avatar's rest vectors.
5. For each frame, compute human segment direction change.
6. Convert direction change into a quaternion.
7. Apply per-bone gain and angle limits.
8. Smooth the render-space quaternion.
9. Apply it to the controlled avatar joint.

This is better than directly forcing avatar bones to raw world vectors because it uses the change from rest pose. The avatar follows the action rather than copying camera-space orientation literally.

## 14. How `analysis_results/` Is Produced

The `analysis_results/` folder is produced by `tools/generate_analysis_charts.py`.

Input:

```text
data/offline_validation_runs/*_metrics.csv
```

Output:

```text
analysis_results/fig1_variance.png
analysis_results/fig1_jitter.png
analysis_results/fig1_gantt.png
analysis_results/fig1_bone_lengths.png
analysis_results/fig1_symmetry.png
analysis_results/fig1_visibility.png
analysis_results/fig1_fps.png
analysis_results/fig1_dashboard.png
```

For each metrics CSV, the script assigns a chronological prefix:

```python
prefix_name = f"fig{i+1}"
```

So the oldest metrics CSV becomes `fig1_*`, the next becomes `fig2_*`, and so on.

The charts are:

| Chart | What it shows |
|---|---|
| `variance` | bone variance over time and QA threshold |
| `jitter` | motion jump plus consistency score |
| `gantt` | detected/interpolated/loss timeline |
| `bone_lengths` | per-bone length traces against reference lengths |
| `symmetry` | left-vs-right limb length scatter |
| `visibility` | mean visibility, minimum visibility, and quality score |
| `fps` | processing trend and effective FPS estimate |
| `dashboard` | combined review page with variance, jitter, FPS, state pie, asymmetry, quality histogram |

The full offline validation script is:

```bash
scripts/run_offline_validation_linux.sh "/absolute/path/to/video.mp4"
```

That script:

1. Creates or uses a local virtual environment.
2. Installs offline dependencies.
3. Ensures MediaPipe model files exist.
4. Runs a Python syntax sweep.
5. Runs `tools/process_video.py`.
6. Writes annotated MP4, metrics CSV, raw 3D node CSV, and summary JSON.
7. Re-encodes MP4 with `ffmpeg -movflags +faststart` if available.
8. Runs balanced and strict quality gates.

`generate_analysis_charts.py` is then used to convert the metrics CSVs into PNG diagnostics.

## 15. Result Artifacts

Offline run artifacts usually appear in:

```text
data/offline_validation_runs/
```

Common files:

- `annotated_<run_id>.mp4`
- `annotated_<run_id>_metrics.csv`
- `annotated_<run_id>_raw_3d_nodes.csv`
- `summary_<run_id>.json`
- `quality_report_<run_id>.json`
- `quality_report_strict_<run_id>.json`

Analysis charts appear in:

```text
analysis_results/
```

The README currently references charts such as:

- `analysis_results/fig1_dashboard.png`
- `analysis_results/fig1_symmetry.png`

## 16. Architectural Changes and Novelty

The strongest architectural changes are:

1. Offline validation became a first-class pipeline.
2. The system now separates detection, correction, metric tracking, quality scoring, visualization, and validation.
3. Offline processing uses deterministic synthetic timestamps.
4. World-space landmarks are preferred for physics and metrics.
5. Bone references are learned per subject from valid early frames.
6. Low-confidence joints are held, rejected, or reconstructed instead of blindly trusted.
7. Short missing pose gaps are interpolated.
8. Long pose gaps are explicitly marked as tracking loss.
9. Multi-camera mode adds timestamp synchronization, clock correction, and triangulation.
10. The database path has layered persistence for frame metadata, raw landmarks, 3D joints, kinematics, and validation outputs.
11. Avatar retargeting uses rest-pose delta rotations with per-bone limits and smoothing.

The novelty is not the base pose detector itself. MediaPipe provides the landmark model. The novel part is the reliability architecture around it:

- synthetic timestamp replay
- quality-aware offline validation
- subject-specific skeleton calibration
- visibility-aware correction
- perspective-aware reliability interpretation
- auditable CSV/JSON/PNG artifacts
- bridge from normal video to stable 3D avatar retargeting

## 17. Known Issues and Limitations

### 17.1 Offline Video Limitations

- The offline path is still fundamentally single-camera unless paired with multi-camera capture data.
- MediaPipe world landmarks are estimated from monocular input, not true measured 3D.
- Fast motion blur can still reduce visibility and cause tracking loss.
- If the first 30 usable frames are poor, bone reference calibration can be weaker.
- Short gaps are interpolated, but interpolation is not true recovered motion.
- Long missing gaps are held/tracking-loss, so motion can freeze.

### 17.2 Webcam Issues

- Webcam quality depends heavily on lighting, exposure, and shutter speed.
- Low-light webcams can produce jitter even when the person is visible.
- OpenCV camera indexes differ by machine, so `0`, `1`, etc. may need testing.
- Some cameras ignore requested FPS or resolution.
- Mirroring changes display orientation; metric interpretation should account for left/right expectations.

### 17.3 Mobile / Phone Stream Issues

- Phone stream latency depends on WiFi quality.
- MJPEG/IP camera apps can have variable FPS.
- Some phone stream URLs may not be readable by OpenCV.
- Network jitter can cause capture stalls or uneven live motion.
- Phone autofocus/exposure changes can shift landmark confidence.
- If the phone is handheld, camera shake can appear as body motion.

### 17.4 Multi-Camera Issues

- Requires stable network ports and firewall permission.
- Clock sync reduces systematic offset but cannot eliminate random WiFi latency.
- Triangulation quality depends on camera calibration and camera placement.
- If camera timestamps spread too far, sync can fail.
- Remote data may still be saved in fallback mode, but true stereo reconstruction requires synchronized frames.
- Calibration ID and camera ID mismatches can cause the coordinator to use fallback paths.

### 17.5 GPU / Runtime Issues

- `INFERENCE_BACKEND = 'mps'` is configured for macOS Metal acceleration, but the detector can fall back to CPU if GPU initialization fails.
- Apple GPU paths require image heights compatible with CVPixelBuffer constraints; the detector rounds resized height to a multiple of 16 when GPU is active.
- Offline validation script may download MediaPipe models, so first run needs network access.
- `ffmpeg` is optional. If it is missing, MP4 faststart re-encoding is skipped.

### 17.6 Analysis Result Issues

- `analysis_results/figN_*` numbering depends on chronological order of metrics CSV files.
- If old CSV files remain in `data/offline_validation_runs/`, generated figure numbers may not match the latest video names.
- Charts assume expected CSV columns exist; changing `tools/process_video.py` headers can break chart generation.
- The metrics CSV column named `processing_time_s` is currently cumulative elapsed processing time since the run started, not isolated per-frame processing time. Because `chart_processing_fps()` computes `1.0 / processing_time_s`, the FPS chart is best treated as a rough processing trend until the CSV exports true per-frame duration.

## 18. Practical Commands

Run live webcam:

```bash
python launch_multi_camera.py --mode single --camera-source 0
```

Run phone stream:

```bash
python launch_multi_camera.py --mode single --camera-source http://<PHONE_IP>:8080/video
```

Run local video as live source:

```bash
python launch_multi_camera.py --mode single --camera-source path/to/video.mp4
```

Run offline validation:

```bash
scripts/run_offline_validation_linux.sh "/absolute/path/to/video.mp4"
```

Generate analysis charts:

```bash
python tools/generate_analysis_charts.py
```

Run multi-camera server:

```bash
python launch_multi_camera.py --mode server
```

Run multi-camera master:

```bash
python launch_multi_camera.py --mode master --remote-ip <SERVER_IP>
```

Run local avatar studio:

```bash
python tools/local_3d_studio.py
```
