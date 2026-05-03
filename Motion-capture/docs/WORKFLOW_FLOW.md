# Actual System Workflow: From Video to 3D Avatar

This document describes the step-by-step logic of the **Offline Intelligence Pipeline**.

---

## 🟢 Step 1: Frame Acquisition
*   **Source:** MP4/WebM video file.
*   **Ingestion:** OpenCV `VideoCapture` reads frames at native FPS.
*   **Preprocessing:** Frames are resized to `MAX_FRAME_WIDTH` (640px) to ensure consistent inference latency across different input qualities.

## 🟢 Step 2: Neural Inference (MediaPipe)
*   **Model:** `Pose Landmarker` (Heavy profile).
*   **Raw Output:** 33 landmarks in 2D (pixel space) and 3D (World-meters relative to hips).
*   **Coordinate System:** 
    *   `Origin (0,0,0)`: Hip center.
    *   `Unit`: Meters.
    *   `Visibility`: 0.0 - 1.0 confidence score per joint.

## 🟢 Step 3: Pose Correction (`PoseCorrector`)
This is the "Brain" of the stabilization pipeline.
1.  **Visibility Hard Gate (0.1):** If a landmark's visibility is < 0.1, it is discarded immediately to prevent "phantom joints."
2.  **Temporal Smoothing (OneEuro):** 
    *   High-speed movements are tracked with low lag.
    *   Static poses are smoothed with high drag to eliminate jitter.
3.  **Calibration Phase (First 30 Frames):**
    *   The system collects bone length samples in a buffer.
    *   Once 30 valid frames are seen, it freezes a **Reference Skeleton**.
4.  **Physics Constraints:**
    *   Every incoming bone length is compared to the Reference.
    *   If a bone is > 15% longer/shorter than its reference (biomechanically impossible), it is **clamped** to the limit.
    *   This eliminates the "stretching" effect common in 2D-to-3D mocap.

## 🟢 Step 4: Reliability Analysis (`ReliabilityEngine`)
*   **Asymmetry Analysis:** Compares Left vs Right limb lengths.
*   **Perspective Logic:** 
    *   If variance is LOW but asymmetry is HIGH → **Warning: Perspective Angle.**
    *   If variance is HIGH but asymmetry is HIGH → **Error: Tracking Failure.**
*   **Jitter Spiking:** Scans for `motion_jump > 0.2m` per frame.

## 🟢 Step 5: Data Export
*   **`_metrics.csv`**: Statistical summary of skeletal health.
*   **`_raw_3d_nodes.csv`**: Frame-by-frame XYZ coordinates for all 33 joints.
*   **`summary.json`**: Pass/Fail report for the Quality Gate.

## 🟢 Step 6: 3D Studio Rendering (`local_3d_studio.py`)
*   **Native Loading:** Panda3D loads the character `.glb`.
*   **Skeletal Mapping:** 
    *   The `_raw_3d_nodes.csv` is parsed.
    *   The engine computes quaternions between joints (e.g. Shoulder → Elbow).
    *   The `mixamorig` bone rotations are updated in real-time on the GPU.
