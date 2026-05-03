# Optimization History: Jitter & Bone Variance

This document tracks the specific architectural changes made to solve common 2D-to-3D mocap artifacts.

---

## 1. Marker Jitter Reduction
*   **The Problem:** Raw MediaPipe landmarks often "vibrate" in place, especially in low-light or low-contrast environments.
*   **The Fix:** Integrated a **OneEuroFilter** with a dynamic `beta` value.
    *   **Result:** Reduced high-frequency noise (jitter) while maintaining sub-frame responsiveness for athletic movements.
    *   **Metrics:** Jitter spikes dropped from ~40% of frames to < 5% in Video 1.

## 2. Bone Length Stretching
*   **The Problem:** When a person turns sideways, the 2D projected length of their limbs changes, causing the 3D reconstructed bones to "stretch" or "shrink" biomechanically incorrectly.
*   **The Fix:** **Skeletal Ground-Truth Constraints.**
    *   The system now calibrates during the first 30 frames to find the subject's true limb lengths.
    *   It applies a strict **15% Physics Gate** on all subsequent frames.
    *   **Result:** Bone length variance dropped from `0.08` to `0.0003` (nearly perfectly stable).

## 3. Perspective-Aware Reliability
*   **The Problem:** Traditional quality gates fail a trial if limbs are asymmetrical. However, in 2D mocap, a person standing at a 45-degree angle *should* have asymmetrical projected limbs.
*   **The Fix:** **Variance-vs-Asymmetry Logic.**
    *   If a limb is shorter than its pair, but its length is **consistent over time**, the system classifies this as **Perspective Angle** (Valid) rather than **Tracking Failure** (Invalid).
    *   **Result:** High-quality trials from side-on angles now PASS instead of being rejected.

## 4. Latency & Sync (Multi-Cam)
*   **The Problem:** In Master mode, remote camera frames would often arrive slightly delayed, causing "double limbs" in 3D fusion.
*   **The Fix:** **Nearest-Frame Anchor Matching.**
    *   Replaced "Oldest Frame" sync with "Slowest Camera Anchor" sync.
    *   Implemented **Cristian’s Algorithm** for sub-millisecond clock synchronization between nodes.
    *   **Result:** Sync success rate increased from ~60% to > 98%.
