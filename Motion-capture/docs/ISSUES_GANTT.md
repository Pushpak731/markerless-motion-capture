# Stabilization Status and Verification Plan

As of 2026-05-01, the main cleanup work is implemented: bone-length stabilization, offline video verification, and source generalization are all in place. The remaining work is practical validation on a real sample video and a live capture session.

```mermaid
gantt
    title Motion Capture Issue Recovery Plan
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Findings
    Bone-length scale instability     :crit, a1, 2026-05-01, 2d
    Raw vs normalized metric mismatch  :crit, a2, after a1, 2d
    Misleading GUI labels              :a3, after a2, 1d
    Offline verifier missing           :crit, a4, 2026-05-01, 2d
    Source abstraction too narrow      :crit, a5, 2026-05-01, 1d

    section Fixes
    Prefer world-space metrics         :b1, after a1, 2d
    Add offline video annotation tool  :b2, after a4, 3d
    Generalize capture source          :b3, after a5, 2d
    Wire video timestamps by frame     :b4, after b2, 1d
    Add summary output + smoke test    :b5, after b2, 2d

    section Validation
    Run static jitter regression       :c1, after b1, 1d
    Run known-angle comparison         :c2, after b1, 1d
    Verify offline annotated export    :c3, after b4, 1d
    Update docs and run end-to-end     :c4, after c3, 1d
```

## Notes

- Length metrics now come from the stabilized tracker rather than a frame-varying normalization path.
- The offline verifier is available as a CLI tool that annotates uploaded or recorded videos with pose markers.
- Remaining work is validation against a sample clip and one live session, then tuning any residual thresholds if needed.