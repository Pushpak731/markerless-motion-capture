#!/usr/bin/env python3
"""Offline video processing and verification CLI.

Reads a video file, runs the existing detector per frame, applies the shared
bone-length stabilization path, draws the skeleton, and writes an annotated
output video.
"""

import argparse
import json
import os
import sys
import time
from statistics import mean, pstdev


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _pick_writer(path: str, fps: float, width: int, height: int):
    import cv2

    ext = os.path.splitext(path)[1].lower()
    codec_candidates = ['mp4v', 'XVID', 'MJPG'] if ext in ('.mp4', '.m4v') else ['XVID', 'MJPG', 'mp4v']

    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer

    raise RuntimeError(f'Could not open output video writer with codecs {codec_candidates}: {path}')


def process_video(input_path: str, output_path: str, max_frames: int = 0) -> dict:
    import cv2

    from src.calculations import Calculations, BoneLengthTracker
    from src.detector import MocapDetector
    from src.visualizer import Visualizer

    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f'Could not open input video: {input_path}')

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError('Could not determine input video dimensions')

    writer = _pick_writer(output_path, fps, width, height)
    detector = MocapDetector()
    visualizer = Visualizer()
    bone_tracker = BoneLengthTracker()

    stats = {
        'input_path': input_path,
        'output_path': output_path,
        'fps': fps,
        'frames_seen': 0,
        'frames_annotated': 0,
        'pose_frames': 0,
        'face_frames': 0,
        'hand_frames': 0,
        'mean_consistency_score': 0.0,
        'min_consistency_score': 0.0,
        'p50_consistency_score': 0.0,
        'p90_consistency_score': 0.0,
        'stddev_consistency_score': 0.0,
        'pose_coverage': 0.0,
        'face_coverage': 0.0,
        'hand_coverage': 0.0,
        'processing_fps': 0.0,
        'realtime_ratio': 0.0,
        'mean_abs_normalized_deviation': 0.0,
        'max_abs_normalized_deviation': 0.0,
        'p90_abs_normalized_deviation': 0.0,
        'p95_abs_normalized_deviation': 0.0,
        'bone_variance_mean': 0.0,
        'bone_variance_max': 0.0,
        'bone_stddev_mean': 0.0,
        'bone_stddev_max': 0.0,
        'processing_seconds': 0.0,
    }

    consistency_scores = []
    abs_norm_deviation = []
    final_variance_values = []
    final_stddev_values = []
    started = time.perf_counter()
    frame_idx = 0

    def _percentile(values, pct):
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        ordered = sorted(float(v) for v in values)
        pos = (len(ordered) - 1) * (pct / 100.0)
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        frac = pos - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if max_frames and frame_idx >= max_frames:
                break

            timestamp_ms = int((frame_idx / fps) * 1000.0)
            results = detector.process(frame, timestamp_ms=timestamp_ms)

            pose_obj = results.get('pose')
            world_lm = None
            pose_lm = []
            if pose_obj and getattr(pose_obj, 'pose_landmarks', None):
                pose_lm = [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z, 'v': getattr(lm, 'visibility', 1.0)}
                    for lm in pose_obj.pose_landmarks[0]
                ]
                stats['pose_frames'] += 1
                if getattr(pose_obj, 'pose_world_landmarks', None):
                    world_lm = [
                        {'x': lm.x, 'y': lm.y, 'z': lm.z, 'v': getattr(lm, 'visibility', 1.0)}
                        for lm in pose_obj.pose_world_landmarks[0]
                    ]

            if results.get('face') and results['face'].face_landmarks:
                stats['face_frames'] += 1
            if results.get('hand') and results['hand'].hand_landmarks:
                stats['hand_frames'] += 1

            bone_result = bone_tracker.process(pose_lm, world_lm)
            consistency_scores.append(float(bone_result.get('consistency_score', 0.0)))

            norm_values = []
            for value in bone_result.get('normalized_lengths', {}).values():
                try:
                    norm_values.append(float(value))
                except Exception:
                    pass
            if norm_values and pose_lm:
                frame_abs_dev = [abs(v - 1.0) for v in norm_values]
                abs_norm_deviation.extend(frame_abs_dev)

            variance_values = []
            for value in bone_result.get('variance_lengths', {}).values():
                try:
                    variance_values.append(float(value))
                except Exception:
                    pass
            if variance_values:
                final_variance_values = variance_values

            stddev_values = []
            for value in bone_result.get('stddev_lengths', {}).values():
                try:
                    stddev_values.append(float(value))
                except Exception:
                    pass
            if stddev_values:
                final_stddev_values = stddev_values

            annotated = visualizer.draw_landmarks(frame.copy(), results)
            annotated = visualizer.draw_fps(annotated)
            cv2.putText(
                annotated,
                f'Frame {frame_idx + 1}',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f'Bone consistency: {bone_result.get("consistency_score", 0.0):.3f}',
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 136),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)

            stats['frames_seen'] += 1
            stats['frames_annotated'] += 1
            frame_idx += 1
    finally:
        stats['processing_seconds'] = round(time.perf_counter() - started, 3)

        if consistency_scores:
            stats['mean_consistency_score'] = round(mean(consistency_scores), 4)
            stats['min_consistency_score'] = round(min(consistency_scores), 4)
            stats['p50_consistency_score'] = round(_percentile(consistency_scores, 50), 4)
            stats['p90_consistency_score'] = round(_percentile(consistency_scores, 90), 4)
            stats['stddev_consistency_score'] = round(pstdev(consistency_scores), 4) if len(consistency_scores) > 1 else 0.0

        total = max(stats['frames_seen'], 1)
        stats['pose_coverage'] = round(stats['pose_frames'] / total, 4)
        stats['face_coverage'] = round(stats['face_frames'] / total, 4)
        stats['hand_coverage'] = round(stats['hand_frames'] / total, 4)

        elapsed = max(stats['processing_seconds'], 1e-9)
        stats['processing_fps'] = round(stats['frames_seen'] / elapsed, 3)
        stats['realtime_ratio'] = round((stats['processing_fps'] / fps), 4) if fps > 0 else 0.0

        if abs_norm_deviation:
            stats['mean_abs_normalized_deviation'] = round(mean(abs_norm_deviation), 4)
            stats['max_abs_normalized_deviation'] = round(max(abs_norm_deviation), 4)
            stats['p90_abs_normalized_deviation'] = round(_percentile(abs_norm_deviation, 90), 4)
            stats['p95_abs_normalized_deviation'] = round(_percentile(abs_norm_deviation, 95), 4)

        if final_variance_values:
            stats['bone_variance_mean'] = round(mean(final_variance_values), 6)
            stats['bone_variance_max'] = round(max(final_variance_values), 6)

        if final_stddev_values:
            stats['bone_stddev_mean'] = round(mean(final_stddev_values), 6)
            stats['bone_stddev_max'] = round(max(final_stddev_values), 6)

        capture.release()
        writer.release()

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description='Process a video file and export an annotated mocap copy.')
    parser.add_argument('--input', required=True, help='Input video file path')
    parser.add_argument('--output', required=True, help='Annotated output video path')
    parser.add_argument('--max-frames', type=int, default=0, help='Optional frame cap for quicker verification')
    parser.add_argument('--summary-json', default='', help='Optional JSON summary output path')
    args = parser.parse_args()

    stats = process_video(args.input, args.output, max_frames=args.max_frames)

    if args.summary_json:
        with open(args.summary_json, 'w', encoding='utf-8') as handle:
            json.dump(stats, handle, indent=2)

    print(json.dumps(stats, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())