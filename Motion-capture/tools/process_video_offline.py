#!/usr/bin/env python3
"""Offline video verification utility.

Reads an input video, runs the existing mocap detector frame-by-frame, draws
pose/face/hand markers on top of the frames, and writes an annotated output
video. This is intended for reproducible verification of recorded footage.
"""

import argparse
import json
import os
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _pick_writer(path: str, fps: float, width: int, height: int):
    import cv2

    ext = os.path.splitext(path)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*('avc1' if ext in ('.mp4', '.m4v') else 'mp4v'))
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open output video writer: {path}')
    return writer


def process_video(input_path: str, output_path: str, max_frames: int = 0) -> dict:
    import cv2

    from src.detector import MocapDetector  # noqa: E402
    from src.visualizer import Visualizer  # noqa: E402

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

    stats = {
        'input_path': input_path,
        'output_path': output_path,
        'fps': fps,
        'frames_seen': 0,
        'frames_annotated': 0,
        'pose_frames': 0,
        'face_frames': 0,
        'hand_frames': 0,
        'processing_seconds': 0.0,
    }

    started = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if max_frames and frame_idx >= max_frames:
                break

            timestamp_ms = int((frame_idx / fps) * 1000.0)
            results = detector.process(frame, timestamp_ms=timestamp_ms)

            if results.get('pose') and results['pose'].pose_landmarks:
                stats['pose_frames'] += 1
            if results.get('face') and results['face'].face_landmarks:
                stats['face_frames'] += 1
            if results.get('hand') and results['hand'].hand_landmarks:
                stats['hand_frames'] += 1

            annotated = visualizer.draw_landmarks(frame.copy(), results)
            annotated = visualizer.draw_fps(annotated)
            cv2.putText(
                annotated,
                f'Offline frame: {frame_idx + 1}',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)

            stats['frames_seen'] += 1
            stats['frames_annotated'] += 1
            frame_idx += 1
    finally:
        stats['processing_seconds'] = round(time.perf_counter() - started, 3)
        capture.release()
        writer.release()

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run offline mocap verification on a video and export an annotated copy.'
    )
    parser.add_argument('--input', required=True, help='Input video file path')
    parser.add_argument('--output', required=True, help='Annotated output video path')
    parser.add_argument('--max-frames', type=int, default=0, help='Limit frames processed')
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