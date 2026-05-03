# !/usr/bin/env python3
# """Offline video verification utility.
# 
# This wrapper reuses the main offline exporter so the annotated output,
# quality scoring, interpolation, and reports stay identical across entry points.
# """
# 
# import argparse
# import json
# import os
# import sys
# 
# 
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
# 
# from process_video import process_video  # noqa: E402
# 
# 
# def main() -> int:
#     parser = argparse.ArgumentParser(
#         description='Run offline mocap verification on a video and export an annotated copy.'
#     )
#     parser.add_argument('--input', required=True, help='Input video file path')
#     parser.add_argument('--output', required=True, help='Annotated output video path')
#     parser.add_argument('--max-frames', type=int, default=0, help='Limit frames processed')
#     parser.add_argument('--summary-json', default='', help='Optional JSON summary output path')
#     args = parser.parse_args()
# 
#     stats = process_video(args.input, args.output, max_frames=args.max_frames)
# 
#     if args.summary_json:
#         with open(args.summary_json, 'w', encoding='utf-8') as handle:
#             json.dump(stats, handle, indent=2)
# 
#     print(json.dumps(stats, indent=2))
#     return 0
# 
# 
# if __name__ == '__main__':
#     raise SystemExit(main())
# !/usr/bin/env python3
# """Offline video verification utility.
# 
# This wrapper reuses the main offline exporter so the annotated output,
# quality scoring, interpolation, and reports stay identical across entry points.
# """
# 
# import argparse
# import json
# import os
# import sys
# 
# 
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
# 
# from process_video import process_video  # noqa: E402
# 
# 
# def main() -> int:
#     parser = argparse.ArgumentParser(
#         description='Run offline mocap verification on a video and export an annotated copy.'
#     )
#     parser.add_argument('--input', required=True, help='Input video file path')
#     parser.add_argument('--output', required=True, help='Annotated output video path')
#     parser.add_argument('--max-frames', type=int, default=0, help='Limit frames processed')
#     parser.add_argument('--summary-json', default='', help='Optional JSON summary output path')
#     args = parser.parse_args()
# 
#     stats = process_video(args.input, args.output, max_frames=args.max_frames)
# 
#     if args.summary_json:
#         with open(args.summary_json, 'w', encoding='utf-8') as handle:
#             json.dump(stats, handle, indent=2)
# 
#     print(json.dumps(stats, indent=2))
#     return 0
# 
# 
# if __name__ == '__main__':
#     raise SystemExit(main())

#!/usr/bin/env python3
"""Offline video verification utility.

This wrapper reuses the main offline exporter so the annotated output,
quality scoring, interpolation, and reports stay identical across entry points.
"""

import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from process_video import process_video  # noqa: E402


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
import csv


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


CSV_BONE_FIELDS = [
    'UpperArm_L',
    'LowerArm_L',
    'UpperArm_R',
    'LowerArm_R',
    'UpperLeg_L',
    'LowerLeg_L',
    'UpperLeg_R',
    'LowerLeg_R',
    'Shoulder',
    'Hip',
]


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

    from src.calculations import BoneLengthTracker  # noqa: E402
    from src.detector import MocapDetector  # noqa: E402
    from src.pose_corrector import PoseCorrector  # noqa: E402
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
    detector = MocapDetector(enable_face=False, enable_hand=False)
    detector.set_imaging_params(enable_face=False, enable_hand=False, enable_roi=False)
    corrector = PoseCorrector()
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
        'processing_seconds': 0.0,
    }

    started = time.perf_counter()
    frame_idx = 0

    metrics_csv_path = os.path.splitext(output_path)[0] + "_metrics.csv"
    csv_file = open(metrics_csv_path, "w", newline='')
    csv_writer = csv.writer(csv_file)
    csv_header = ['frame_idx', 'timestamp_ms', 'pose_detected']
    csv_header.extend(['pose_visibility_mean', 'pose_visibility_min', 'pose_low_visibility_count'])
    for bone_name in CSV_BONE_FIELDS:
        csv_header.extend([
            f'Length_{bone_name}',
            f'Normalized_{bone_name}',
            f'Variance_{bone_name}',
            f'StdDev_{bone_name}',
        ])
    csv_writer.writerow(csv_header)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if max_frames and frame_idx >= max_frames:
                break

            timestamp_ms = int((frame_idx / fps) * 1000.0)
            results = detector.process(frame, timestamp_ms=timestamp_ms)
            results = corrector.process(results, timestamp_ms=timestamp_ms)

            pose_obj = results.get('pose')

            pose_lm = []
            world_lm = None
            if pose_obj and getattr(pose_obj, 'pose_landmarks', None):
                stats['pose_frames'] += 1
                pose_lm = [
                    {'x': lm.x, 'y': lm.y, 'z': getattr(lm, 'z', 0.0), 'v': getattr(lm, 'visibility', 1.0)}
                    for lm in pose_obj.pose_landmarks[0]
                ]
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

            visibilities = []
            low_visibility_count = 0
            if pose_obj and getattr(pose_obj, 'pose_landmarks', None):
                for lm in pose_obj.pose_landmarks[0]:
                    vis = float(getattr(lm, 'visibility', 0.0))
                    visibilities.append(vis)
                    if vis < 0.5:
                        low_visibility_count += 1
            pose_vis_mean = round(sum(visibilities) / len(visibilities), 4) if visibilities else 0.0
            pose_vis_min = round(min(visibilities), 4) if visibilities else 0.0

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

            # write per-frame CSV with per-bone metrics
            try:
                row = [
                    frame_idx,
                    int((frame_idx / fps) * 1000.0),
                    int(bool(results.get('pose'))),
                    pose_vis_mean,
                    pose_vis_min,
                    low_visibility_count,
                ]
                for bone_name in CSV_BONE_FIELDS:
                    row.extend([
                        float(bone_result.get('raw_lengths', {}).get(f'Length_{bone_name}', 0.0)) if bone_result else 0.0,
                        float(bone_result.get('normalized_lengths', {}).get(f'Normalized_{bone_name}', 0.0)) if bone_result else 0.0,
                        float(bone_result.get('variance_lengths', {}).get(f'Bone_Length_Variance_{bone_name}', 0.0)) if bone_result else 0.0,
                        float(bone_result.get('stddev_lengths', {}).get(f'Bone_Length_StdDev_{bone_name}', 0.0)) if bone_result else 0.0,
                    ])
                csv_writer.writerow(row)
            except Exception:
                pass

            stats['frames_seen'] += 1
            stats['frames_annotated'] += 1
            frame_idx += 1
    finally:
        stats['processing_seconds'] = round(time.perf_counter() - started, 3)
        capture.release()
        writer.release()
        try:
            csv_file.close()
        except Exception:
            pass

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