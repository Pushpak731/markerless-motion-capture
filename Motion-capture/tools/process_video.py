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
import csv
from statistics import mean, pstdev


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

CSV_ANGLE_FIELDS = [
    'Angle_Elbow_L',
    'Angle_Elbow_R',
    'Angle_Shoulder_L',
    'Angle_Shoulder_R',
    'Angle_Hip_L',
    'Angle_Hip_R',
    'Angle_Knee_L',
    'Angle_Knee_R',
]

CSV_TRACKED_JOINTS = [
    'Shoulder_L',
    'Shoulder_R',
    'Elbow_L',
    'Elbow_R',
    'Wrist_L',
    'Wrist_R',
    'Hip_L',
    'Hip_R',
    'Knee_L',
    'Knee_R',
    'Ankle_L',
    'Ankle_R',
]


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
    from src.frame_quality import FrameQualityAnalyzer, FrameQuality, clone_landmarks, interpolate_landmarks, make_pose_output
    from src.kinematics import KinematicsTracker
    from src.pose_corrector import PoseCorrector
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
    detector = MocapDetector(enable_face=False, enable_hand=False)
    detector.set_imaging_params(enable_face=False, enable_hand=False, enable_roi=False)
    corrector = PoseCorrector()
    visualizer = Visualizer()
    bone_tracker = BoneLengthTracker()
    kinematics_tracker = KinematicsTracker()

    stats = {
        'input_path': input_path,
        'output_path': output_path,
        'fps': fps,
        'frames_seen': 0,
        'frames_annotated': 0,
        'pose_frames': 0,
        'usable_pose_frames': 0,
        'face_frames': 0,
        'hand_frames': 0,
        'mean_consistency_score': 0.0,
        'min_consistency_score': 0.0,
        'p50_consistency_score': 0.0,
        'p90_consistency_score': 0.0,
        'stddev_consistency_score': 0.0,
        'pose_coverage': 0.0,
        'detected_pose_coverage': 0.0,
        'usable_pose_coverage': 0.0,
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
        'longest_missing_pose_streak': 0,
        'longest_unusable_pose_streak': 0,
        'interpolated_frame_count': 0,
        'tracking_loss_frame_count': 0,
        'low_confidence_frame_count': 0,
        'partial_pose_frame_count': 0,
        'unstable_frame_count': 0,
        'missing_frame_count': 0,
        'input_quality_warnings': [],
        'recommendations': [],
    }

    consistency_scores = []
    abs_norm_deviation = []
    final_variance_map = {}
    final_stddev_map = {}
    started = time.perf_counter()
    frame_idx = 0
    analyzer = FrameQualityAnalyzer()
    max_interpolation_gap = 5
    pending_missing_frames = []
    long_gap_active = False
    asymmetry_deltas = {
        'arms': [],
        'legs': []
    }
    high_jitter_frames = 0

    # --- Raw 3D Node Export (for local Panda3D avatar studio) ---
    raw_nodes_csv_path = os.path.splitext(output_path)[0] + '_raw_3d_nodes.csv'
    raw_nodes_file = open(raw_nodes_csv_path, 'w', newline='')
    raw_nodes_writer = csv.writer(raw_nodes_file)
    _raw_nodes_header = ['frame_idx', 'timestamp_ms', 'root_x', 'root_y', 'root_z']
    MEDIAPIPE_JOINT_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]
    for jname in MEDIAPIPE_JOINT_NAMES:
        _raw_nodes_header += [f'{jname}_x', f'{jname}_y', f'{jname}_z', f'{jname}_v']
    raw_nodes_writer.writerow(_raw_nodes_header)
    last_detected_pose = None
    last_detected_world = None

    # prepare CSV metrics file next to annotated output
    metrics_csv_path = os.path.splitext(output_path)[0] + "_metrics.csv"
    csv_file = open(metrics_csv_path, "w", newline='')
    csv_writer = csv.writer(csv_file)
    csv_header = [
        'frame_idx',
        'timestamp_ms',
        'processing_time_s',
        'frame_state',
        'pose_detected',
        'pose_usable',
        'quality_reason',
        'quality_score',
        'pose_visibility_mean',
        'pose_visibility_min',
        'pose_low_visibility_count',
        'visible_joint_ratio',
        'motion_jump',
        'bbox_area_ratio',
        'consistency_score',
        'mean_abs_normalized_deviation',
        'bone_variance_mean',
        'bone_stddev_mean',
        'pose_coverage',
        'detected_pose_coverage',
        'usable_pose_coverage',
        'source_name',
        'stable_frame_used',
        'correction_metadata',
    ]
    for bone_name in CSV_BONE_FIELDS:
        csv_header.extend([
            f'Length_{bone_name}',
            f'Normalized_{bone_name}',
            f'Variance_{bone_name}',
            f'StdDev_{bone_name}',
        ])
    # Reference lengths exported for debugging reference updates
    for bone_name in CSV_BONE_FIELDS:
        csv_header.append(f'Reference_Length_{bone_name}')
    csv_header.extend(CSV_ANGLE_FIELDS)
    for angle_name in CSV_ANGLE_FIELDS:
        csv_header.extend([
            f'Velocity_{angle_name}',
            f'Acceleration_{angle_name}',
        ])
    for joint_name in CSV_TRACKED_JOINTS:
        csv_header.append(f'Velocity_{joint_name}')
        csv_header.extend([
            f'Velocity_{joint_name}_X',
            f'Velocity_{joint_name}_Y',
            f'Velocity_{joint_name}_Z',
            f'Acceleration_{joint_name}',
            f'Acceleration_{joint_name}_X',
            f'Acceleration_{joint_name}_Y',
            f'Acceleration_{joint_name}_Z',
        ])
    csv_header.extend([
        'Jerk_Wrist_L',
        'Jerk_Wrist_R',
        'Jerk_Ankle_L',
        'Jerk_Ankle_R',
        'Coordinate_Space',
        'Kinematics_Smoothing_Enabled',
        'Kinematics_Smoothing_Method',
        'Kinematics_Smoothing_Alpha',
    ])
    for landmark_idx in range(33):
        csv_header.append(f'visibility_{landmark_idx}')
    csv_writer.writerow(csv_header)

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

    def _landmarks_from_pose_object(pose_obj):
        pose_lm = []
        world_lm = None
        if pose_obj and getattr(pose_obj, 'pose_landmarks', None):
            pose_lm = [
                {'x': lm.x, 'y': lm.y, 'z': getattr(lm, 'z', 0.0), 'v': getattr(lm, 'visibility', 1.0)}
                for lm in pose_obj.pose_landmarks[0]
            ]
            if getattr(pose_obj, 'pose_world_landmarks', None):
                world_lm = [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z, 'v': getattr(lm, 'visibility', 1.0)}
                    for lm in pose_obj.pose_world_landmarks[0]
                ]
        return pose_lm, world_lm

    def _pose_visibility_metrics(pose_obj):
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
        return pose_vis_mean, pose_vis_min, low_visibility_count

    def _build_render_results(pose_lm, world_lm, base_results):
        if pose_lm:
            return {
                'pose': make_pose_output(pose_lm, world_lm),
                'face': base_results.get('face') if base_results else None,
                'hand': base_results.get('hand') if base_results else None,
            }
        return {
            'pose': None,
            'face': base_results.get('face') if base_results else None,
            'hand': base_results.get('hand') if base_results else None,
        }

    def _weighted_mean_from_map(value_map):
        if not value_map:
            return 0.0
        weighted_total = 0.0
        weight_sum = 0.0
        for key, value in value_map.items():
            bone_name = key.replace('Bone_Length_Variance_', '').replace('Bone_Length_StdDev_', '')
            weight = BoneLengthTracker._bone_weight(bone_name)
            weighted_total += float(value) * weight
            weight_sum += weight
        if weight_sum <= 0:
            return 0.0
        return weighted_total / weight_sum

    def _emit_frame(frame_idx_local, timestamp_local, frame_image, base_results, pose_lm, world_lm, quality, frame_state):
        nonlocal last_detected_pose, last_detected_world, final_variance_map, final_stddev_map, high_jitter_frames, asymmetry_deltas

        pose_obj = base_results.get('pose') if base_results else None
        pose_vis_mean, pose_vis_min, low_visibility_count = _pose_visibility_metrics(pose_obj)
        if frame_state != 'detected':
            pose_vis_mean = 0.0
            pose_vis_min = 0.0
            low_visibility_count = 0

        if pose_lm:
            bone_result = bone_tracker.process(
                pose_lm,
                world_lm,
                update_reference=bool(getattr(quality, 'pose_usable', False)),
            )
        else:
            bone_result = {
                'consistency_score': 0.0,
                'raw_lengths': {},
                'normalized_lengths': {},
                'variance_lengths': {},
                'stddev_lengths': {},
            }

        kinematic_landmarks = bone_result.get('smoothed_landmarks') if bone_result else None
        if not kinematic_landmarks:
            kinematic_landmarks = world_lm if world_lm else pose_lm
        coordinate_space = bone_result.get('source_name', 'world' if world_lm else 'normalized') if bone_result else (
            'world' if world_lm else 'normalized'
        )
        angle_metrics = Calculations.get_joint_angles(kinematic_landmarks) if kinematic_landmarks else {}
        extended_kinematics = {}
        if kinematic_landmarks:
            extended_kinematics = kinematics_tracker.process(
                kinematic_landmarks,
                angle_metrics,
                timestamp_local,
                coordinate_space=coordinate_space,
            )

        if getattr(quality, 'pose_usable', False):
            consistency_scores.append(float(bone_result.get('consistency_score', 0.0)))
            norm_values = []
            for value in bone_result.get('normalized_lengths', {}).values():
                try:
                    norm_values.append(float(value))
                except Exception:
                    pass
            if norm_values:
                abs_norm_deviation.extend(abs(v - 1.0) for v in norm_values)

            variance_map = bone_result.get('variance_lengths', {})
            stddev_map = bone_result.get('stddev_lengths', {})
            if variance_map:
                final_variance_map = dict(variance_map)
            if stddev_map:
                final_stddev_map = dict(stddev_map)

        if world_lm and len(world_lm) == 33:
            # Arm asymmetry
            l_arm = bone_result.get('raw_lengths', {}).get('Length_UpperArm_L', 0) + bone_result.get('raw_lengths', {}).get('Length_LowerArm_L', 0)
            r_arm = bone_result.get('raw_lengths', {}).get('Length_UpperArm_R', 0) + bone_result.get('raw_lengths', {}).get('Length_LowerArm_R', 0)
            if l_arm > 0 and r_arm > 0:
                asymmetry_deltas['arms'].append(abs(l_arm - r_arm) / max(l_arm, r_arm, 1e-9))
            
            # Leg asymmetry
            l_leg = bone_result.get('raw_lengths', {}).get('Length_UpperLeg_L', 0) + bone_result.get('raw_lengths', {}).get('Length_LowerLeg_L', 0)
            r_leg = bone_result.get('raw_lengths', {}).get('Length_UpperLeg_R', 0) + bone_result.get('raw_lengths', {}).get('Length_LowerLeg_R', 0)
            if l_leg > 0 and r_leg > 0:
                asymmetry_deltas['legs'].append(abs(l_leg - r_leg) / max(l_leg, r_leg, 1e-9))

        if float(getattr(quality, 'motion_jump', 0.0) or 0.0) > 0.2:
            high_jitter_frames += 1

        render_results = _build_render_results(pose_lm, world_lm, base_results)
        annotated = visualizer.draw_landmarks(frame_image.copy(), render_results)
        annotated = visualizer.draw_fps(annotated)
        cv2.putText(
            annotated,
            f'{frame_state.title()} frame: {frame_idx_local + 1}',
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f'Quality {getattr(quality, "score", 0.0):.3f} | {getattr(quality, "reason", frame_state)}',
            (10, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 136),
            2,
            cv2.LINE_AA,
        )
        if frame_state == 'detected' and bone_result:
            cv2.putText(
                annotated,
                f'Bone consistency: {bone_result.get("consistency_score", 0.0):.3f}',
                (10, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 136),
                2,
                cv2.LINE_AA,
            )
        writer.write(annotated)

        stats['frames_seen'] += 1
        stats['frames_annotated'] += 1
        if frame_state == 'detected':
            stats['pose_frames'] += 1
            if getattr(quality, 'pose_usable', False):
                stats['usable_pose_frames'] += 1

        detected_pose_coverage = round(stats['pose_frames'] / max(stats['frames_seen'], 1), 4)
        usable_pose_coverage = round(stats['usable_pose_frames'] / max(stats['frames_seen'], 1), 4)

        mean_abs_dev = float(bone_result.get('mean_abs_normalized_deviation', 0.0)) if getattr(quality, 'pose_usable', False) else 0.0
        bone_var_mean = _weighted_mean_from_map(bone_result.get('variance_lengths', {})) if getattr(quality, 'pose_usable', False) else 0.0
        bone_std_mean = _weighted_mean_from_map(bone_result.get('stddev_lengths', {})) if getattr(quality, 'pose_usable', False) else 0.0
        pose_coverage = detected_pose_coverage
        row = [
            frame_idx_local,
            timestamp_local,
            round(time.perf_counter() - started, 6),
            frame_state,
            int(bool(getattr(quality, 'pose_detected', False))),
            int(bool(getattr(quality, 'pose_usable', False))),
            getattr(quality, 'reason', frame_state),
            float(getattr(quality, 'score', 0.0) or 0.0),
            pose_vis_mean,
            pose_vis_min,
            low_visibility_count,
            float(getattr(quality, 'visible_joint_ratio', 0.0) or 0.0),
            float(getattr(quality, 'motion_jump', 0.0) or 0.0),
            float(getattr(quality, 'bbox_area_ratio', 0.0) or 0.0),
            float(bone_result.get('consistency_score', 0.0)) if bone_result else 0.0,
            mean_abs_dev,
            bone_var_mean,
            bone_std_mean,
            pose_coverage,
            detected_pose_coverage,
            usable_pose_coverage,
            bone_result.get('source_name', '') if bone_result else '',
            bone_result.get('stable_frame_used', False) if bone_result else False,
            json.dumps((base_results.get('pose_correction_metadata', {}).get('normalized') or [{}])[0]) if base_results else '{}',
        ]
        variance_map = bone_result.get('variance_lengths', {}) if bone_result else {}
        stddev_map = bone_result.get('stddev_lengths', {}) if bone_result else {}
        for bone_name in CSV_BONE_FIELDS:
            row.extend([
                float(bone_result.get('raw_lengths', {}).get(f'Length_{bone_name}', 0.0)) if bone_result else 0.0,
                float(bone_result.get('normalized_lengths', {}).get(f'Normalized_{bone_name}', 0.0)) if bone_result else 0.0,
                float(variance_map.get(f'Bone_Length_Variance_{bone_name}', 0.0)) if variance_map else 0.0,
                float(stddev_map.get(f'Bone_Length_StdDev_{bone_name}', 0.0)) if stddev_map else 0.0,
            ])
        # Append reference lengths for debugging whether reference was set/updated
        ref_map = bone_result.get('reference_lengths', {}) if bone_result else {}
        for bone_name in CSV_BONE_FIELDS:
            row.append(float(ref_map.get(f'Reference_Length_{bone_name}', 0.0)) if ref_map else 0.0)
        for angle_name in CSV_ANGLE_FIELDS:
            row.append(float(angle_metrics.get(angle_name, 0.0)))
        for angle_name in CSV_ANGLE_FIELDS:
            row.extend([
                float(extended_kinematics.get(f'Velocity_{angle_name}', 0.0)),
                float(extended_kinematics.get(f'Acceleration_{angle_name}', 0.0)),
            ])
        for joint_name in CSV_TRACKED_JOINTS:
            row.append(float(extended_kinematics.get(f'Velocity_{joint_name}', 0.0)))
            row.extend([
                float(extended_kinematics.get(f'Velocity_{joint_name}_X', 0.0)),
                float(extended_kinematics.get(f'Velocity_{joint_name}_Y', 0.0)),
                float(extended_kinematics.get(f'Velocity_{joint_name}_Z', 0.0)),
                float(extended_kinematics.get(f'Acceleration_{joint_name}', 0.0)),
                float(extended_kinematics.get(f'Acceleration_{joint_name}_X', 0.0)),
                float(extended_kinematics.get(f'Acceleration_{joint_name}_Y', 0.0)),
                float(extended_kinematics.get(f'Acceleration_{joint_name}_Z', 0.0)),
            ])
        row.extend([
            float(extended_kinematics.get('Jerk_Wrist_L', 0.0)),
            float(extended_kinematics.get('Jerk_Wrist_R', 0.0)),
            float(extended_kinematics.get('Jerk_Ankle_L', 0.0)),
            float(extended_kinematics.get('Jerk_Ankle_R', 0.0)),
            extended_kinematics.get('Coordinate_Space', coordinate_space),
            bool(extended_kinematics.get('Kinematics_Smoothing_Enabled', False)),
            extended_kinematics.get('Kinematics_Smoothing_Method', ''),
            float(extended_kinematics.get('Kinematics_Smoothing_Alpha', 0.0)),
        ])
        vis_source = kinematic_landmarks or []
        for landmark_idx in range(33):
            if landmark_idx < len(vis_source):
                item = vis_source[landmark_idx]
                row.append(float(item.get('v', item.get('visibility', 0.0))) if isinstance(item, dict) else 0.0)
            else:
                row.append(0.0)
        csv_writer.writerow(row)

        # --- Write raw 3D world node coordinates ---
        # Calculate normalized root (center of hips) for translation
        root_x, root_y, root_z = 0.0, 0.0, 0.0
        if pose_lm and len(pose_lm) >= 25: # At least up to hips
            root_x = (pose_lm[23]['x'] + pose_lm[24]['x']) / 2.0
            root_y = (pose_lm[23]['y'] + pose_lm[24]['y']) / 2.0
            root_z = (pose_lm[23]['z'] + pose_lm[24]['z']) / 2.0

        raw_row = [frame_idx_local, timestamp_local, round(root_x, 6), round(root_y, 6), round(root_z, 6)]
        if world_lm and len(world_lm) == 33:
            for lm in world_lm:
                raw_row += [round(lm['x'], 6), round(lm['y'], 6), round(lm['z'], 6), round(lm.get('v', 1.0), 4)]
        else:
            raw_row += [0.0] * (33 * 4)
        raw_nodes_writer.writerow(raw_row)

        return bone_result, render_results, detected_pose_coverage, usable_pose_coverage


    def _emit_detected_frame(frame_idx_local, timestamp_local, frame_image, results_obj, quality_obj, pose_lm, world_lm):
        nonlocal last_detected_pose, last_detected_world
        bone_result, _, _, _ = _emit_frame(
            frame_idx_local,
            timestamp_local,
            frame_image,
            results_obj,
            pose_lm,
            world_lm,
            quality_obj,
            'detected',
        )
        if pose_lm:
            last_detected_pose = clone_landmarks(pose_lm)
            last_detected_world = clone_landmarks(world_lm) if world_lm else None
        return bone_result

    def _flush_missing_buffer(current_frame=None, current_results=None, current_quality=None):
        nonlocal pending_missing_frames, long_gap_active
        if not pending_missing_frames:
            long_gap_active = False
            return

        if long_gap_active or current_frame is None or not last_detected_pose:
            for buffered in pending_missing_frames:
                missing_quality = FrameQuality(pose_detected=False, pose_usable=False, reason='tracking_loss', score=0.0)
                _emit_frame(
                    buffered['frame_idx'],
                    buffered['timestamp_ms'],
                    buffered['frame'],
                    buffered.get('results') or {},
                    clone_landmarks(last_detected_pose) if last_detected_pose else [],
                    clone_landmarks(last_detected_world) if last_detected_world else None,
                    missing_quality,
                    'tracking_loss',
                )
                stats['tracking_loss_frame_count'] += 1
            pending_missing_frames = []
            long_gap_active = False
            return

        start_pose = clone_landmarks(last_detected_pose)
        start_world = clone_landmarks(last_detected_world) if last_detected_world else None
        end_pose, end_world = current_frame
        buffered_count = len(pending_missing_frames)
        if buffered_count <= max_interpolation_gap:
            for offset, buffered in enumerate(pending_missing_frames, start=1):
                fraction = offset / float(buffered_count + 1)
                interpolated_pose = interpolate_landmarks(start_pose, end_pose, fraction)
                interpolated_world = interpolate_landmarks(start_world, end_world, fraction) if start_world and end_world else None
                interpolated_quality = FrameQuality(
                    pose_detected=False,
                    pose_usable=False,
                    reason='interpolated',
                    score=0.0,
                )
                _emit_frame(
                    buffered['frame_idx'],
                    buffered['timestamp_ms'],
                    buffered['frame'],
                    buffered.get('results') or {},
                    interpolated_pose,
                    interpolated_world,
                    interpolated_quality,
                    'interpolated',
                )
                stats['interpolated_frame_count'] += 1
        else:
            for buffered in pending_missing_frames:
                missing_quality = FrameQuality(pose_detected=False, pose_usable=False, reason='tracking_loss', score=0.0)
                _emit_frame(
                    buffered['frame_idx'],
                    buffered['timestamp_ms'],
                    buffered['frame'],
                    buffered.get('results') or {},
                    start_pose,
                    start_world,
                    missing_quality,
                    'tracking_loss',
                )
                stats['tracking_loss_frame_count'] += 1
        pending_missing_frames = []
        long_gap_active = False

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
            pose_lm, world_lm = _landmarks_from_pose_object(pose_obj)
            quality = analyzer.analyze(pose_lm, width, height)
            results = corrector.process(results, timestamp_ms=timestamp_ms, frame_quality=quality)
            pose_obj = results.get('pose')
            pose_lm, world_lm = _landmarks_from_pose_object(pose_obj)
            if results.get('face') and results['face'].face_landmarks:
                stats['face_frames'] += 1
            if results.get('hand') and results['hand'].hand_landmarks:
                stats['hand_frames'] += 1

            if not quality.pose_detected:
                if long_gap_active:
                    missing_quality = FrameQuality(pose_detected=False, pose_usable=False, reason='tracking_loss', score=0.0)
                    _emit_frame(
                        frame_idx,
                        timestamp_ms,
                        frame,
                        results,
                        clone_landmarks(last_detected_pose) if last_detected_pose else [],
                        clone_landmarks(last_detected_world) if last_detected_world else None,
                        missing_quality,
                        'tracking_loss',
                    )
                    stats['tracking_loss_frame_count'] += 1
                else:
                    pending_missing_frames.append({
                        'frame_idx': frame_idx,
                        'timestamp_ms': timestamp_ms,
                        'frame': frame,
                        'results': results,
                    })
                    if len(pending_missing_frames) > max_interpolation_gap:
                        long_gap_active = True
                        _flush_missing_buffer()
                frame_idx += 1
                continue

            if pending_missing_frames:
                _flush_missing_buffer(current_frame=(clone_landmarks(pose_lm), clone_landmarks(world_lm) if world_lm else None), current_results=results, current_quality=quality)

            _emit_detected_frame(frame_idx, timestamp_ms, frame, results, quality, pose_lm, world_lm)
            frame_idx += 1
    finally:
        stats['processing_seconds'] = round(time.perf_counter() - started, 3)

        if consistency_scores:
            stats['mean_consistency_score'] = round(mean(consistency_scores), 4)
            stats['min_consistency_score'] = round(min(consistency_scores), 4)
            stats['p50_consistency_score'] = round(_percentile(consistency_scores, 50), 4)
            stats['p90_consistency_score'] = round(_percentile(consistency_scores, 90), 4)
            stats['stddev_consistency_score'] = round(pstdev(consistency_scores), 4) if len(consistency_scores) > 1 else 0.0

        if pending_missing_frames:
            long_gap_active = True
            _flush_missing_buffer()

        total = max(stats['frames_seen'], 1)
        stats['pose_coverage'] = round(stats['pose_frames'] / total, 4)
        stats['detected_pose_coverage'] = stats['pose_coverage']
        stats['usable_pose_coverage'] = round(stats['usable_pose_frames'] / total, 4)
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

        if final_variance_map:
            variance_values = list(final_variance_map.values())
            stats['bone_variance_mean'] = round(_weighted_mean_from_map(final_variance_map), 6)
            stats['bone_variance_max'] = round(max(float(v) for v in variance_values), 6)

        if final_stddev_map:
            stddev_values = list(final_stddev_map.values())
            stats['bone_stddev_mean'] = round(_weighted_mean_from_map(final_stddev_map), 6)
            stats['bone_stddev_max'] = round(max(float(v) for v in stddev_values), 6)

        stats['asymmetry_analysis'] = {
            'mean_arm_asymmetry': round(mean(asymmetry_deltas['arms']), 4) if asymmetry_deltas['arms'] else 0.0,
            'mean_leg_asymmetry': round(mean(asymmetry_deltas['legs']), 4) if asymmetry_deltas['legs'] else 0.0,
            'max_arm_asymmetry': round(max(asymmetry_deltas['arms']), 4) if asymmetry_deltas['arms'] else 0.0,
            'max_leg_asymmetry': round(max(asymmetry_deltas['legs']), 4) if asymmetry_deltas['legs'] else 0.0,
        }
        stats['jitter_analysis'] = {
            'high_jitter_count': high_jitter_frames,
            'high_jitter_ratio': round(high_jitter_frames / total, 4),
        }

        quality_summary = analyzer.build_summary(
            total_frames=total,
            interpolated_frame_count=stats['interpolated_frame_count'],
            tracking_loss_frame_count=stats['tracking_loss_frame_count'],
        )
        stats.update(quality_summary)

        capture.release()
        writer.release()
        try:
            csv_file.close()
        except Exception:
            pass
        try:
            raw_nodes_file.close()
        except Exception:
            pass

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
