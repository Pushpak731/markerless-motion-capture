from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from statistics import mean
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_VISIBLE_THRESHOLD = 0.5
DEFAULT_USABLE_SCORE_THRESHOLD = 0.65
DEFAULT_PARTIAL_RATIO_THRESHOLD = 0.75
DEFAULT_UNSTABLE_JUMP_THRESHOLD = 0.22
DEFAULT_LOW_CONFIDENCE_SCORE_THRESHOLD = 0.45
DEFAULT_BBOX_TARGET_RATIO = 0.08


@dataclass
class FrameQuality:
    pose_detected: bool = False
    pose_usable: bool = False
    score: float = 0.0
    reason: str = 'missing'
    mean_visibility: float = 0.0
    visible_joint_ratio: float = 0.0
    motion_jump: float = 0.0
    bbox_area_ratio: float = 0.0
    visible_joint_count: int = 0
    joint_visibility: Dict[int, float] = field(default_factory=dict)


def _value_from_landmark(landmark: Any, key: str, default: float = 0.0) -> float:
    if landmark is None:
        return float(default)
    if isinstance(landmark, dict):
        return float(landmark.get(key, default))
    return float(getattr(landmark, key, default))


def _landmark_to_dict(landmark: Any) -> Dict[str, float]:
    return {
        'x': _value_from_landmark(landmark, 'x'),
        'y': _value_from_landmark(landmark, 'y'),
        'z': _value_from_landmark(landmark, 'z'),
        'v': _value_from_landmark(landmark, 'visibility', _value_from_landmark(landmark, 'v', 1.0)),
    }


def _landmark_sequence(pose_landmarks: Any) -> List[Dict[str, float]]:
    if not pose_landmarks:
        return []
    if isinstance(pose_landmarks, dict):
        ordered: List[Dict[str, float]] = []
        for idx in sorted(int(key) for key in pose_landmarks.keys()):
            ordered.append(_landmark_to_dict(pose_landmarks[idx]))
        return ordered

    sequence: List[Dict[str, float]] = []
    for landmark in pose_landmarks:
        sequence.append(_landmark_to_dict(landmark))
    return sequence


def clone_landmarks(landmarks: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    return [dict(item) for item in landmarks]


def interpolate_landmarks(
    start_landmarks: Sequence[Dict[str, float]],
    end_landmarks: Sequence[Dict[str, float]],
    fraction: float,
) -> List[Dict[str, float]]:
    fraction = max(0.0, min(1.0, float(fraction)))
    count = max(len(start_landmarks), len(end_landmarks))
    output: List[Dict[str, float]] = []
    for idx in range(count):
        start = start_landmarks[idx] if idx < len(start_landmarks) else None
        end = end_landmarks[idx] if idx < len(end_landmarks) else None
        if start is None and end is None:
            output.append({'x': 0.0, 'y': 0.0, 'z': 0.0, 'v': 0.0})
            continue
        if start is None:
            output.append(dict(end))
            continue
        if end is None:
            output.append(dict(start))
            continue
        output.append({
            'x': start['x'] + (end['x'] - start['x']) * fraction,
            'y': start['y'] + (end['y'] - start['y']) * fraction,
            'z': start['z'] + (end['z'] - start['z']) * fraction,
            'v': start.get('v', 1.0) + (end.get('v', 1.0) - start.get('v', 1.0)) * fraction,
        })
    return output


def make_pose_output(
    pose_landmarks: Optional[Sequence[Dict[str, float]]],
    world_landmarks: Optional[Sequence[Dict[str, float]]] = None,
) -> SimpleNamespace:
    pose_list = [SimpleNamespace(**landmark) for landmark in pose_landmarks] if pose_landmarks else None
    world_list = [SimpleNamespace(**landmark) for landmark in world_landmarks] if world_landmarks else None
    return SimpleNamespace(pose_landmarks=[pose_list] if pose_list else None, pose_world_landmarks=[world_list] if world_list else None)


class FrameQualityAnalyzer:
    def __init__(
        self,
        visible_threshold: float = DEFAULT_VISIBLE_THRESHOLD,
        usable_score_threshold: float = DEFAULT_USABLE_SCORE_THRESHOLD,
        partial_ratio_threshold: float = DEFAULT_PARTIAL_RATIO_THRESHOLD,
        unstable_jump_threshold: float = DEFAULT_UNSTABLE_JUMP_THRESHOLD,
        low_confidence_score_threshold: float = DEFAULT_LOW_CONFIDENCE_SCORE_THRESHOLD,
        bbox_target_ratio: float = DEFAULT_BBOX_TARGET_RATIO,
    ):
        self.visible_threshold = float(visible_threshold)
        self.usable_score_threshold = float(usable_score_threshold)
        self.partial_ratio_threshold = float(partial_ratio_threshold)
        self.unstable_jump_threshold = float(unstable_jump_threshold)
        self.low_confidence_score_threshold = float(low_confidence_score_threshold)
        self.bbox_target_ratio = float(bbox_target_ratio)

        self.frames_seen = 0
        self.detected_frames = 0
        self.usable_frames = 0
        self.reason_counts: Dict[str, int] = {'missing': 0, 'low_confidence': 0, 'partial_pose': 0, 'unstable': 0, 'valid': 0}
        self.longest_missing_streak = 0
        self.longest_unusable_streak = 0
        self._missing_streak = 0
        self._unusable_streak = 0
        self._joint_seen_counts = [0] * 33
        self._joint_dropout_counts = [0] * 33
        self._previous_detected_landmarks: List[Dict[str, float]] = []
        self._previous_bbox_ratio = 0.0

    @staticmethod
    def _bbox_area_ratio(landmarks: Sequence[Dict[str, float]], frame_width: int, frame_height: int) -> float:
        xs = [float(item['x']) for item in landmarks]
        ys = [float(item['y']) for item in landmarks]
        if not xs or not ys:
            return 0.0
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        area = max(0.0, width * height)
        if area <= 0.0:
            return 0.0
        if max(xs) > 1.5 or max(ys) > 1.5 or (frame_width > 1 and frame_height > 1 and area > 1.5):
            frame_area = max(float(frame_width * frame_height), 1.0)
            return area / frame_area
        return min(1.0, area)

    @staticmethod
    def _normalized_motion_jump(
        current_landmarks: Sequence[Dict[str, float]],
        previous_landmarks: Sequence[Dict[str, float]],
        current_bbox_ratio: float,
        previous_bbox_ratio: float,
        visible_threshold: float,
    ) -> float:
        if not current_landmarks or not previous_landmarks:
            return 0.0
        count = min(len(current_landmarks), len(previous_landmarks))
        if count == 0:
            return 0.0
        distances: List[float] = []
        for idx in range(count):
            current = current_landmarks[idx]
            previous = previous_landmarks[idx]
            if current.get('v', 1.0) < visible_threshold and previous.get('v', 1.0) < visible_threshold:
                continue
            dx = float(current['x']) - float(previous['x'])
            dy = float(current['y']) - float(previous['y'])
            dz = float(current['z']) - float(previous['z'])
            distances.append(sqrt(dx * dx + dy * dy + dz * dz))
        if not distances:
            return 0.0
        motion = mean(distances)
        scale = max(current_bbox_ratio, previous_bbox_ratio, 1e-3) ** 0.5
        return motion / scale

    def analyze(self, pose_landmarks: Any, frame_width: int, frame_height: int) -> FrameQuality:
        self.frames_seen += 1
        landmarks = _landmark_sequence(pose_landmarks)

        if not landmarks or len(landmarks) < 33:
            self.reason_counts['missing'] += 1
            self._missing_streak += 1
            self._unusable_streak = 0
            self.longest_missing_streak = max(self.longest_missing_streak, self._missing_streak)
            return FrameQuality(reason='missing')

        self.detected_frames += 1
        visibilities = [float(item.get('v', 1.0)) for item in landmarks[:33]]
        visible_joint_count = sum(1 for value in visibilities if value >= self.visible_threshold)
        mean_visibility = mean(visibilities) if visibilities else 0.0
        visible_joint_ratio = visible_joint_count / 33.0
        bbox_area_ratio = self._bbox_area_ratio(landmarks[:33], frame_width, frame_height)
        motion_jump = self._normalized_motion_jump(
            landmarks[:33],
            self._previous_detected_landmarks,
            bbox_area_ratio,
            self._previous_bbox_ratio,
            self.visible_threshold,
        )

        bbox_score = 1.0 - min(1.0, abs(bbox_area_ratio - self.bbox_target_ratio) / max(self.bbox_target_ratio, 1e-6))
        motion_score = 1.0 - min(1.0, motion_jump / max(self.unstable_jump_threshold, 1e-6))
        score = (
            0.38 * mean_visibility +
            0.27 * visible_joint_ratio +
            0.20 * bbox_score +
            0.15 * motion_score
        )

        if mean_visibility < self.low_confidence_score_threshold or score < self.low_confidence_score_threshold:
            reason = 'low_confidence'
        elif visible_joint_ratio < self.partial_ratio_threshold:
            reason = 'partial_pose'
        elif motion_jump > self.unstable_jump_threshold:
            reason = 'unstable'
        else:
            reason = 'valid'

        pose_usable = reason == 'valid' and score >= self.usable_score_threshold
        if pose_usable:
            self.usable_frames += 1
            self.reason_counts['valid'] += 1
            self._missing_streak = 0
            self._unusable_streak = 0
        else:
            self.reason_counts[reason] += 1
            self._missing_streak = 0
            self._unusable_streak += 1
            self.longest_unusable_streak = max(self.longest_unusable_streak, self._unusable_streak)

        self.longest_missing_streak = max(self.longest_missing_streak, self._missing_streak)
        for idx, value in enumerate(visibilities[:33]):
            self._joint_seen_counts[idx] += 1
            if value < self.visible_threshold:
                self._joint_dropout_counts[idx] += 1

        self._previous_detected_landmarks = clone_landmarks(landmarks[:33])
        self._previous_bbox_ratio = bbox_area_ratio

        return FrameQuality(
            pose_detected=True,
            pose_usable=pose_usable,
            score=round(score, 4),
            reason=reason,
            mean_visibility=round(mean_visibility, 4),
            visible_joint_ratio=round(visible_joint_ratio, 4),
            motion_jump=round(motion_jump, 4),
            bbox_area_ratio=round(bbox_area_ratio, 4),
            visible_joint_count=visible_joint_count,
            joint_visibility={idx: round(value, 4) for idx, value in enumerate(visibilities[:33])},
        )

    def joint_dropout_rates(self, total_frames: Optional[int] = None) -> Dict[int, float]:
        denominator = float(total_frames or self.frames_seen or 1)
        return {
            idx: round(count / denominator, 4)
            for idx, count in enumerate(self._joint_dropout_counts)
        }

    def build_summary(
        self,
        total_frames: int,
        interpolated_frame_count: int = 0,
        tracking_loss_frame_count: int = 0,
    ) -> Dict[str, Any]:
        total_frames = int(total_frames or self.frames_seen or 0)
        detected_pose_coverage = round((self.detected_frames / total_frames), 4) if total_frames > 0 else 0.0
        usable_pose_coverage = round((self.usable_frames / total_frames), 4) if total_frames > 0 else 0.0
        dropout_rates = self.joint_dropout_rates(total_frames=total_frames)
        max_dropout = max(dropout_rates.values()) if dropout_rates else 0.0

        warnings: List[str] = []
        if detected_pose_coverage < 0.9:
            warnings.append('pose detections are sparse')
        if usable_pose_coverage < 0.7:
            warnings.append('usable pose coverage is low')
        if self.longest_missing_streak > 5:
            warnings.append('missing runs exceed interpolation window')
        if self.longest_unusable_streak > 3:
            warnings.append('several consecutive frames are unstable or low confidence')
        if max_dropout > 0.25:
            warnings.append('some joints drop out frequently')

        recommendations: List[str] = []
        if self.longest_missing_streak > 5:
            recommendations.append('Increase capture quality or reduce motion blur so short gaps stay within interpolation range.')
        if usable_pose_coverage < 0.7:
            recommendations.append('Keep the full body in frame and improve lighting to raise usable pose coverage.')
        if max_dropout > 0.25:
            recommendations.append('Pay attention to the joints with the highest dropout rate; they are the weakest parts of the track.')
        if self.longest_unusable_streak > 3:
            recommendations.append('Slow down rapid movement or move the camera farther back to reduce unstable detections.')
        if not recommendations:
            recommendations.append('Input quality is stable; keep the same capture setup.')

        return {
            'frames_seen': total_frames,
            'detected_pose_coverage': detected_pose_coverage,
            'usable_pose_coverage': usable_pose_coverage,
            'pose_coverage': detected_pose_coverage,
            'longest_missing_pose_streak': int(self.longest_missing_streak),
            'longest_unusable_pose_streak': int(self.longest_unusable_streak),
            'interpolated_frame_count': int(interpolated_frame_count),
            'tracking_loss_frame_count': int(tracking_loss_frame_count),
            'low_confidence_frame_count': int(self.reason_counts['low_confidence']),
            'partial_pose_frame_count': int(self.reason_counts['partial_pose']),
            'unstable_frame_count': int(self.reason_counts['unstable']),
            'missing_frame_count': int(self.reason_counts['missing']),
            'usable_frame_count': int(self.usable_frames),
            'input_quality_warnings': warnings,
            'recommendations': recommendations,
            'joint_dropout_rate_per_joint': dropout_rates,
        }
