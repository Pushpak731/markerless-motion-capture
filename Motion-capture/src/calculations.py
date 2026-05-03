import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
from config import (
    VISIBILITY_MIN_METRIC, ANGLE_OUTLIER_BASE_THRESHOLD,
    ANGLE_OUTLIER_VELOCITY_COEFF, MAX_LINEAR_VELOCITY,
    SMOOTHING_ALPHA_DEFAULT, SMOOTHING_ALPHA_FACE,
    BONE_LENGTH_CALIBRATION_FRAMES, BONE_LENGTH_EMA_ALPHA,
    BONE_LENGTH_MAX_DEVIATION, BONE_LENGTH_MIN_CONFIDENCE,
    IPD_DEFAULT
)


@dataclass
class BoneLengthStats:
    """Per-bone statistics used for variance/consistency reporting."""
    samples: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.samples += 1
        delta = value - self.mean
        self.mean += delta / self.samples
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / self.samples if self.samples > 0 else 0.0

    @property
    def stddev(self) -> float:
        return self.variance ** 0.5


class BoneLengthTracker:
    """Stateful bone-length stabilizer.

    The tracker smooths joint positions with EMA before any length computation,
    learns a fixed calibration reference from the first N frames, and then
    reports raw, normalized, constrained, and world-space bone lengths without
    mixing coordinate systems in one calculation path.
    """

    def __init__(
        self,
        calibration_frames: int = BONE_LENGTH_CALIBRATION_FRAMES,
        ema_alpha: float = BONE_LENGTH_EMA_ALPHA,
        max_deviation: float = BONE_LENGTH_MAX_DEVIATION,
        min_confidence: float = BONE_LENGTH_MIN_CONFIDENCE,
    ):
        self.calibration_frames = int(calibration_frames)
        self.ema_alpha = float(ema_alpha)
        self.max_deviation = float(max_deviation)
        self.min_confidence = float(min_confidence)

        self._smoothed_positions: dict[int, dict] = {}
        self._reference_buffers = defaultdict(list)
        self._reference_lengths: dict[str, float] = {}
        self._bone_stats = defaultdict(BoneLengthStats)
        self._frame_count = 0

    @staticmethod
    def _bone_weight(bone_name: str, joint_weights: dict[str, float] | None = None) -> float:
        if joint_weights and bone_name in joint_weights:
            return float(joint_weights[bone_name])

        if 'UpperArm' in bone_name:
            return 0.7
        if 'LowerArm' in bone_name:
            return 0.4
        if 'UpperLeg' in bone_name:
            return 0.8
        if 'LowerLeg' in bone_name:
            return 0.5
        if 'Shoulder' in bone_name:
            return 1.0
        if 'Hip' in bone_name:
            return 1.0
        return 0.6

    @staticmethod
    def _landmarks_to_map(landmarks):
        if not landmarks:
            return {}
        if isinstance(landmarks, dict):
            return {int(k): v for k, v in landmarks.items() if isinstance(v, dict)}
        mapped = {}
        for idx, lm in enumerate(landmarks):
            if isinstance(lm, dict):
                mapped[int(idx)] = lm
        return mapped

    @staticmethod
    def _format_bone_name(a: int, b: int) -> str:
        return f"{Calculations.POSE_IDX_NAMES.get(a, str(a))}_{Calculations.POSE_IDX_NAMES.get(b, str(b))}"

    def _smooth_landmarks(self, landmarks):
        input_map = self._landmarks_to_map(landmarks)
        smoothed = {}

        def _zero_landmark():
            return {
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'visibility': 0.0,
                'v': 0.0,
                'xyz': (0.0, 0.0, 0.0),
            }

        for idx in range(33):
            lm = input_map.get(idx)
            if lm is None:
                previous = self._smoothed_positions.get(idx)
                if previous is None:
                    smoothed[idx] = _zero_landmark()
                else:
                    smoothed[idx] = previous
                continue

            visibility = float(lm.get('visibility', lm.get('v', 1.0)))
            current = np.array([
                float(lm.get('x', 0.0)),
                float(lm.get('y', 0.0)),
                float(lm.get('z', 0.0)),
            ], dtype=float)

            previous = self._smoothed_positions.get(idx)
            if previous is None:
                if visibility < self.min_confidence:
                    smoothed[idx] = _zero_landmark()
                    continue
                smoothed_vec = current
            else:
                alpha = self.ema_alpha if visibility >= self.min_confidence else 0.0
                smoothed_vec = alpha * current + (1.0 - alpha) * np.array(previous['xyz'], dtype=float)

            smoothed[idx] = {
                'x': float(smoothed_vec[0]),
                'y': float(smoothed_vec[1]),
                'z': float(smoothed_vec[2]),
                'visibility': visibility,
                'v': visibility,
                'xyz': (float(smoothed_vec[0]), float(smoothed_vec[1]), float(smoothed_vec[2])),
            }

        # Ensure full landmark vector is present for downstream indexing paths.
        for idx in range(33):
            if idx not in smoothed:
                previous = self._smoothed_positions.get(idx)
                smoothed[idx] = previous if previous is not None else _zero_landmark()

        self._smoothed_positions = smoothed
        return [smoothed[idx] for idx in range(33)]

    def _compute_lengths(self, landmarks):
        metrics = Calculations.get_body_metrics(landmarks)
        return {
            key: value
            for key, value in metrics.items()
            if key.startswith('Length_') or key.startswith('Width_')
        }

    def _update_reference(self, raw_lengths: dict[str, float]):
        self._frame_count += 1

        for bone_name, length in raw_lengths.items():
            self._bone_stats[bone_name].update(length)

            if bone_name not in self._reference_lengths:
                self._reference_buffers[bone_name].append(length)

        if self._frame_count <= self.calibration_frames:
            return

        # Freeze a reference after calibration; bones that were unseen can still
        # adopt their first valid post-calibration measurement so the pipeline
        # keeps working without mixing in a per-frame body-height estimate.
        for bone_name, samples in self._reference_buffers.items():
            if bone_name not in self._reference_lengths and samples:
                self._reference_lengths[bone_name] = float(sum(samples) / len(samples))
        self._reference_buffers.clear()

    def process(self, pose_landmarks, world_landmarks=None, update_reference: bool = True, joint_weights: dict[str, float] | None = None):
        """Return stabilized bone metrics for the best available coordinate source."""
        source_name = 'world' if world_landmarks else 'pose'
        source_landmarks = world_landmarks if world_landmarks else pose_landmarks
        smoothed_landmarks = self._smooth_landmarks(source_landmarks)
        raw_lengths = self._compute_lengths(smoothed_landmarks)

        if update_reference:
            self._update_reference(raw_lengths)

        reference_lengths = dict(self._reference_lengths)
        normalized_lengths = {}
        constrained_lengths = {}
        variance_lengths = {}
        stddev_lengths = {}

        weighted_terms = []
        weighted_total = 0.0
        for bone_name, raw_value in raw_lengths.items():
            reference_value = reference_lengths.get(bone_name)
            if reference_value is None or reference_value <= 1e-9:
                reference_value = raw_value
                if update_reference and bone_name not in self._reference_lengths:
                    self._reference_lengths[bone_name] = raw_value

            normalized_lengths[f'Normalized_{bone_name}'] = round(raw_value / max(reference_value, 1e-9), 4)

            deviation = abs(raw_value - reference_value) / max(reference_value, 1e-9)
            if deviation > self.max_deviation:
                constrained_value = reference_value + (self.max_deviation * reference_value * (1.0 if raw_value >= reference_value else -1.0))
            else:
                constrained_value = raw_value
            constrained_lengths[f'Constrained_{bone_name}'] = round(constrained_value, 4)

            stats = self._bone_stats[bone_name]
            variance_lengths[f'Bone_Length_Variance_{bone_name}'] = round(stats.variance, 6)
            stddev_lengths[f'Bone_Length_StdDev_{bone_name}'] = round(stats.stddev, 6)

            weight = self._bone_weight(bone_name, joint_weights=joint_weights)
            weighted_terms.append(min(1.0, deviation / max(self.max_deviation, 1e-9)) * weight)
            weighted_total += weight

        consistency_score = 1.0
        if weighted_terms and weighted_total > 0:
            consistency_score = max(0.0, 1.0 - (sum(weighted_terms) / weighted_total))

        world_lengths = {
            f'World_{name}': round(value, 4)
            for name, value in raw_lengths.items()
        } if source_name == 'world' else {}

        metrics = {}
        metrics.update({f'Length_{name}': round(value, 4) for name, value in raw_lengths.items()})
        metrics.update(normalized_lengths)
        metrics.update(constrained_lengths)
        metrics.update(variance_lengths)
        metrics.update(stddev_lengths)
        metrics.update(world_lengths)
        metrics.update({f'Reference_{name}': round(value, 4) for name, value in reference_lengths.items()})
        metrics['Bone_Consistency_Score'] = round(consistency_score, 4)
        metrics['Stable_Frame_Used'] = bool(update_reference)

        return {
            'source_name': source_name,
            'smoothed_landmarks': smoothed_landmarks,
            'metrics': metrics,
            'raw_lengths': {f'Length_{name}': round(value, 4) for name, value in raw_lengths.items()},
            'normalized_lengths': normalized_lengths,
            'constrained_lengths': constrained_lengths,
            'world_lengths': world_lengths,
            'variance_lengths': variance_lengths,
            'stddev_lengths': stddev_lengths,
            'reference_lengths': {f'Reference_{name}': round(value, 4) for name, value in reference_lengths.items()},
            'consistency_score': round(consistency_score, 4),
            'stable_frame_used': bool(update_reference),
        }
class Calculations:
    POSE_IDX = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }

    @staticmethod
    def calculate_angle(a, b, c, vector_b_to_a=None):
        """
        Calculate 3D angle. 
        Optionally accept pre-calculated vector_b_to_a (v1) to allow for Spine-relative angles.
        Default: Angle between BA and BC.
        """
        # Vector 1 (BA)
        if vector_b_to_a is not None:
            ba = vector_b_to_a
        else:
            ba = np.array([a['x'] - b['x'], a['y'] - b['y'], a['z'] - b['z']])
            
        # Vector 2 (BC)
        bc = np.array([c['x'] - b['x'], c['y'] - b['y'], c['z'] - b['z']])
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
        
        # Calculate cosine using dot product
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        
        # Clip to handle floating point errors slightly outside [-1, 1]
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.arccos(cosine_angle)
        return round(np.degrees(angle), 2)

    @staticmethod
    def calculate_distance(a, b):
        """Calculate 3D Euclidean distance."""
        pa = np.array([a['x'], a['y'], a['z']])
        pb = np.array([b['x'], b['y'], b['z']])
        return round(float(np.linalg.norm(pa - pb)), 4)

    @staticmethod
    def get_segment_vectors_from_pose_3d(pose_3d):
        """
        Return canonical body segment vectors from a 3D pose dict/list.
        Conventions:
            Upper Arm (R): Elbow - Shoulder
            Forearm (R): Wrist - Elbow
            Trunk Axis: MidShoulder - MidHip
            Pelvic Axis: RightHip - LeftHip
        """
        def p(index):
            if isinstance(pose_3d, dict):
                return pose_3d.get(index)
            if isinstance(pose_3d, list) and index < len(pose_3d):
                return pose_3d[index]
            return None

        ls = p(Calculations.POSE_IDX['left_shoulder'])
        rs = p(Calculations.POSE_IDX['right_shoulder'])
        re = p(Calculations.POSE_IDX['right_elbow'])
        rw = p(Calculations.POSE_IDX['right_wrist'])
        lh = p(Calculations.POSE_IDX['left_hip'])
        rh = p(Calculations.POSE_IDX['right_hip'])

        segments = {}
        if rs and re:
            segments['upper_arm_r'] = {
                'x': re['x'] - rs['x'],
                'y': re['y'] - rs['y'],
                'z': re['z'] - rs['z']
            }
        if re and rw:
            segments['forearm_r'] = {
                'x': rw['x'] - re['x'],
                'y': rw['y'] - re['y'],
                'z': rw['z'] - re['z']
            }
        if ls and rs and lh and rh:
            mid_shoulder = {
                'x': (ls['x'] + rs['x']) / 2.0,
                'y': (ls['y'] + rs['y']) / 2.0,
                'z': (ls['z'] + rs['z']) / 2.0,
            }
            mid_hip = {
                'x': (lh['x'] + rh['x']) / 2.0,
                'y': (lh['y'] + rh['y']) / 2.0,
                'z': (lh['z'] + rh['z']) / 2.0,
            }
            segments['trunk_axis'] = {
                'x': mid_shoulder['x'] - mid_hip['x'],
                'y': mid_shoulder['y'] - mid_hip['y'],
                'z': mid_shoulder['z'] - mid_hip['z']
            }
        if lh and rh:
            segments['pelvic_axis'] = {
                'x': rh['x'] - lh['x'],
                'y': rh['y'] - lh['y'],
                'z': rh['z'] - lh['z']
            }

        return segments

    @staticmethod
    def get_joint_angles_from_pose_3d(pose_3d):
        """Compute canonical 3D joint angles in degrees, clamped to [0, 180]."""
        def p(index):
            if isinstance(pose_3d, dict):
                return pose_3d.get(index)
            if isinstance(pose_3d, list) and index < len(pose_3d):
                return pose_3d[index]
            return None

        def angle(a_idx, b_idx, c_idx):
            a = p(a_idx)
            b = p(b_idx)
            c = p(c_idx)
            if not all([a, b, c]):
                return None
            value = Calculations.calculate_angle(a, b, c)
            return float(np.clip(value, 0.0, 180.0))

        angles = {}
        pairs = {
            'Angle_Elbow_L': ('left_shoulder', 'left_elbow', 'left_wrist'),
            'Angle_Elbow_R': ('right_shoulder', 'right_elbow', 'right_wrist'),
            'Angle_Knee_L': ('left_hip', 'left_knee', 'left_ankle'),
            'Angle_Knee_R': ('right_hip', 'right_knee', 'right_ankle'),
        }

        for key, (a_name, b_name, c_name) in pairs.items():
            value = angle(
                Calculations.POSE_IDX[a_name],
                Calculations.POSE_IDX[b_name],
                Calculations.POSE_IDX[c_name]
            )
            if value is not None:
                angles[key] = value

        return angles

    POSE_IDX_NAMES = {
        11: 'left_shoulder',
        12: 'right_shoulder',
        13: 'left_elbow',
        14: 'right_elbow',
        15: 'left_wrist',
        16: 'right_wrist',
        23: 'left_hip',
        24: 'right_hip',
        25: 'left_knee',
        26: 'right_knee',
        27: 'left_ankle',
        28: 'right_ankle',
    }

    @staticmethod
    def get_joint_angles(pose_landmarks):
        """Calculate joint angles only, without any bone-length normalization."""
        if not pose_landmarks or len(pose_landmarks) < 33:
            return {}

        metrics = {}
        lm = pose_landmarks
        min_vis = VISIBILITY_MIN_METRIC

        def v(indices):
            for idx in indices:
                if 'v' in lm[idx] and lm[idx]['v'] < min_vis:
                    return False
            return True

        mh_x = (lm[23]['x'] + lm[24]['x']) / 2
        mh_y = (lm[23]['y'] + lm[24]['y']) / 2
        mh_z = (lm[23]['z'] + lm[24]['z']) / 2

        ms_x = (lm[11]['x'] + lm[12]['x']) / 2
        ms_y = (lm[11]['y'] + lm[12]['y']) / 2
        ms_z = (lm[11]['z'] + lm[12]['z']) / 2

        spine_vec = np.array([ms_x - mh_x, ms_y - mh_y, ms_z - mh_z])

        if v([11,13,15]): metrics['Angle_Elbow_L'] = Calculations.calculate_angle(lm[11], lm[13], lm[15])
        if v([12,14,16]): metrics['Angle_Elbow_R'] = Calculations.calculate_angle(lm[12], lm[14], lm[16])
        if v([11,13]): metrics['Angle_Shoulder_L'] = Calculations.calculate_angle(None, lm[11], lm[13], vector_b_to_a=spine_vec)
        if v([12,14]): metrics['Angle_Shoulder_R'] = Calculations.calculate_angle(None, lm[12], lm[14], vector_b_to_a=spine_vec)
        if v([23,25]): metrics['Angle_Hip_L'] = Calculations.calculate_angle(None, lm[23], lm[25], vector_b_to_a=spine_vec)
        if v([24,26]): metrics['Angle_Hip_R'] = Calculations.calculate_angle(None, lm[24], lm[26], vector_b_to_a=spine_vec)
        if v([23,25,27]): metrics['Angle_Knee_L'] = Calculations.calculate_angle(lm[23], lm[25], lm[27])
        if v([24,26,28]): metrics['Angle_Knee_R'] = Calculations.calculate_angle(lm[24], lm[26], lm[28])

        return metrics

    @staticmethod
    def get_body_metrics(pose_landmarks):
        """
        Calculate joint angles and raw limb lengths from a pose landmark set.
        This helper remains for backward compatibility; bone-length stabilization
        is now handled by BoneLengthTracker using a fixed calibration reference.
        """
        metrics = Calculations.get_joint_angles(pose_landmarks)
        if not pose_landmarks or len(pose_landmarks) < 33:
            return metrics

        lm = pose_landmarks
        min_vis = VISIBILITY_MIN_METRIC

        def v(indices):
            for idx in indices:
                if 'v' in lm[idx] and lm[idx]['v'] < min_vis:
                    return False
            return True

        if v([11,13]): metrics['Length_UpperArm_L'] = Calculations.calculate_distance(lm[11], lm[13])
        if v([13,15]): metrics['Length_LowerArm_L'] = Calculations.calculate_distance(lm[13], lm[15])
        if v([12,14]): metrics['Length_UpperArm_R'] = Calculations.calculate_distance(lm[12], lm[14])
        if v([14,16]): metrics['Length_LowerArm_R'] = Calculations.calculate_distance(lm[14], lm[16])
        if v([23,25]): metrics['Length_UpperLeg_L'] = Calculations.calculate_distance(lm[23], lm[25])
        if v([25,27]): metrics['Length_LowerLeg_L'] = Calculations.calculate_distance(lm[25], lm[27])
        if v([24,26]): metrics['Length_UpperLeg_R'] = Calculations.calculate_distance(lm[24], lm[26])
        if v([26,28]): metrics['Length_LowerLeg_R'] = Calculations.calculate_distance(lm[26], lm[28])
        if v([11,12]): metrics['Width_Shoulder'] = Calculations.calculate_distance(lm[11], lm[12])
        if v([23,24]): metrics['Width_Hip'] = Calculations.calculate_distance(lm[23], lm[24])

        return metrics

    @staticmethod
    def get_face_metrics(face_landmarks):
        """
        Calculate facial features (Mouth openness, etc.).
        Expects list of dicts.
        """
        # 468 landmarks default
        if not face_landmarks or len(face_landmarks) < 468:
            return {}
            
        metrics = {}
        lm = face_landmarks
        
        # --- MOUTH ---
        # Lips Vertical: Upper(13) <-> Lower(14)
        mouth_h = Calculations.calculate_distance(lm[13], lm[14])
        # Lips Horizontal: Left(61) <-> Right(291)
        mouth_w = Calculations.calculate_distance(lm[61], lm[291])
        
        # --- EYES (Interpupillary Distance - IPD) ---
        ipd = IPD_DEFAULT  # Default from config
        if 468 in lm and 473 in lm:
             ipd = Calculations.calculate_distance(lm[159], lm[386])
        else:
             ipd = Calculations.calculate_distance(lm[33], lm[263]) * 0.5 
             
        if ipd == 0: ipd = 1.0
        
        # Metrics normalized by IPD
        metrics['Face_Mouth_Openness'] = round(mouth_h / ipd, 4)
        metrics['Face_Smile_Ratio'] = round(mouth_w / (mouth_h + 1e-6), 4) # Ratio stays W/H
        
        # ... Eyes Openness (Ratio) ...
        # Left Eye: Vertical 159-145, Horizontal 33-133
        l_eye_v = Calculations.calculate_distance(lm[159], lm[145])
        l_eye_h = Calculations.calculate_distance(lm[33], lm[133])
        metrics['Face_Eye_L_Openness'] = round(l_eye_v / (l_eye_h + 1e-6), 4)
        
        # Right Eye: Vertical 386-374, Horizontal 362-263
        r_eye_v = Calculations.calculate_distance(lm[386], lm[374])
        r_eye_h = Calculations.calculate_distance(lm[362], lm[263])
        metrics['Face_Eye_R_Openness'] = round(r_eye_v / (r_eye_h + 1e-6), 4)

        return metrics

    @staticmethod
    def get_kinematics(current_lm, prev_lm, current_metrics, prev_metrics, dt):
        """
        Calculate instantaneous linear and angular velocities.
        Returns a dict of velocities.
        Units: Meters/sec (approx if coords normalized) or Units/sec, and Deg/sec.
        """
        if dt <= 0: return {}
        
        kinematics = {}
        
        # --- ANGULAR VELOCITY (Deg/s) ---
        # Keys like 'Angle_Elbow_L'
        for key, val in current_metrics.items():
            if key.startswith('Angle_') and key in prev_metrics:
                diff = val - prev_metrics[key]
                # Handle wrapping if needed (though joint angles usually don't wrap abruptly like heading)
                vel = diff / dt
                kinematics[f'Velocity_{key}'] = vel

        # --- LINEAR VELOCITY (Units/s) ---
        # Calculate for key joints: Wrists(15/16), Ankles(27/28), Hips(23/24)
        # current_lm is list of dicts {'x', 'y', 'z', 'v'}
        
        def calc_vel(idx, name):
            if idx < len(current_lm) and idx < len(prev_lm):
                c = current_lm[idx]
                p = prev_lm[idx]
                # Check visibility
                if c.get('v', 1) < VISIBILITY_MIN_METRIC or p.get('v', 1) < VISIBILITY_MIN_METRIC:
                    return
                
                # dist = sqrt(dx^2 + dy^2 + dz^2)
                dx = c['x'] - p['x']
                dy = c['y'] - p['y']
                dz = c['z'] - p['z']
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                vel = dist / dt
                
                # Sanity Check: Human motion cap (approx 20 km/h or 6 m/s for limbs)
                if vel > 6.0: 
                    return
                    
                kinematics[f'Velocity_{name}'] = vel

        calc_vel(15, 'Wrist_L')
        calc_vel(16, 'Wrist_R')
        calc_vel(27, 'Ankle_L')
        calc_vel(28, 'Ankle_R')
        
        return kinematics

    @staticmethod
    def normalize_metrics(metrics, reference_lengths=None, scale_factor=None):
        """
        Normalize all limb lengths by a fixed calibration reference or a global
        scale factor.

        This intentionally avoids frame-varying body-height normalization so the
        same metric does not oscillate just because the hips/ankles jitter.
        """
        if not metrics:
            return metrics

        ref_map = reference_lengths or {}
        global_scale = float(scale_factor) if scale_factor and scale_factor > 1e-9 else None

        for k, v in list(metrics.items()):
            if not (k.startswith('Length_') or k.startswith('Width_')):
                continue

            reference_value = ref_map.get(k)
            if reference_value is not None and reference_value > 1e-9:
                metrics[f"Normalized_{k}"] = round(v / reference_value, 4)
            elif global_scale is not None:
                metrics[f"Normalized_{k}"] = round(v / global_scale, 4)

        return metrics
    
    @staticmethod
    def filter_and_smooth(current, prev, alpha=SMOOTHING_ALPHA_DEFAULT):
        """
        1. Outlier Rejection: Velocity-Dependent Threshold.
        2. Smoothing: Apply EMA (Exponential Moving Average).
        """
        if not prev:
            return current
            
        filtered = {}
        for k, v in current.items():
            # Keep length metrics as the tracker computed them; only smooth
            # angular / face signals here. This avoids re-mixing raw, normalized,
            # and world-space bone lengths after the stabilized pipeline.
            if k.startswith(('Length_', 'Normalized_', 'World_', 'Constrained_', 'Bone_Length_')):
                filtered[k] = v
                continue

            # Only smooth Angles and non-length metrics
            if k not in prev:
                filtered[k] = v
                continue
                
            p_val = prev[k]
            
            # 1. Outlier Rejection (Angles only)
            if k.startswith('Angle_'):
                diff = abs(v - p_val)
                
                # Dynamic Threshold: 50 deg + k * velocity
                # If we have a stored velocity for this metric, use it.
                # Since 'Velocity_' keys might not be in 'prev' (or are computed *after* this),
                # we can approximate angular velocity by just looking at the raw diff if needed,
                # BUT the user suggested using 'angular_velocity_prev'.
                # Let's check if 'Velocity_{k}' exists in prev.
                vel_key = f"Velocity_{k}"
                threshold = ANGLE_OUTLIER_BASE_THRESHOLD
                if vel_key in prev:
                    threshold += ANGLE_OUTLIER_VELOCITY_COEFF * abs(prev[vel_key])
                
                if diff > threshold:
                    # Spike detected! Ignore new value, keep old.
                    filtered[k] = p_val 
                    continue
            
                if diff > threshold:
                    # Spike detected! Ignore new value, keep old.
                    filtered[k] = p_val 
                    continue
            
            # 2. EMA Smoothing
            # Default alpha 0.5
            a = alpha
            
            # Use smoother alpha for Face Metrics (micro-jitter)
            if k.startswith('Face_'):
                a = SMOOTHING_ALPHA_FACE  # Stronger smoothing for face
            
            smoothed_val = a * v + (1 - a) * p_val
            
            # Rounding
            if k.startswith('Angle_'):
                filtered[k] = round(smoothed_val, 2)
            else:
                filtered[k] = round(smoothed_val, 4)
                
        return filtered
