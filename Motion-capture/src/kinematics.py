"""Extended stateful kinematics metrics.

This module extends the existing metric pipeline without recomputing joint
angles or limb lengths. It consumes the already-smoothed landmarks and angle
columns, then adds velocity components, accelerations, optional jerk, and
metadata about the coordinate space and smoothing method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from config import (
    ENABLE_JERK_METRICS,
    ENABLE_KINEMATICS_VECTOR_COMPONENTS,
    KINEMATICS_MAX_ANGULAR_ACCELERATION,
    KINEMATICS_MAX_ANGULAR_VELOCITY,
    KINEMATICS_MAX_LINEAR_ACCELERATION,
    KINEMATICS_MAX_LINEAR_VELOCITY,
    KINEMATICS_SMOOTHING_ALPHA,
    KINEMATICS_SMOOTHING_ENABLED,
    KINEMATICS_SMOOTHING_METHOD,
    KINEMATICS_VISIBILITY_THRESHOLD,
)


TRACKED_JOINTS = {
    "Shoulder_L": 11,
    "Shoulder_R": 12,
    "Elbow_L": 13,
    "Elbow_R": 14,
    "Wrist_L": 15,
    "Wrist_R": 16,
    "Hip_L": 23,
    "Hip_R": 24,
    "Knee_L": 25,
    "Knee_R": 26,
    "Ankle_L": 27,
    "Ankle_R": 28,
}


ANGLE_KEYS = (
    "Angle_Elbow_L",
    "Angle_Elbow_R",
    "Angle_Shoulder_L",
    "Angle_Shoulder_R",
    "Angle_Hip_L",
    "Angle_Hip_R",
    "Angle_Knee_L",
    "Angle_Knee_R",
)


@dataclass
class JointState:
    timestamp_ms: float
    position: tuple[float, float, float]
    velocity: Optional[tuple[float, float, float]] = None
    acceleration: Optional[tuple[float, float, float]] = None


@dataclass
class AngleState:
    timestamp_ms: float
    angle: float
    angular_velocity: Optional[float] = None


def _visibility(landmark: Dict[str, Any]) -> float:
    return float(landmark.get("v", landmark.get("visibility", landmark.get("conf", 1.0))))


def _position(landmark: Dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(landmark.get("x", 0.0)),
        float(landmark.get("y", 0.0)),
        float(landmark.get("z", 0.0)),
    )


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(v: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (v[0] * scalar, v[1] * scalar, v[2] * scalar)


def _lerp(a: tuple[float, float, float], b: tuple[float, float, float], alpha: float) -> tuple[float, float, float]:
    return (
        alpha * a[0] + (1.0 - alpha) * b[0],
        alpha * a[1] + (1.0 - alpha) * b[1],
        alpha * a[2] + (1.0 - alpha) * b[2],
    )


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _angle_delta_degrees(current: float, previous: float) -> float:
    """Smallest signed angular delta in degrees."""
    return (current - previous + 180.0) % 360.0 - 180.0


class KinematicsTracker:
    """Stateful velocity, acceleration, and jerk calculator."""

    def __init__(
        self,
        visibility_threshold: float = KINEMATICS_VISIBILITY_THRESHOLD,
        smoothing_enabled: bool = KINEMATICS_SMOOTHING_ENABLED,
        smoothing_method: str = KINEMATICS_SMOOTHING_METHOD,
        smoothing_alpha: float = KINEMATICS_SMOOTHING_ALPHA,
        enable_vector_components: bool = ENABLE_KINEMATICS_VECTOR_COMPONENTS,
        enable_jerk: bool = ENABLE_JERK_METRICS,
    ):
        self.visibility_threshold = float(visibility_threshold)
        self.smoothing_enabled = bool(smoothing_enabled)
        self.smoothing_method = str(smoothing_method)
        self.smoothing_alpha = float(smoothing_alpha)
        self.enable_vector_components = bool(enable_vector_components)
        self.enable_jerk = bool(enable_jerk)
        self._joint_state: Dict[str, JointState] = {}
        self._angle_state: Dict[str, AngleState] = {}

    @staticmethod
    def metadata(
        coordinate_space: str,
        smoothing_enabled: bool = KINEMATICS_SMOOTHING_ENABLED,
        smoothing_method: str = KINEMATICS_SMOOTHING_METHOD,
        smoothing_alpha: float = KINEMATICS_SMOOTHING_ALPHA,
    ) -> Dict[str, Any]:
        return {
            "Coordinate_Space": coordinate_space,
            "Kinematics_Smoothing_Enabled": bool(smoothing_enabled),
            "Kinematics_Smoothing_Method": smoothing_method,
            "Kinematics_Smoothing_Alpha": float(smoothing_alpha),
        }

    def process(
        self,
        landmarks: Iterable[Dict[str, Any]],
        metrics: Dict[str, Any],
        timestamp_ms: float,
        coordinate_space: str,
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = self.metadata(
            coordinate_space=coordinate_space,
            smoothing_enabled=self.smoothing_enabled,
            smoothing_method=self.smoothing_method,
            smoothing_alpha=self.smoothing_alpha,
        )

        landmarks_list = list(landmarks or [])
        output.update(self._linear_kinematics(landmarks_list, float(timestamp_ms)))
        output.update(self._angular_kinematics(metrics or {}, float(timestamp_ms)))
        return output

    def _linear_kinematics(self, landmarks: list[Dict[str, Any]], timestamp_ms: float) -> Dict[str, Any]:
        output: Dict[str, Any] = {}

        for name, idx in TRACKED_JOINTS.items():
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            if not isinstance(lm, dict) or _visibility(lm) < self.visibility_threshold:
                continue

            pos = _position(lm)
            prev = self._joint_state.get(name)
            next_state = JointState(timestamp_ms=timestamp_ms, position=pos)

            if prev is not None:
                dt = (timestamp_ms - prev.timestamp_ms) / 1000.0
                if dt > 0:
                    velocity = _scale(_sub(pos, prev.position), 1.0 / dt)
                    velocity_mag = _norm(velocity)
                    if velocity_mag <= KINEMATICS_MAX_LINEAR_VELOCITY:
                        if self.smoothing_enabled and prev.velocity is not None:
                            velocity = _lerp(velocity, prev.velocity, self.smoothing_alpha)
                            velocity_mag = _norm(velocity)

                        output[f"Velocity_{name}"] = velocity_mag
                        if self.enable_vector_components:
                            output[f"Velocity_{name}_X"] = velocity[0]
                            output[f"Velocity_{name}_Y"] = velocity[1]
                            output[f"Velocity_{name}_Z"] = velocity[2]

                        next_state.velocity = velocity

                        if prev.velocity is not None:
                            acceleration = _scale(_sub(velocity, prev.velocity), 1.0 / dt)
                            acceleration_mag = _norm(acceleration)
                            if acceleration_mag <= KINEMATICS_MAX_LINEAR_ACCELERATION:
                                output[f"Acceleration_{name}"] = acceleration_mag
                                if self.enable_vector_components:
                                    output[f"Acceleration_{name}_X"] = acceleration[0]
                                    output[f"Acceleration_{name}_Y"] = acceleration[1]
                                    output[f"Acceleration_{name}_Z"] = acceleration[2]
                                next_state.acceleration = acceleration

                                if self.enable_jerk and prev.acceleration is not None:
                                    jerk = _scale(_sub(acceleration, prev.acceleration), 1.0 / dt)
                                    output[f"Jerk_{name}"] = _norm(jerk)

            self._joint_state[name] = next_state

        return output

    def _angular_kinematics(self, metrics: Dict[str, Any], timestamp_ms: float) -> Dict[str, Any]:
        output: Dict[str, Any] = {}

        for angle_key in ANGLE_KEYS:
            if angle_key not in metrics:
                continue
            try:
                angle = float(metrics[angle_key])
            except Exception:
                continue

            prev = self._angle_state.get(angle_key)
            next_state = AngleState(timestamp_ms=timestamp_ms, angle=angle)
            if prev is not None:
                dt = (timestamp_ms - prev.timestamp_ms) / 1000.0
                if dt > 0:
                    angular_velocity = _angle_delta_degrees(angle, prev.angle) / dt
                    if abs(angular_velocity) <= KINEMATICS_MAX_ANGULAR_VELOCITY:
                        velocity_key = f"Velocity_{angle_key}"
                        output[velocity_key] = angular_velocity
                        next_state.angular_velocity = angular_velocity

                        if prev.angular_velocity is not None:
                            angular_accel = (angular_velocity - prev.angular_velocity) / dt
                            if abs(angular_accel) <= KINEMATICS_MAX_ANGULAR_ACCELERATION:
                                output[f"Acceleration_{angle_key}"] = angular_accel

            self._angle_state[angle_key] = next_state

        return output

