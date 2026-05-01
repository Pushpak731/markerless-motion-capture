#!/usr/bin/env python3
"""Offline quality gate for motion-capture verification summaries.

Consumes the JSON summary produced by tools/process_video.py and evaluates
10 concrete pass/fail checks focused on reliability and bone stability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    value: float
    comparator: str
    threshold: float
    passed: bool


def _check_ge(name: str, value: float, threshold: float) -> CheckResult:
    return CheckResult(name=name, value=value, comparator=">=", threshold=threshold, passed=value >= threshold)


def _check_le(name: str, value: float, threshold: float) -> CheckResult:
    return CheckResult(name=name, value=value, comparator="<=", threshold=threshold, passed=value <= threshold)


PROFILES = {
    "balanced": {
        "pose_coverage_min": 0.70,
        "mean_consistency_min": 0.10,
        "p50_consistency_min": 0.10,
        "stddev_consistency_max": 0.30,
        "mean_abs_norm_dev_max": 0.45,
        "p90_abs_norm_dev_max": 0.85,
        "bone_variance_mean_max": 0.05,
        "bone_stddev_mean_max": 0.25,
        "processing_fps_min": 8.0,
    },
    "strict": {
        "pose_coverage_min": 0.80,
        "mean_consistency_min": 0.20,
        "p50_consistency_min": 0.20,
        "stddev_consistency_max": 0.22,
        "mean_abs_norm_dev_max": 0.30,
        "p90_abs_norm_dev_max": 0.60,
        "bone_variance_mean_max": 0.03,
        "bone_stddev_mean_max": 0.18,
        "processing_fps_min": 10.0,
    },
}


def evaluate(summary: dict, profile_name: str = "balanced") -> dict:
    profile = PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(f"Unknown profile: {profile_name}")

    frames_seen = float(summary.get("frames_seen", 0) or 0)
    frames_annotated = float(summary.get("frames_annotated", 0) or 0)

    results = [
        _check_ge("frames_seen_nonzero", frames_seen, 1.0),
        _check_ge("frames_annotation_coverage", (frames_annotated / frames_seen) if frames_seen > 0 else 0.0, 1.0),
        _check_ge("pose_coverage", float(summary.get("pose_coverage", 0.0) or 0.0), profile["pose_coverage_min"]),
        _check_ge("mean_consistency_score", float(summary.get("mean_consistency_score", 0.0) or 0.0), profile["mean_consistency_min"]),
        _check_ge("p50_consistency_score", float(summary.get("p50_consistency_score", 0.0) or 0.0), profile["p50_consistency_min"]),
        _check_le("stddev_consistency_score", float(summary.get("stddev_consistency_score", 0.0) or 0.0), profile["stddev_consistency_max"]),
        _check_le("mean_abs_normalized_deviation", float(summary.get("mean_abs_normalized_deviation", 0.0) or 0.0), profile["mean_abs_norm_dev_max"]),
        _check_le("p90_abs_normalized_deviation", float(summary.get("p90_abs_normalized_deviation", 0.0) or 0.0), profile["p90_abs_norm_dev_max"]),
        _check_le("bone_variance_mean", float(summary.get("bone_variance_mean", 0.0) or 0.0), profile["bone_variance_mean_max"]),
        _check_le("bone_stddev_mean", float(summary.get("bone_stddev_mean", 0.0) or 0.0), profile["bone_stddev_mean_max"]),
        _check_ge("processing_fps", float(summary.get("processing_fps", 0.0) or 0.0), profile["processing_fps_min"]),
    ]

    passed_count = sum(1 for r in results if r.passed)
    total = len(results)

    return {
        "profile": profile_name,
        "overall_pass": passed_count == total,
        "passed_count": passed_count,
        "total_checks": total,
        "checks": [
            {
                "name": r.name,
                "value": round(r.value, 6),
                "rule": f"{r.comparator} {r.threshold}",
                "passed": r.passed,
            }
            for r in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate offline mocap summary with 10 quality checks.")
    parser.add_argument("--summary", required=True, help="Path to summary JSON from tools/process_video.py")
    parser.add_argument("--report", default="", help="Optional output report JSON path")
    parser.add_argument("--profile", choices=["balanced", "strict"], default="balanced", help="Quality threshold profile")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"Summary file not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = evaluate(summary, profile_name=args.profile)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
