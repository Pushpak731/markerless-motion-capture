#!/usr/bin/env bash
set -euo pipefail

# One-command Linux offline verification + 10-check quality gate.
# Usage:
#   scripts/run_offline_validation_linux.sh "/path/to/input.mp4"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input_video_path>"
  exit 1
fi

INPUT_VIDEO="$1"
if [[ $# -gt 1 ]]; then
  echo "[note] Frame caps are ignored here; the full video will always be processed"
fi
MAX_FRAMES="${2:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
OUTPUT_DIR="$PROJECT_ROOT/data/offline_validation_runs"
mkdir -p "$OUTPUT_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
ANNOTATED="$OUTPUT_DIR/annotated_${RUN_ID}.mp4"
SUMMARY="$OUTPUT_DIR/summary_${RUN_ID}.json"
REPORT="$OUTPUT_DIR/quality_report_${RUN_ID}.json"
STRICT_REPORT="$OUTPUT_DIR/quality_report_strict_${RUN_ID}.json"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input not found: $INPUT_VIDEO"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

echo "[setup] Installing offline runtime dependencies"
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet opencv-python mediapipe numpy

echo "[setup] Ensuring MediaPipe model files"
mkdir -p "$PROJECT_ROOT/models"
[[ -f "$PROJECT_ROOT/models/pose_landmarker_lite.task" ]] || curl -fsSL "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" -o "$PROJECT_ROOT/models/pose_landmarker_lite.task"
[[ -f "$PROJECT_ROOT/models/pose_landmarker_full.task" ]] || curl -fsSL "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task" -o "$PROJECT_ROOT/models/pose_landmarker_full.task"
[[ -f "$PROJECT_ROOT/models/pose_landmarker_heavy.task" ]] || curl -fsSL "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" -o "$PROJECT_ROOT/models/pose_landmarker_heavy.task"
[[ -f "$PROJECT_ROOT/models/face_landmarker.task" ]] || curl -fsSL "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -o "$PROJECT_ROOT/models/face_landmarker.task"
[[ -f "$PROJECT_ROOT/models/hand_landmarker.task" ]] || curl -fsSL "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -o "$PROJECT_ROOT/models/hand_landmarker.task"

pushd "$PROJECT_ROOT" >/dev/null

echo "[check] Python syntax sweep"
"$PY" -m py_compile main_gui.py src/calculations.py src/visualizer.py tools/process_video.py tools/offline_quality_gate.py

echo "[run] Processing video"
"$PY" tools/process_video.py \
  --input "$INPUT_VIDEO" \
  --output "$ANNOTATED" \
  --summary-json "$SUMMARY" \
  --max-frames "$MAX_FRAMES"

echo "[gate] Running 10-check quality gate (balanced)"
set +e
"$PY" tools/offline_quality_gate.py --summary "$SUMMARY" --report "$REPORT" --profile balanced
GATE_EXIT=$?

echo "[gate] Running 10-check quality gate (strict)"
"$PY" tools/offline_quality_gate.py --summary "$SUMMARY" --report "$STRICT_REPORT" --profile strict
STRICT_EXIT=$?
set -e

popd >/dev/null

echo
echo "Annotated video: $ANNOTATED"
echo "Summary JSON:    $SUMMARY"
echo "Quality report:  $REPORT"
echo "Strict report:   $STRICT_REPORT"

if [[ $GATE_EXIT -eq 0 ]]; then
  echo "Quality gate: PASS"
else
  echo "Quality gate: FAIL (inspect report)"
fi

if [[ $STRICT_EXIT -eq 0 ]]; then
  echo "Strict gate: PASS"
else
  echo "Strict gate: FAIL (expected to be harder)"
fi

exit $GATE_EXIT
