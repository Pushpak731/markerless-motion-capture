import cv2
import sys
import os

PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector import MocapDetector

detector = MocapDetector(enable_face=False, enable_hand=False)
cap = cv2.VideoCapture('data/video1_source.mp4')
ret, frame = cap.read()
if ret:
    results = detector.process(frame, timestamp_ms=0)
    pose = results.get('pose')
    if pose and pose.pose_landmarks:
        print(f"SUCCESS: Found {len(pose.pose_landmarks[0])} landmarks")
        print(f"First landmark: {pose.pose_landmarks[0][0]}")
    else:
        print("FAILURE: No pose detected in first frame")
else:
    print("FAILURE: Could not read video")
cap.release()
