import cv2
import time
import numpy as np
from collections import deque
from config import SHOW_FPS, THEME_COLOR
from src.calculations import Calculations


# Minimal connection sets to avoid hard dependency on mediapipe.solutions.
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (27, 29), (27, 31), (28, 30), (28, 32),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


class Visualizer:
    def __init__(self):
        self.frame_times = deque(maxlen=30)
        self.last_time = None
        
    def draw_landmarks(self, frame, results):
        """Draw pose, face, and hand landmarks."""
        pose_result = results.get('pose')
        face_result = results.get('face')
        hand_result = results.get('hand')
        h, w = frame.shape[:2]

        def xy(lm):
            x = int(float(lm.x) * w)
            y = int(float(lm.y) * h)
            return x, y

        def draw_connections(landmarks, connections, color, thickness=2):
            for a, b in connections:
                if a >= len(landmarks) or b >= len(landmarks):
                    continue
                p1 = xy(landmarks[a])
                p2 = xy(landmarks[b])
                cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

        def draw_points(landmarks, color, radius=2, step=1):
            for idx, lm in enumerate(landmarks):
                if step > 1 and idx % step != 0:
                    continue
                p = xy(lm)
                cv2.circle(frame, p, radius, color, -1, cv2.LINE_AA)
        
        # 1. Draw Pose
        if pose_result and pose_result.pose_landmarks:
            for pose_landmarks in pose_result.pose_landmarks:
                draw_connections(pose_landmarks, POSE_CONNECTIONS, (0, 255, 0), thickness=2)
                draw_points(pose_landmarks, (0, 200, 255), radius=3)

        # 2. Draw Face Mesh
        if face_result and face_result.face_landmarks:
            for face_landmarks in face_result.face_landmarks:
                # Face mesh can be dense; draw every 4th point for speed/readability.
                draw_points(face_landmarks, (255, 140, 0), radius=1, step=4)

        # 3. Draw Hands
        if hand_result and hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                draw_connections(hand_landmarks, HAND_CONNECTIONS, (255, 255, 0), thickness=2)
                draw_points(hand_landmarks, (255, 0, 255), radius=2)
            
        return frame

    def draw_fps(self, frame):
        """Draw FPS on frame."""
        fps = self.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, THEME_COLOR, 2)
        return frame
    
    def get_fps(self):
        """Calculate and return current FPS."""
        now = time.time()
        if self.last_time:
            frame_time = now - self.last_time
            self.frame_times.append(frame_time)
        self.last_time = now
        
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            if avg_frame_time > 0:
                return 1.0 / avg_frame_time
        return 0.0
