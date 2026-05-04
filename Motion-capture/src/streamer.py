import cv2
import threading
import queue
import time
from src.camera import Camera
from src.detector import MocapDetector
from src.visualizer import Visualizer
from src.database import MocapDB
from src.pose_corrector import PoseCorrector
from src.calculations import Calculations, BoneLengthTracker
from src.kinematics import KinematicsTracker
from config import DRAW_LANDMARKS

class VideoStreamer:
    def __init__(self):
        self.camera = Camera()
        self.detector = MocapDetector()
        self.visualizer = Visualizer()
        self.db = MocapDB()
        self.corrector = PoseCorrector() # Init Physics Engine
        self.running = False
        self.thread = None
        self.output_frame = None
        self.latest_metrics = {}
        self.bone_tracker = BoneLengthTracker()
        self.kinematics_tracker = KinematicsTracker()
        # Kinematics State
        self.prev_lm = []
        self.prev_metrics = {}
        self.prev_time = None
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.db.stop_recording() # Ensure DB stops
        if self.thread:
            self.thread.join()
        self.camera.release()

    def _process_loop(self):
        while self.running and self.camera.is_opened():
            frame = self.camera.read()
            if frame is None:
                continue

            # Process
            results = self.detector.process(frame)
            
            # --- PHYSICS CORRECTION ---
            # Corrects bone lengths and smooths jitter
            results = self.corrector.process(results)
            # --------------------------
            
            # Save to Database (if recording)
            self.db.save_frame(results)

            # Update latest metrics
            if results.get('pose') and results.get('pose').pose_landmarks:
                try:
                    # Uses the corrected landmarks
                    # Prioritize World Landmarks for Physics/Metrics (Meters)
                    world_lm = results.get('pose').pose_world_landmarks[0] if results.get('pose').pose_world_landmarks else None
                    plm = world_lm if world_lm else results.get('pose').pose_landmarks[0]

                    lm_dict = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'v': lm.visibility} for lm in plm]
                    world_dict = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'v': getattr(lm, 'visibility', 1.0)} for lm in world_lm] if world_lm else None
                    
                    # Core Metrics
                    source_metrics = Calculations.get_body_metrics(lm_dict)
                    angle_metrics = {
                        k: v for k, v in source_metrics.items()
                        if k.startswith('Angle_')
                    }

                    bone_result = self.bone_tracker.process(lm_dict, world_dict)
                    bone_metrics = bone_result.get('metrics', {})
                    smoothed_lm = bone_result.get('smoothed_landmarks') or lm_dict
                    coordinate_space = bone_result.get('source_name', 'world' if world_dict else 'normalized')
                    
                    # Face Metrics
                    face_metrics = {}
                    if results.get('face') and results.get('face').face_landmarks:
                        # 468 landmarks
                        flm = results.get('face').face_landmarks[0]
                        flm_dict = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in flm]
                        face_metrics = Calculations.get_face_metrics(flm_dict)
                        
                    current_metrics = Calculations.filter_and_smooth({**angle_metrics, **bone_metrics, **face_metrics}, self.prev_metrics)
                    # -------------------------------------------------
                    
                    # Kinematics
                    now = time.time()
                    current_metrics.update(
                        self.kinematics_tracker.process(
                            smoothed_lm,
                            current_metrics,
                            now * 1000.0,
                            coordinate_space=coordinate_space,
                        )
                    )
                    
                    # Update State
                        self.prev_lm = smoothed_lm
                    self.prev_metrics = current_metrics
                    self.prev_time = now
                    
                    self.latest_metrics = current_metrics
                except Exception as e:
                    # print(f"Processing error: {e}") 
                    pass
            
            # Visualize
            if DRAW_LANDMARKS:
                frame = self.visualizer.draw_landmarks(frame, results)
            
            frame = self.visualizer.draw_fps(frame)

            # Encode for streaming/storage
            with self.lock:
                self.output_frame = frame.copy()

    def generate(self):
        """Generator function for streaming"""
        while self.running:
            frame_to_encode = None
            with self.lock:
                if self.output_frame is None:
                    pass
                else:
                    frame_to_encode = self.output_frame.copy()

            if frame_to_encode is None:
                time.sleep(0.01)
                continue

            # Encode as JPEG outside lock
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode)
            if not flag:
                time.sleep(0.01)
                continue
            
            # Yield frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            
            # Limit streaming FPS slightly to save bandwidth
            time.sleep(0.01)
