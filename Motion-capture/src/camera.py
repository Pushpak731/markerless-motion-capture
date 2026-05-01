import cv2
import threading
import time
from config import CAMERA_ID, CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT, FPS

class Camera:
    def __init__(self, camera_id=None):
        self.camera_id = CAMERA_SOURCE if camera_id is None else camera_id
        self.source = self._normalize_source(self.camera_id)
        self.cap = cv2.VideoCapture(self.source)
        
        # Configure Camera
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        self.grabbed = False
        self.frame = None
        self.frame_event = threading.Event()
        self.running = False
        self.thread = None
        
        if self.cap.isOpened():
            self.grabbed, self.frame = self.cap.read()
            self.running = True
            self.start_thread()
        else:
            raise RuntimeError(f"Could not open camera source {self.source!r}")

    @staticmethod
    def _normalize_source(source):
        if isinstance(source, str):
            stripped = source.strip()
            if stripped.isdigit():
                return int(stripped)
            return stripped
        return source
            
    def start_thread(self):
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while self.running:
            if self.cap.isOpened():
                grabbed, frame = self.cap.read()
                if grabbed:
                    self.grabbed = grabbed
                    self.frame = frame
                    self.frame_event.set()
                else:
                    self.running = False
            else:
                self.running = False

    def read(self):
        """Return the most recent frame."""
        self.frame_event.clear()
        return self.frame if self.grabbed else None

    def wait_for_frame(self, timeout=0.033):
        """Block until a new frame is available or timeout elapses."""
        return self.frame_event.wait(timeout)
        
    def release(self):
        """Release the camera resource."""
        self.running = False
        self.frame_event.set()
        if self.thread:
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
            
    def is_opened(self):
        return self.cap.isOpened()
