#!/usr/bin/env python3
"""
local_3d_studio.py — Native Python 3D Avatar Retargeting Studio
Uses Panda3D for hardware-accelerated 3D rendering and Tkinter for the file picker UI.

Launch:
    python tools/local_3d_studio.py

Workflow:
    1. A Tkinter popup lets you pick a .glb character and a _raw_3d_nodes.csv file.
    2. The Panda3D window opens and renders the character.
    3. The CSV is played back frame-by-frame, with joint positions mapped to bone rotations.

Bone Mapping:
    Currently supports Mixamo rig naming convention.
    Bones are matched using the BONE_MAP dictionary below.
"""

import os
import sys
import csv
import time
import tkinter as tk
from tkinter import filedialog, messagebox

# --- DEBUG OPTIONS ---
DEBUG_ONLY_BONE = 'Skeleton_torso_joint_1' # Isolate torso first as recommended
SHOW_DEBUG_SKELETON = True
# --------------------

# ─── Mixamo Bone Mapping ────────────────────────────────────────────────────
# Maps MediaPipe joint name pairs → Mixamo bone names
# Each entry: (joint_start_idx, joint_end_idx, mixamo_bone_name)
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
JOINT_IDX = {name: i for i, name in enumerate(MEDIAPIPE_JOINT_NAMES)}

BONE_MAP_MIXAMO = {
    'mixamorigLeftArm':       ('left_shoulder',   'left_elbow'),
    'mixamorigLeftForeArm':   ('left_elbow',      'left_wrist'),
    'mixamorigRightArm':      ('right_shoulder',  'right_elbow'),
    'mixamorigRightForeArm':  ('right_elbow',     'right_wrist'),
    'mixamorigLeftUpLeg':     ('left_hip',        'left_knee'),
    'mixamorigLeftLeg':       ('left_knee',       'left_ankle'),
    'mixamorigRightUpLeg':    ('right_hip',       'right_knee'),
    'mixamorigRightLeg':      ('right_knee',      'right_ankle'),
    'mixamorigSpine':         ('left_hip',        'left_shoulder'),
    'mixamorigNeck':          ('left_shoulder',   'nose'),
}

BONE_MAP_CESIUM = {
    'Skeleton_arm_joint_L__4_': ('left_shoulder',   'left_elbow'),
    'Skeleton_arm_joint_L__3_': ('left_elbow',      'left_wrist'),
    'Skeleton_arm_joint_R':      ('right_shoulder',  'right_elbow'),
    'Skeleton_arm_joint_R__2_':  ('right_elbow',     'right_wrist'),
    'leg_joint_L_1':             ('left_hip',        'left_knee'),
    'leg_joint_L_2':             ('left_knee',       'left_ankle'),
    'leg_joint_R_1':             ('right_hip',       'right_knee'),
    'leg_joint_R_2':             ('right_knee',      'right_ankle'),
    'Skeleton_torso_joint_1':    ('left_hip',        'left_shoulder'),
}

# Default
BONE_MAP = BONE_MAP_MIXAMO

# ─── File Picker ─────────────────────────────────────────────────────────────

def pick_files():
    """Show a Tkinter dialog to pick a GLB and a raw 3D nodes CSV."""
    root = tk.Tk()
    root.withdraw()
    root.title("MoCap Local 3D Studio")

    messagebox.showinfo(
        "3D Avatar Studio — Step 1",
        "Pick your character model (.glb or .gltf)"
    )
    glb_path = filedialog.askopenfilename(
        title="Select Character Model",
        filetypes=[("3D Model", "*.glb *.gltf"), ("All files", "*.*")]
    )
    if not glb_path:
        messagebox.showwarning("Cancelled", "No model selected. Exiting.")
        root.destroy()
        return None, None

    messagebox.showinfo(
        "3D Avatar Studio — Step 2",
        "Pick the raw 3D nodes CSV from an offline validation run.\n"
        "(Files ending in _raw_3d_nodes.csv)"
    )
    csv_path = filedialog.askopenfilename(
        title="Select Raw 3D Nodes CSV",
        filetypes=[("CSV Files", "*_raw_3d_nodes.csv *.csv"), ("All files", "*.*")]
    )
    root.destroy()

    if not csv_path:
        print("[studio] No CSV selected — will show static T-pose.")

    return glb_path, csv_path


# ─── CSV Loader ──────────────────────────────────────────────────────────────

def load_csv_frames(csv_path):
    """Load all frames from the raw 3D nodes CSV into a list of dicts."""
    frames = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if not headers or 'timestamp_ms' not in headers or 'nose_x' not in headers:
            print(f"[studio] ERROR: Invalid CSV format. This looks like a metrics file, not a raw 3D nodes file.")
            print(f"[studio] Please select the file ending in '_raw_3d_nodes.csv'")
            return []

        for row in reader:
            joints = {}
            for jname in MEDIAPIPE_JOINT_NAMES:
                try:
                    joints[jname] = {
                        'x': float(row.get(f'{jname}_x', 0)),
                        'y': float(row.get(f'{jname}_y', 0)),
                        'z': float(row.get(f'{jname}_z', 0)),
                        'v': float(row.get(f'{jname}_v', 0)),
                    }
                except (ValueError, TypeError):
                    joints[jname] = {'x': 0, 'y': 0, 'z': 0, 'v': 0}
            frames.append({
                'frame_idx': int(row.get('frame_idx', 0)),
                'timestamp_ms': float(row.get('timestamp_ms', 0)),
                'root': {
                    'x': float(row.get('root_x', 0.5)),
                    'y': float(row.get('root_y', 0.5)),
                    'z': float(row.get('root_z', 0)),
                },
                'joints': joints,
            })
    return frames


# ─── Panda3D App ─────────────────────────────────────────────────────────────

try:
    from direct.showbase.ShowBase import ShowBase
    from direct.actor.Actor import Actor
    from panda3d.core import (
        AmbientLight, DirectionalLight, NodePath,
        Vec3, Vec4, LQuaternionf, LVecBase3f,
        WindowProperties, AntialiasAttrib
    )
    import gltf as panda3d_gltf
except ImportError as e:
    try:
        import panda3d_gltf
    except ImportError:
        print(f"[studio] ERROR: Panda3D GLTF loader is not installed: {e}")
        print("Install with:  pip install panda3d-gltf")
        sys.exit(1)


class AvatarStudio(ShowBase):
    def __init__(self, glb_path, frames, fps=30.0):
        ShowBase.__init__(self)

        # Window styling
        props = WindowProperties()
        props.setTitle("MoCap 3D Avatar Studio — Local")
        props.setSize(1280, 720)
        self.win.requestProperties(props)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.setBackgroundColor(0.08, 0.07, 0.07, 1)

        self._frames = frames
        self._fps = max(fps, 1.0)
        self._frame_idx = 0
        self._last_frame_time = time.time()
        self._playing = True
        self._controlled_joints = {}

        self._setup_lights()
        self._load_avatar(glb_path)
        self._setup_camera()
        self._setup_controls()
        self._draw_hud()

        # Tick loop
        self.taskMgr.add(self._update_task, 'update')

    def _setup_lights(self):
        ambient = AmbientLight('ambient')
        ambient.setColor(Vec4(0.4, 0.4, 0.45, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight('sun')
        sun.setColor(Vec4(1, 0.98, 0.9, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)
        self.render.setLight(sun_np)

        fill = DirectionalLight('fill')
        fill.setColor(Vec4(0.3, 0.35, 0.5, 1))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(-135, -20, 0)
        self.render.setLight(fill_np)

    def _load_avatar(self, glb_path):
        # Determine bone map based on filename
        global BONE_MAP
        if "CesiumMan" in glb_path:
            BONE_MAP = BONE_MAP_CESIUM
            print("[studio] Using CesiumMan bone mapping")
        else:
            BONE_MAP = BONE_MAP_MIXAMO
            print("[studio] Using Mixamo bone mapping")

        try:
            # Patch the loader to support GLTF if not already done
            if hasattr(panda3d_gltf, 'patch_loader'):
                panda3d_gltf.patch_loader(self.loader)

            # Initialize Actor directly with the path
            self._avatar = Actor(glb_path)
            self._avatar.reparentTo(self.render)
            self._avatar.setPos(0, 0, 0)
            self._avatar.setH(180) # Face the camera (Default for most GLBs)
            self._avatar.stop()    # Stop default animations
            
            if SHOW_DEBUG_SKELETON:
                self._debug_np = self.render.attachNewNode("debug_skeleton")
                self._debug_np.setPos(2, 0, 0) # Offset to the side

            # Debug: Print ALL joint names
            print("[studio] Full joint list found in model:")
            joints = self._avatar.getJoints()
            for j in joints:
                print(f"  - {j.getName()}")

            # Scale to ~1.7m height
            bounds = self._avatar.getTightBounds()
            if bounds:
                min_pt, max_pt = bounds
                height = max_pt[2] - min_pt[2]
                if height > 0:
                    self._avatar.setScale(1.7 / height)

            # Map bones using controlJoint on the Actor
            available_joints = {j.getName() for j in self._avatar.getJoints()}
            for bone_name in BONE_MAP:
                if bone_name in available_joints:
                    # Use controlJoint with partName=None for direct skeletal override
                    bone_np = self._avatar.controlJoint(None, 'modelRoot', bone_name)
                    if not bone_np or bone_np.isEmpty():
                        bone_np = self._avatar.controlJoint(None, 'model', bone_name)
                    if not bone_np or bone_np.isEmpty():
                        bone_np = self._avatar.controlJoint(None, None, bone_name)
                    
                    if bone_np and not bone_np.isEmpty():
                        self._controlled_joints[bone_name] = bone_np
                        print(f"  [studio] Successfully controlling: {bone_name}")
                else:
                    print(f"  [studio] Skipping missing joint: {bone_name}")

            print(f"[studio] Model loaded: {os.path.basename(glb_path)}")
            print(f"[studio] Bones successfully controlled: {len(self._controlled_joints)} / {len(BONE_MAP)}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[studio] ERROR loading model: {e}")
            self._avatar = None

    def _setup_camera(self):
        self.disableMouse()
        self.camera.setPos(0, -4, 1.5)
        self.camera.lookAt(0, 0, 1.0)

    def _setup_controls(self):
        self.accept('escape', self.userExit)
        self.accept('space', self._toggle_playback)
        self.accept('r', self._restart)
        self.accept('arrow_right', self._step_forward)
        self.accept('arrow_left', self._step_backward)
        # Rotation and orientation controls
        self.accept('a', self._rotate_left)
        self.accept('d', self._rotate_right)
        self.accept('f', self._flip_character)

    def _rotate_left(self):
        if hasattr(self, '_avatar'):
            self._avatar.setH(self._avatar.getH() + 10)

    def _rotate_right(self):
        if hasattr(self, '_avatar'):
            self._avatar.setH(self._avatar.getH() - 10)

    def _flip_character(self):
        if hasattr(self, '_avatar'):
            self._avatar.setH(self._avatar.getH() + 180)

    def _draw_hud(self):
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode
        self._hud_text = OnscreenText(
            text="Loading...",
            pos=(-1.3, 0.9),
            scale=0.04,
            fg=(0.8, 1.0, 0.9, 1),
            align=TextNode.ALeft,
            shadow=(0, 0, 0, 0.5),
            mayChange=True,
        )
        OnscreenText(
            text="[Space] Play/Pause | [<- / ->] Step | [R] Restart | [Esc] Quit",
            pos=(0, -0.93),
            scale=0.035,
            fg=(0.6, 0.6, 0.6, 1),
            mayChange=False,
        )

    def _toggle_playback(self):
        self._playing = not self._playing

    def _restart(self):
        self._frame_idx = 0
        self._playing = True

    def _step_forward(self):
        self._playing = False
        self._apply_frame(min(self._frame_idx + 1, len(self._frames) - 1))

    def _step_backward(self):
        self._playing = False
        self._apply_frame(max(self._frame_idx - 1, 0))

    def _update_task(self, task):
        if not self._frames:
            return task.cont

        now = time.time()
        dt = now - self._last_frame_time
        frame_duration = 1.0 / self._fps

        if self._playing and dt >= frame_duration:
            self._apply_frame(self._frame_idx)
            self._frame_idx = (self._frame_idx + 1) % len(self._frames)
            self._last_frame_time = now

        return task.cont

    def _apply_frame(self, frame_idx):
        if not self._frames or not self._controlled_joints:
            return

        frame = self._frames[frame_idx]
        joints = frame['joints']
        self._frame_idx = frame_idx

        # Update HUD
        status_text = "PLAYING" if self._playing else "PAUSED"
        self._hud_text.setText(
            f"Frame: {frame_idx + 1} / {len(self._frames)}\n"
            f"Time: {frame['timestamp_ms'] / 1000:.2f}s\n"
            f"Status: {status_text}"
        )
        
        # Helper for calculating vectors between joints (User's suggested mapping)
        def joint_vec(js, a_name, b_name):
            ja = js[a_name] if isinstance(a_name, str) else a_name
            jb = js[b_name] if isinstance(b_name, str) else b_name
            return Vec3(
                jb['x'] - ja['x'],
                jb['z'] - ja['z'],
                -(jb['y'] - ja['y'])
            )

        # Calculate midpoints for stable torso/root
        l_hip, r_hip = joints['left_hip'], joints['right_hip']
        l_sho, r_sho = joints['left_shoulder'], joints['right_shoulder']
        
        mid_hip = {
            'x': (l_hip['x'] + r_hip['x']) / 2,
            'y': (l_hip['y'] + r_hip['y']) / 2,
            'z': (l_hip['z'] + r_hip['z']) / 2,
        }
        mid_sho = {
            'x': (l_sho['x'] + r_sho['x']) / 2,
            'y': (l_sho['y'] + r_sho['y']) / 2,
            'z': (l_sho['z'] + r_sho['z']) / 2,
        }

        # 1. Root Translation (Increased scale for noticeable movement)
        scale = 8.0 # 8 meters room scale
        root_x = mid_hip['x'] * scale
        root_y = mid_hip['z'] * scale
        root_z = -mid_hip['y'] * scale
        
        raw_root = Vec3(root_x, root_y, root_z)
        
        if not hasattr(self, "_root_origin"):
            self._root_origin = raw_root
            print(f"[studio] Root Origin set: {self._root_origin}")

        # Final position relative to frame 0
        target_pos = raw_root - self._root_origin
        
        # Smooth and Ground
        if not hasattr(self, "_last_root_pos"): self._last_root_pos = target_pos
        self._last_root_pos = self._last_root_pos * 0.7 + target_pos * 0.3
        
        # Force grounded at Z=0
        self._avatar.setPos(self._last_root_pos[0], self._last_root_pos[1], 0)

        # 2. Capture Rest Data (Frame 0)
        if frame_idx == 0:
            self._human_rest_joints = joints
            self._human_rest_mid_hip = mid_hip
            self._human_rest_mid_sho = mid_sho
            self._bone_rest_data = {} 
            
            for bone_name, (start_name, end_name) in BONE_MAP.items():
                bone_np = self._controlled_joints.get(bone_name)
                if not bone_np: continue
                
                # Capture the ACTUAL rest vector of the bone in the avatar's local space
                child_joint = None
                for child in bone_np.getChildren():
                    if 'joint' in child.getName().lower() or 'mixamo' in child.getName().lower():
                        child_joint = child
                        break
                
                avatar_rest_vec = child_joint.getPos() if child_joint else Vec3(0, 0, 1)
                if avatar_rest_vec.length() < 1e-6: avatar_rest_vec = Vec3(0, 0, 1)
                avatar_rest_vec.normalize()

                self._bone_rest_data[bone_name] = {
                    'initial_local_quat': bone_np.getQuat(),
                    'avatar_rest_vec': avatar_rest_vec
                }

        # Debug Print playback
        if frame_idx % 30 == 0:
            print(f"[studio] Playback at frame {frame_idx}/{len(self._frames)}")

        # 3. Process Bones
        for bone_name, (start_name, end_name) in BONE_MAP.items():
            if DEBUG_ONLY_BONE and bone_name != DEBUG_ONLY_BONE:
                continue
                
            bone_np = self._controlled_joints.get(bone_name)
            rest_info = self._bone_rest_data.get(bone_name)
            if not bone_np or not rest_info: continue

            # --- Human Pose Vector ---
            if 'torso' in bone_name or 'Spine' in bone_name:
                h_rest_vec = joint_vec({'h': self._human_rest_mid_hip, 's': self._human_rest_mid_sho}, 'h', 's')
                h_now_vec = joint_vec({'h': mid_hip, 's': mid_sho}, 'h', 's')
                conf = 1.0
            else:
                h_rest_vec = joint_vec(self._human_rest_joints, start_name, end_name)
                h_now_vec = joint_vec(joints, start_name, end_name)
                conf = min(joints[start_name]['v'], joints[end_name]['v'])

            if h_rest_vec.length() < 1e-6 or h_now_vec.length() < 1e-6: continue
            h_rest_vec.normalize()
            h_now_vec.normalize()

            # --- Target Orientation ---
            # We want to rotate our 'avatar_rest_vec' to match the human 'h_now_vec'
            # (Note: This is a simplified T-pose assumption where rest_vec matches h_rest_vec)
            # For better accuracy, we calculate the delta rotation from h_rest to h_now
            h_delta_quat = LQuaternionf()
            h_axis = h_rest_vec.cross(h_now_vec)
            h_angle = h_rest_vec.angleDeg(h_now_vec)
            
            limit = 120
            if 'torso' in bone_name: limit = 40
            elif 'leg' in bone_name: limit = 95
            
            damping = 0.85 if limit > 50 else 0.5
            final_angle = min(max(h_angle * damping, -limit), limit)

            if h_axis.length() > 1e-6:
                h_axis.normalize()
                h_delta_quat.setFromAxisAngle(final_angle, h_axis)

            # Apply delta to the rest orientation
            target_q = h_delta_quat * rest_info['initial_local_quat']
            
            # Slerp-like smoothing
            alpha = 0.3 if conf > 0.4 else 0.1
            current_q = bone_np.getQuat()
            smooth_q = current_q * (1.0 - alpha) + target_q * alpha
            smooth_q.normalize()
            bone_np.setQuat(smooth_q)

        self._avatar.update()
        if SHOW_DEBUG_SKELETON:
            self._draw_debug_skeleton(joints, mid_hip, mid_sho)

    def _draw_debug_skeleton(self, joints, mid_hip, mid_sho):
        from panda3d.core import LineSegs
        self._debug_np.node().removeAllChildren()
        ls = LineSegs()
        ls.setThickness(2.0)
        s = 3.0 # Larger debug skeleton
        def p(j): return Vec3(j['x']*s, -j['z']*s, -j['y']*s)
        ls.setColor(1, 1, 0, 1)
        ls.moveTo(p(mid_hip))
        ls.drawTo(p(mid_sho))
        ls.setColor(0, 1, 1, 1)
        for _, (start, end) in BONE_MAP.items():
            if isinstance(start, str) and isinstance(end, str):
                ls.moveTo(p(joints[start]))
                ls.drawTo(p(joints[end]))
        self._debug_np.attachNewNode(ls.create())


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  MoCap 3D Avatar Studio — Local Native App")
    print("=" * 60)

    glb_path, csv_path = pick_files()
    if not glb_path:
        print("[studio] No model selected. Exiting.")
        sys.exit(0)

    frames = []
    fps = 30.0
    if csv_path and os.path.isfile(csv_path):
        print(f"[studio] Loading CSV: {os.path.basename(csv_path)}")
        frames = load_csv_frames(csv_path)
        print(f"[studio] Loaded {len(frames)} frames")
    else:
        print("[studio] No CSV loaded — static T-pose display only.")

    print(f"[studio] Starting 3D engine with model: {os.path.basename(glb_path)}")
    app = AvatarStudio(glb_path, frames, fps=fps)
    app.run()
