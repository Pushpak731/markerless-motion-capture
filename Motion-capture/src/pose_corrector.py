import numpy as np
import time
from src.one_euro_filter import OneEuroFilter
from config import (
    VISIBILITY_HARD_GATE,
    FILTER_MIN_CUTOFF,
    FILTER_BETA,
    BONE_LENGTH_CALIBRATION_FRAMES,
    BONE_LENGTH_MIN_CONFIDENCE,
    MIN_VALID_BONES_PER_CALIBRATION_FRAME,
)

class PoseCorrector:
    def __init__(self):
        self.calibrating = True
        self.calibration_frames = 0
        self.max_calibration_frames = BONE_LENGTH_CALIBRATION_FRAMES
        self.min_valid_bones = MIN_VALID_BONES_PER_CALIBRATION_FRAME
        # Buffer for World (meters)
        self.bone_length_buffer_world = {} 
        
        # Final learned lengths
        self.ref_bone_lengths_world = {}
        
        # Backward compatibility
        self.ref_bone_lengths = self.ref_bone_lengths_world
        self.bone_length_buffer = self.bone_length_buffer_world
        
        # One Euro Filters for each landmark (Index -> Filter)
        self.filters = {}
        # Params from config
        self.min_cutoff = FILTER_MIN_CUTOFF
        self.beta = FILTER_BETA 
        
        # Hierarchy: Child -> Parent
        # 12: R_Shoulder, 14: R_Elbow, 16: R_Wrist
        # 11: L_Shoulder, 13: L_Elbow, 15: L_Wrist
        # 24: R_Hip, 26: R_Knee, 28: R_Ankle
        # 23: L_Hip, 25: L_Knee, 27: L_Ankle
        self.hierarchy = {
            14: 12, # R_Elbow -> R_Shoulder
            16: 14, # R_Wrist -> R_Elbow
            13: 11, # L_Elbow -> L_Shoulder
            15: 13, # L_Wrist -> L_Elbow
            26: 24, # R_Knee -> R_Hip
            28: 26, # R_Ankle -> R_Knee
            25: 23, # L_Knee -> L_Hip
            27: 25  # L_Ankle -> L_Knee
        }

    def process(self, results, timestamp_ms=None, frame_quality=None):
        """
        Input: MediaPipe results object.
        Output: Modified MediaPipe results object (In-Place).
        """
        if not results.get('pose'):
            return results

        # 1. Preserve Raw Landmarks before correction
        pose_res = results['pose']
        if not hasattr(pose_res, 'raw_pose_landmarks'):
            pose_res.raw_pose_landmarks = self._copy_landmarks(pose_res.pose_landmarks)
        if not hasattr(pose_res, 'raw_pose_world_landmarks'):
            pose_res.raw_pose_world_landmarks = self._copy_landmarks(pose_res.pose_world_landmarks)

        # Initialize Metadata
        results['pose_correction_metadata'] = {
            'normalized': [{} for _ in range(len(pose_res.pose_landmarks))] if pose_res.pose_landmarks else [],
            'world': [{} for _ in range(len(pose_res.pose_world_landmarks))] if pose_res.pose_world_landmarks else []
        }

        # 2. Correct Normalized Landmarks (for Display)
        # BYPASSED to eliminate visual lag and perspective-distorted constraints.
        # Raw MediaPipe landmarks are returned directly to the visualizer.
        if pose_res.pose_landmarks:
            for i in range(len(pose_res.pose_landmarks)):
                meta = results['pose_correction_metadata']['normalized'][i]
                for j in range(33): # 33 joints in MediaPipe Pose
                    meta[j] = "raw_mediapipe"
            
        # 3. Correct World Landmarks (for Physics/Metrics)
        frame_time = (timestamp_ms / 1000.0) if timestamp_ms is not None else time.time()
        quality_score = 1.0
        if frame_quality is not None:
            if isinstance(frame_quality, dict):
                quality_score = float(frame_quality.get('score', 1.0) or 1.0)
            else:
                quality_score = float(getattr(frame_quality, 'score', 1.0) or 1.0)
        quality_score = max(0.0, min(1.0, quality_score))

        if pose_res.pose_world_landmarks:
            for i in range(len(pose_res.pose_world_landmarks)):
                meta = results['pose_correction_metadata']['world'][i]
                self._correct_skeleton(pose_res.pose_world_landmarks[i], frame_time, is_world=True, 
                                      quality_score=quality_score, frame_quality=frame_quality, 
                                      metadata=meta)

        return results

    def _copy_landmarks(self, landmarks_list):
        """Deep copy of MediaPipe landmarks for preservation."""
        if not landmarks_list: return None
        copied_list = []
        for person in landmarks_list:
            person_copied = []
            for lm in person:
                # Use a simple class to mimic the landmark object
                class LandmarkStub:
                    def __init__(self, x, y, z, v, p):
                        self.x = x
                        self.y = y
                        self.z = z
                        self.visibility = v
                        self.presence = p
                person_copied.append(LandmarkStub(lm.x, lm.y, lm.z, lm.visibility, getattr(lm, 'presence', 0.0)))
            copied_list.append(person_copied)
        return copied_list

    def _correct_skeleton(self, landmarks, frame_time, is_world=False, quality_score=1.0, frame_quality=None, metadata=None):
        """Shared logic for both landmark types."""
        # Create coords and visibility map
        coords = {}
        visibility = {}
        for i, lm in enumerate(landmarks):
            coords[i] = np.array([lm.x, lm.y, lm.z])
            visibility[i] = lm.visibility

        # 1. 1 Euro Smoothing + Visibility Handling
        # Use prefix "w_" for world landmarks
        prefix = "w_"
        ref_lengths = self.ref_bone_lengths_world

        for i in coords:
            key = f"{prefix}{i}"
            vis = visibility.get(i, 1.0)
            blended_quality = max(0.0, min(1.0, vis * quality_score))
            source = "raw"
            
            # --- VISIBILITY HARD GATE ---
            # If confidence is low, ignore this frame's update and reconstruct or HOLD.
            if vis < VISIBILITY_HARD_GATE:
                reconstructed = False
                parent_idx = self.hierarchy.get(i)
                
                # RECONSTRUCTION: If child is bad but parent is good, reconstruct limb segment
                if parent_idx is not None and visibility.get(parent_idx, 0.0) >= VISIBILITY_HARD_GATE:
                    if i in ref_lengths and key in self.filters:
                        prev_child = self.filters[key].x_prev
                        parent_key = f"{prefix}{parent_idx}"
                        prev_parent = self.filters[parent_key].x_prev if parent_key in self.filters else None
                        
                        if prev_parent is not None:
                            prev_vec = prev_child - prev_parent
                            prev_len = np.linalg.norm(prev_vec)
                            if prev_len > 1e-6:
                                direction = prev_vec / prev_len
                                # Reconstruct child from parent using previous direction and ref length
                                coords[i] = coords[parent_idx] + direction * ref_lengths[i]
                                source = "reconstructed_from_parent"
                                reconstructed = True

                if not reconstructed:
                    # If we have a filter, use its last valid state (LATCH/HOLD)
                    if key in self.filters:
                        coords[i] = self.filters[key].x_prev
                        source = "held_previous"
                    else:
                        # Skip initialization from low-confidence landmarks
                        source = "rejected_uninitialized"
                        # Keep raw coords but mark as uninitialized
                
                if metadata is not None: metadata[i] = source
                continue
            # -----------------------------

            # NORMAL UPDATE (vis >= VISIBILITY_HARD_GATE)
            if key not in self.filters:
                # Initialize filter only from high-confidence landmarks
                self.filters[key] = OneEuroFilter(frame_time, coords[i], min_cutoff=self.min_cutoff, beta=self.beta)
                source = "raw"
            else:
                if blended_quality < 0.1:
                    coords[i] = self.filters[key].x_prev
                    source = "held_previous"
                else:
                    held = self.filters[key].x_prev
                    adjusted = blended_quality * coords[i] + (1.0 - blended_quality) * held
                    coords[i] = self.filters[key](frame_time, adjusted)
                    source = "smoothed"
            
            if metadata is not None: metadata[i] = source

        # 2. Calibration vs Correction
        if self.calibrating:
            self._calibrate(coords, visibility, frame_quality)
        else:
             # Apply constraints to World
             coords = self._apply_constraints(coords, visibility, metadata)

        # 3. Write back
        for i, lm in enumerate(landmarks):
            lm.x, lm.y, lm.z = coords[i][0], coords[i][1], coords[i][2]

    def _calibrate(self, coords, visibility, frame_quality=None):
        """Learn the user's bone lengths (World Units)."""
        usable = True
        if frame_quality is not None:
            if isinstance(frame_quality, dict):
                usable = bool(frame_quality.get('pose_usable', True))
            else:
                usable = bool(getattr(frame_quality, 'pose_usable', True))

        if not usable:
            return

        buffer = self.bone_length_buffer_world
        samples_collected_this_frame = 0

        for child, parent in self.hierarchy.items():
            vis_child = visibility.get(child, 0.0)
            vis_parent = visibility.get(parent, 0.0)

            # Require both joints to exceed min confidence to count toward calibration
            if vis_child < BONE_LENGTH_MIN_CONFIDENCE or vis_parent < BONE_LENGTH_MIN_CONFIDENCE:
                continue

            dist = np.linalg.norm(coords[child] - coords[parent])
            if not np.isfinite(dist) or dist < 1e-6:
                continue

            if child not in buffer:
                buffer[child] = []
            buffer[child].append(dist)
            samples_collected_this_frame += 1

        # Only increment calibration frame counter when enough valid bones collected.
        if samples_collected_this_frame >= self.min_valid_bones:
            self.calibration_frames += 1

        if self.calibration_frames >= self.max_calibration_frames:
            print(f"Pose Corrector: Calibration Complete ({self.calibration_frames} frames). Physics Constraints Active.")
            
            # Process World
            for child, lengths in self.bone_length_buffer_world.items():
                if len(lengths) >= 10: # Minimum 10 samples per bone to accept reference
                    self.ref_bone_lengths_world[child] = np.median(lengths)
            
            self.bone_length_buffer_world.clear()
            self.calibrating = False

    def _apply_constraints(self, coords, visibility, metadata=None):
        """Force bones to match reference lengths."""
        order = [14, 16, 13, 15, 26, 28, 25, 27]
        ref_lengths = self.ref_bone_lengths_world
        
        for child in order:
            parent = self.hierarchy[child]
            
            current_vec = coords[child] - coords[parent]
            current_len = np.linalg.norm(current_vec)
            
            if current_len < 1e-6: continue
            
            # Check if we have a reference length for this bone
            if child not in ref_lengths: continue
                
            target_len = ref_lengths[child]
            
            # Continuous Visibility Blending (Quadratic)
            vis = visibility.get(child, 1.0)
            
            # Weight = vis^3 (More forgiving for high visibility)
            alpha = vis * vis * vis
            
            # strictness (Model Weight) = 1 - alpha
            strictness = 1.0 - alpha
            
            # Clamp strictness
            if strictness < 0: strictness = 0.0
            if strictness > 1: strictness = 1.0
            
            ideal_pos = coords[parent] + (current_vec / current_len) * target_len
            
            # If we are applying significant correction, update metadata
            if strictness > 0.3 and metadata is not None:
                # Only overwrite if not already reconstructed
                if metadata.get(child) != "reconstructed_from_parent":
                    metadata[child] = "constrained"

            coords[child] = (1 - strictness) * coords[child] + strictness * ideal_pos
            
        # 2. Physiological Constraints (IK Lite) - Currently Placeholder
        coords = self._apply_physiological_limits(coords)
        
        return coords

    def _apply_physiological_limits(self, coords):
        """
        Placeholder for physiological constraints (hyperextension prevention).
        Currently disabled to remain 'honest' as accurate limiting requires 
        stable torso-relative reference frames not yet implemented in this version.
        """
        return coords
