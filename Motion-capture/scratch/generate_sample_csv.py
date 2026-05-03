import csv
import math
import os

# Create a sample motion CSV for 3D Studio testing
# Matches the EXACT format of _raw_3d_nodes.csv

FILENAME = "sample_motion.csv"

def generate_sample():
    header = ["frame_idx", "timestamp_ms"]
    joints = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_heel", "right_heel", "left_foot_index", "right_foot_index"
    ]
    
    for j in joints:
        header.extend([f"{j}_x", f"{j}_y", f"{j}_z", f"{j}_v"])

    with open(FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for frame in range(200):
            row = [frame, frame * 33] # ~30fps
            
            # Simple T-Pose with waving right arm
            t = frame / 30.0 * math.pi
            wave = math.sin(t) * 0.3
            leg_wave = math.cos(t) * 0.1
            
            for j in joints:
                # Default positions (simplified world coordinates in meters)
                # Hip is at (0, 0, 0)
                x, y, z = 0, 0, 0
                
                # Torso
                if j == "left_shoulder":  x, y, z = -0.2, -0.5, 0
                if j == "right_shoulder": x, y, z = 0.2, -0.5, 0
                if j == "left_hip":       x, y, z = -0.1, 0, 0
                if j == "right_hip":      x, y, z = 0.1, 0, 0
                
                # Arms (Right arm waves)
                if j == "left_elbow":     x, y, z = -0.4, -0.5, 0
                if j == "left_wrist":     x, y, z = -0.6, -0.5, 0
                
                if j == "right_elbow":    x, y, z = 0.4, -0.5 + wave, wave
                if j == "right_wrist":    x, y, z = 0.6, -0.5 + wave*2, wave*2
                
                # Legs (Left leg steps)
                if j == "left_knee":      x, y, z = -0.1, 0.4, -leg_wave
                if j == "left_ankle":     x, y, z = -0.1, 0.8, -leg_wave*1.5
                if j == "right_knee":     x, y, z = 0.1, 0.4, 0
                if j == "right_ankle":    x, y, z = 0.1, 0.8, 0
                
                row.extend([x, y, z, 1.0])
            
            writer.writerow(row)

    print(f"Generated {FILENAME} with 200 frames of waving/stepping motion.")

if __name__ == "__main__":
    generate_sample()
