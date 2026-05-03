import csv
import math
import os

# Create a FULL-BODY motion CSV
# Every limb moves to prove the retargeting is working everywhere

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
        
        for frame in range(100):
            row = [frame, frame * 33]
            t = frame / 10.0
            
            # Different phases for different limbs
            wave_l = math.sin(t) * 0.5
            wave_r = math.cos(t) * 0.5
            kick_l = math.sin(t * 0.7) * 0.4
            kick_r = math.cos(t * 0.7) * 0.4
            
            for j in joints:
                x, y, z = 0, 0, 0
                
                # Shoulders
                if j == "left_shoulder":  x, y, z = -0.2, -0.5, 0
                if j == "right_shoulder": x, y, z = 0.2, -0.5, 0
                
                # Left Arm (Waves)
                if j == "left_elbow":     x, y, z = -0.4, -0.5 + wave_l, wave_l
                if j == "left_wrist":     x, y, z = -0.6, -0.5 + wave_l*2, wave_l*2
                
                # Right Arm (Waves)
                if j == "right_elbow":    x, y, z = 0.4, -0.5 + wave_r, wave_r
                if j == "right_wrist":    x, y, z = 0.6, -0.5 + wave_r*2, wave_r*2
                
                # Hips
                if j == "left_hip":       x, y, z = -0.1, 0, 0
                if j == "right_hip":      x, y, z = 0.1, 0, 0
                
                # Left Leg (Kicks)
                if j == "left_knee":      x, y, z = -0.1, 0.4 + kick_l, kick_l
                if j == "left_ankle":     x, y, z = -0.1, 0.8 + kick_l*2, kick_l*2
                
                # Right Leg (Kicks)
                if j == "right_knee":     x, y, z = 0.1, 0.4 + kick_r, kick_r
                if j == "right_ankle":    x, y, z = 0.1, 0.8 + kick_r*2, kick_r*2
                
                row.extend([x, y, z, 1.0])
            
            writer.writerow(row)

    print(f"Generated {FILENAME} with FULL BODY motion.")

if __name__ == "__main__":
    generate_sample()
