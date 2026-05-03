import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_charts_for_csv(csv_path, output_dir="analysis_results"):
    print(f"Processing: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("CSV is empty.")
        return

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Bone Variance Graph
    plt.figure(figsize=(12, 5))
    plt.plot(df['frame_idx'], df['bone_variance_mean'], label='Mean Bone Variance', color='blue', linewidth=2)
    
    # Plot a couple of specific bones if they exist
    if 'Variance_UpperArm_L' in df.columns:
        plt.plot(df['frame_idx'], df['Variance_UpperArm_L'], label='L Upper Arm Variance', color='red', alpha=0.5)
    if 'Variance_UpperLeg_L' in df.columns:
        plt.plot(df['frame_idx'], df['Variance_UpperLeg_L'], label='L Upper Leg Variance', color='green', alpha=0.5)
        
    plt.title(f'Bone Variance Over Time ({base_name})')
    plt.xlabel('Frame Index')
    plt.ylabel('Variance (Meters squared)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_variance.png"), dpi=150)
    plt.close()

    # 2. Jitter Analysis (Motion Jump & Consistency)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    color = 'tab:orange'
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Motion Jump (m/frame)', color=color)
    ax1.plot(df['frame_idx'], df['motion_jump'], color=color, alpha=0.8, label='Motion Jump (Jitter)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Consistency Score', color=color)  
    ax2.plot(df['frame_idx'], df['consistency_score'], color=color, alpha=0.5, label='Consistency Score')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.1)

    plt.title(f'Jitter & Consistency Analysis ({base_name})')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_jitter.png"), dpi=150)
    plt.close()

    # 3. State Gantt Chart
    plt.figure(figsize=(12, 3))
    
    # Map states to numbers
    state_mapping = {'missing': 0, 'low_confidence': 1, 'detected': 2}
    colors = {0: 'red', 1: 'orange', 2: 'green'}
    
    # Provide a default mapping if frame_state contains unexpected values
    df['state_num'] = df['frame_state'].map(lambda x: state_mapping.get(x, 0))
    
    # Create scatter plot for state
    plt.scatter(df['frame_idx'], df['state_num'], 
                c=df['state_num'].map(colors), 
                marker='|', s=500)
    
    # Overlay usable flag as black dots at the top
    usable_frames = df[df['pose_usable'] == 1]
    plt.scatter(usable_frames['frame_idx'], [2.5] * len(usable_frames), 
                color='black', marker='.', s=10, label='Pose Usable Flag')

    plt.yticks([0, 1, 2, 2.5], ['Missing', 'Low Confidence', 'Detected', 'Usable Check'])
    plt.ylim(-0.5, 3.0)
    plt.title(f'Tracking State Timeline ({base_name})')
    plt.xlabel('Frame Index')
    plt.grid(True, axis='x', alpha=0.3)
    
    # Custom legend
    red_patch = mpatches.Patch(color='red', label='Missing')
    orange_patch = mpatches.Patch(color='orange', label='Low Conf')
    green_patch = mpatches.Patch(color='green', label='Detected')
    plt.legend(handles=[green_patch, orange_patch, red_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_gantt.png"), dpi=150)
    plt.close()
    
    print(f"  -> Generated 3 charts in {output_dir}/")

def main():
    csv_files = glob.glob("data/offline_validation_runs/*_metrics.csv")
    if not csv_files:
        print("No CSV metrics found in data/offline_validation_runs/")
        return
        
    for csv_file in csv_files:
        generate_charts_for_csv(csv_file)

if __name__ == "__main__":
    main()
