#!/usr/bin/env python3
"""
generate_analysis_charts.py
Generates comprehensive analytical charts for offline validation runs.
Run from the project root:  python tools/generate_analysis_charts.py
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings('ignore')

# ─── Config ────────────────────────────────────────────────────────────────
STYLE = {
    'bg':        '#1a1515',
    'panel':     '#251f1f',
    'border':    '#3d2f2f',
    'cream':     '#e8d9c0',
    'latte':     '#c4a882',
    'mocha':     '#8b6f5e',
    'cyan':      '#22d3ee',
    'red':       '#f87171',
    'green':     '#4ade80',
    'orange':    '#fb923c',
    'purple':    '#a78bfa',
    'yellow':    '#facc15',
}

plt.rcParams.update({
    'figure.facecolor': STYLE['bg'],
    'axes.facecolor':   STYLE['panel'],
    'axes.edgecolor':   STYLE['border'],
    'axes.labelcolor':  STYLE['cream'],
    'text.color':       STYLE['cream'],
    'xtick.color':      STYLE['latte'],
    'ytick.color':      STYLE['latte'],
    'grid.color':       STYLE['border'],
    'grid.alpha':       0.5,
    'legend.facecolor': STYLE['panel'],
    'legend.edgecolor': STYLE['border'],
    'font.family':      'monospace',
})

BONE_COLS = [
    'Length_UpperArm_L', 'Length_LowerArm_L',
    'Length_UpperArm_R', 'Length_LowerArm_R',
    'Length_UpperLeg_L', 'Length_LowerLeg_L',
    'Length_UpperLeg_R', 'Length_LowerLeg_R',
]
BONE_COLORS = [
    STYLE['cyan'], STYLE['cyan'],
    STYLE['purple'], STYLE['purple'],
    STYLE['green'], STYLE['green'],
    STYLE['orange'], STYLE['orange'],
]
BONE_LABELS = [
    'L UpperArm', 'L LowerArm',
    'R UpperArm', 'R LowerArm',
    'L UpperLeg', 'L LowerLeg',
    'R UpperLeg', 'R LowerLeg',
]

# ─── Chart Generators ──────────────────────────────────────────────────────

def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=STYLE['bg'])
    plt.close(fig)
    print(f"  ✓  {os.path.basename(path)}")


def chart_variance(df, base, out_dir):
    """Bone variance over time with anomaly shading."""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(df['frame_idx'], df['bone_variance_mean'],
                    alpha=0.25, color=STYLE['cyan'])
    ax.plot(df['frame_idx'], df['bone_variance_mean'],
            color=STYLE['cyan'], lw=1.5, label='Mean Bone Variance')

    # Per-bone
    for col, color, label in zip(BONE_COLS, BONE_COLORS, BONE_LABELS):
        if col in df.columns:
            ax.plot(df['frame_idx'], df[col], color=color, lw=0.7,
                    alpha=0.45, linestyle='--', label=label)

    # Threshold line
    threshold = 0.03
    ax.axhline(threshold, color=STYLE['red'], lw=1, linestyle=':', label='QA threshold 0.03')

    ax.set_title(f'Bone Length Variance — {base}', pad=12, fontsize=11)
    ax.set_xlabel('Frame'); ax.set_ylabel('Variance (m²)')
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(True)
    save(fig, os.path.join(out_dir, f'{base}_variance.png'))


def chart_jitter(df, base, out_dir):
    """Motion jitter and consistency dual-axis with anomaly bands."""
    high_jitter = df['motion_jump'] > 0.2

    fig, ax1 = plt.subplots(figsize=(13, 4))
    # Shade anomalous regions
    for _, row in df[high_jitter].iterrows():
        ax1.axvspan(row['frame_idx'] - 0.5, row['frame_idx'] + 0.5,
                    alpha=0.25, color=STYLE['red'])

    ax1.fill_between(df['frame_idx'], df['motion_jump'],
                     alpha=0.3, color=STYLE['orange'])
    ax1.plot(df['frame_idx'], df['motion_jump'],
             color=STYLE['orange'], lw=1.5, label='Motion Jump (jitter)')
    ax1.set_ylabel('Motion Jump (m/frame)', color=STYLE['orange'])
    ax1.tick_params(axis='y', labelcolor=STYLE['orange'])
    ax1.axhline(0.2, color=STYLE['red'], lw=1, linestyle=':', label='Jitter threshold')

    ax2 = ax1.twinx()
    ax2.plot(df['frame_idx'], df['consistency_score'],
             color=STYLE['cyan'], lw=1.5, alpha=0.8, label='Consistency Score')
    ax2.set_ylabel('Consistency Score', color=STYLE['cyan'])
    ax2.tick_params(axis='y', labelcolor=STYLE['cyan'])
    ax2.set_ylim(0, 1.1)

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

    n_bad = high_jitter.sum()
    ax1.set_title(
        f'Jitter & Consistency — {base}  [{n_bad} anomalous frames >0.2m highlighted]',
        pad=12, fontsize=10)
    ax1.set_xlabel('Frame')
    ax1.grid(True)
    save(fig, os.path.join(out_dir, f'{base}_jitter.png'))


def chart_gantt(df, base, out_dir):
    """Tracking state Gantt chart showing per-frame status."""
    state_map = {'detected': 2, 'interpolated': 1, 'tracking_loss': 0, 'missing': 0}
    color_map = {2: STYLE['green'], 1: STYLE['yellow'], 0: STYLE['red']}

    df = df.copy()
    df['state_num'] = df['frame_state'].map(lambda x: state_map.get(x, 0))
    df['s_color']   = df['state_num'].map(color_map)

    fig, ax = plt.subplots(figsize=(13, 2.5))
    ax.scatter(df['frame_idx'], df['state_num'],
               c=df['s_color'], marker='|', s=600, linewidths=1.5)

    usable = df[df['pose_usable'] == 1]
    ax.scatter(usable['frame_idx'], [2.6] * len(usable),
               color=STYLE['cyan'], marker='.', s=8, label='Usable frames')

    ax.set_yticks([0, 1, 2, 2.6])
    ax.set_yticklabels(['Loss / Missing', 'Interpolated', 'Detected', 'Usable'])
    ax.set_ylim(-0.5, 3.1)
    ax.set_xlabel('Frame')
    ax.set_title(f'Tracking State Timeline — {base}', pad=10, fontsize=10)

    patches = [
        mpatches.Patch(color=STYLE['green'],  label='Detected'),
        mpatches.Patch(color=STYLE['yellow'], label='Interpolated'),
        mpatches.Patch(color=STYLE['red'],    label='Loss / Missing'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=8)
    ax.grid(True, axis='x')
    save(fig, os.path.join(out_dir, f'{base}_gantt.png'))


def chart_bone_lengths(df, base, out_dir):
    """Individual bone length traces — shows symmetry and drift."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
    fig.suptitle(f'Per-Bone Length Traces — {base}', fontsize=11, y=1.01)

    for ax, col, color, label in zip(axes.flat, BONE_COLS, BONE_COLORS, BONE_LABELS):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        data = df[col].replace(0, np.nan)
        ax.plot(df['frame_idx'], data, color=color, lw=1.2)

        ref_col = f'Reference_Length_{col.replace("Length_", "")}'
        if ref_col in df.columns:
            ref = df[ref_col].replace(0, np.nan).median()
            if pd.notna(ref) and ref > 0:
                ax.axhline(ref, color=STYLE['red'], lw=1, linestyle='--', alpha=0.7, label=f'Ref={ref:.3f}m')
                ax.legend(fontsize=7)

        ax.set_title(label, fontsize=9)
        ax.grid(True)

    for ax in axes[0]: ax.set_ylabel('Length (m)', fontsize=7)
    for ax in axes[1]: ax.set_xlabel('Frame', fontsize=7)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f'{base}_bone_lengths.png'))


def chart_symmetry(df, base, out_dir):
    """Left vs Right limb length symmetry scatter plots."""
    pairs = [
        ('Length_UpperArm_L', 'Length_UpperArm_R', 'Upper Arm L vs R', STYLE['cyan']),
        ('Length_LowerArm_L', 'Length_LowerArm_R', 'Lower Arm L vs R', STYLE['purple']),
        ('Length_UpperLeg_L', 'Length_UpperLeg_R', 'Upper Leg L vs R', STYLE['green']),
        ('Length_LowerLeg_L', 'Length_LowerLeg_R', 'Lower Leg L vs R', STYLE['orange']),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Left vs Right Symmetry (should cluster on diagonal) — {base}', fontsize=10, y=1.01)

    for ax, (l_col, r_col, title, color) in zip(axes, pairs):
        if l_col not in df.columns or r_col not in df.columns:
            continue
        l = df[l_col].replace(0, np.nan)
        r = df[r_col].replace(0, np.nan)
        valid = l.notna() & r.notna()
        ax.scatter(l[valid], r[valid], color=color, alpha=0.4, s=10)

        # Perfect symmetry line
        mn = min(l[valid].min(), r[valid].min())
        mx = max(l[valid].max(), r[valid].max())
        ax.plot([mn, mx], [mn, mx], color=STYLE['red'], lw=1.5, linestyle='--', label='Perfect symmetry')

        # Asymmetry score
        asym = (l[valid] - r[valid]).abs().mean()
        ax.set_title(f'{title}\nMean Δ={asym:.4f}m', fontsize=8)
        ax.set_xlabel('Left (m)', fontsize=7)
        ax.set_ylabel('Right (m)', fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True)

    fig.tight_layout()
    save(fig, os.path.join(out_dir, f'{base}_symmetry.png'))


def chart_visibility(df, base, out_dir):
    """Pose visibility and quality score heatmap over time."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(f'Visibility & Quality Over Time — {base}', fontsize=11)

    # 1. Mean visibility
    axes[0].fill_between(df['frame_idx'], df['pose_visibility_mean'],
                         alpha=0.3, color=STYLE['cyan'])
    axes[0].plot(df['frame_idx'], df['pose_visibility_mean'],
                 color=STYLE['cyan'], lw=1.5)
    axes[0].axhline(0.5, color=STYLE['red'], lw=1, linestyle=':', label='Min threshold')
    axes[0].set_ylabel('Mean Visibility')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    # 2. Min visibility (weakest tracked joint)
    axes[1].fill_between(df['frame_idx'], df['pose_visibility_min'],
                         alpha=0.3, color=STYLE['orange'])
    axes[1].plot(df['frame_idx'], df['pose_visibility_min'],
                 color=STYLE['orange'], lw=1.5)
    axes[1].set_ylabel('Min Visibility')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True)

    # 3. Quality Score
    axes[2].fill_between(df['frame_idx'], df['quality_score'],
                         alpha=0.3, color=STYLE['green'])
    axes[2].plot(df['frame_idx'], df['quality_score'],
                 color=STYLE['green'], lw=1.5)
    axes[2].set_ylabel('Quality Score')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True)

    fig.tight_layout()
    save(fig, os.path.join(out_dir, f'{base}_visibility.png'))


def chart_processing_fps(df, base, out_dir):
    """Per-frame processing time and effective FPS."""
    fps = 1.0 / df['processing_time_s'].replace(0, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    fig.suptitle(f'Processing Performance — {base}', fontsize=11)

    axes[0].plot(df['frame_idx'], df['processing_time_s'],
                 color=STYLE['orange'], lw=1.2)
    axes[0].fill_between(df['frame_idx'], df['processing_time_s'],
                         alpha=0.2, color=STYLE['orange'])
    axes[0].axhline(df['processing_time_s'].median(), color=STYLE['yellow'],
                    lw=1.2, linestyle='--',
                    label=f'Median {df["processing_time_s"].median():.2f}s')
    axes[0].set_ylabel('Processing Time (s)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].plot(df['frame_idx'], fps, color=STYLE['cyan'], lw=1.2)
    axes[1].axhline(10, color=STYLE['red'], lw=1.2, linestyle='--', label='10 FPS target')
    axes[1].axhline(fps.median(), color=STYLE['green'], lw=1, linestyle='--',
                    label=f'Median {fps.median():.1f} FPS')
    axes[1].set_ylabel('Effective FPS')
    axes[1].set_xlabel('Frame')
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    fig.tight_layout()
    save(fig, os.path.join(out_dir, f'{base}_fps.png'))


def chart_summary_dashboard(df, base, out_dir):
    """Single-page summary dashboard for quick review."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(STYLE['bg'])

    # Title
    total = len(df)
    detected = (df['frame_state'] == 'detected').sum()
    loss = total - detected

    fig.text(0.02, 0.97, f'Summary Dashboard — {base}', fontsize=13,
             fontweight='bold', color=STYLE['cream'], va='top')
    fig.text(0.02, 0.94, 
             f'Frames: {total} | Detected: {detected} ({100*detected/total:.1f}%) | '
             f'Loss: {loss} | Jitter spikes: {(df["motion_jump"]>0.2).sum()}',
             fontsize=9, color=STYLE['latte'], va='top')

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.88, bottom=0.06)

    # -- Panel 1: Variance
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['frame_idx'], df['bone_variance_mean'], color=STYLE['cyan'], lw=1.5)
    ax1.fill_between(df['frame_idx'], df['bone_variance_mean'], alpha=0.2, color=STYLE['cyan'])
    ax1.axhline(0.03, color=STYLE['red'], lw=1, linestyle=':', label='QA limit')
    ax1.set_title('Bone Variance', fontsize=9); ax1.grid(True); ax1.legend(fontsize=7)

    # -- Panel 2: Jitter
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(df['frame_idx'], df['motion_jump'], color=STYLE['orange'], lw=1.3)
    ax2.fill_between(df['frame_idx'], df['motion_jump'], alpha=0.2, color=STYLE['orange'])
    ax2.axhline(0.2, color=STYLE['red'], lw=1, linestyle=':', label='Jitter limit')
    ax2.set_title('Motion Jump (Jitter)', fontsize=9); ax2.grid(True); ax2.legend(fontsize=7)

    # -- Panel 3: FPS
    fps = 1.0 / df['processing_time_s'].replace(0, np.nan)
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.plot(df['frame_idx'], fps, color=STYLE['green'], lw=1.2)
    ax3.axhline(10, color=STYLE['red'], lw=1, linestyle=':', label='10 FPS target')
    ax3.set_title('Effective FPS', fontsize=9); ax3.grid(True); ax3.legend(fontsize=7)
    ax3.set_xlabel('Frame')

    # -- Panel 4: State pie
    ax4 = fig.add_subplot(gs[0, 2])
    state_counts = df['frame_state'].value_counts()
    color_pie = [STYLE['green'] if s=='detected' else STYLE['yellow'] if s=='interpolated' else STYLE['red']
                 for s in state_counts.index]
    ax4.pie(state_counts.values, labels=state_counts.index, colors=color_pie,
            autopct='%1.0f%%', textprops={'fontsize': 7, 'color': STYLE['cream']})
    ax4.set_title('Tracking State', fontsize=9)

    # -- Panel 5: Symmetry bar (mean L-R delta per bone pair)
    ax5 = fig.add_subplot(gs[1, 2])
    pairs_labels = ['UpperArm', 'LowerArm', 'UpperLeg', 'LowerLeg']
    pairs_l = ['Length_UpperArm_L','Length_LowerArm_L','Length_UpperLeg_L','Length_LowerLeg_L']
    pairs_r = ['Length_UpperArm_R','Length_LowerArm_R','Length_UpperLeg_R','Length_LowerLeg_R']
    deltas = []
    for l, r in zip(pairs_l, pairs_r):
        if l in df.columns and r in df.columns:
            d = (df[l].replace(0,np.nan) - df[r].replace(0,np.nan)).abs().mean()
            deltas.append(d if pd.notna(d) else 0)
        else:
            deltas.append(0)
    bar_colors = [STYLE['green'] if d < 0.01 else STYLE['orange'] if d < 0.03 else STYLE['red'] for d in deltas]
    ax5.barh(pairs_labels, deltas, color=bar_colors)
    ax5.axvline(0.01, color=STYLE['green'], lw=1, linestyle=':', label='<1cm ok')
    ax5.set_title('L-R Asymmetry (m)', fontsize=9)
    ax5.legend(fontsize=7); ax5.grid(True, axis='x')

    # -- Panel 6: Quality score histogram
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.hist(df['quality_score'].dropna(), bins=20, color=STYLE['purple'], alpha=0.7, edgecolor=STYLE['border'])
    ax6.set_title('Quality Score Dist.', fontsize=9)
    ax6.set_xlabel('Score', fontsize=7); ax6.grid(True)

    save(fig, os.path.join(out_dir, f'{base}_dashboard.png'))


# ─── Main ──────────────────────────────────────────────────────────────────

def generate_all(csv_path, out_dir='analysis_results'):
    print(f"\nProcessing: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    if df.empty:
        print("  Empty CSV, skipping."); return

    base = os.path.splitext(os.path.basename(csv_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    chart_variance(df, base, out_dir)
    chart_jitter(df, base, out_dir)
    chart_gantt(df, base, out_dir)
    chart_bone_lengths(df, base, out_dir)
    chart_symmetry(df, base, out_dir)
    chart_visibility(df, base, out_dir)
    chart_processing_fps(df, base, out_dir)
    chart_summary_dashboard(df, base, out_dir)


if __name__ == '__main__':
    csvs = sorted(glob.glob('data/offline_validation_runs/*_metrics.csv'))
    if not csvs:
        print("No CSV metrics found in data/offline_validation_runs/")
    for csv in csvs:
        generate_all(csv)
    print('\nAll charts saved to analysis_results/')
