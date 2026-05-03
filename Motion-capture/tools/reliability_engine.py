#!/usr/bin/env python3
"""
reliability_engine.py
Offline reliability engine for 2D perspective-aware motion capture validation.
"""

def calculate_reliability(summary: dict) -> dict:
    # 1. Perspective Analysis
    asym = summary.get('asymmetry_analysis', {})
    arm_asym = asym.get('mean_arm_asymmetry', 0.0)
    leg_asym = asym.get('mean_leg_asymmetry', 0.0)
    
    # Bone variance (how much lengths change over time)
    # Higher variance = tracking error. Lower variance + high asymmetry = perspective.
    bone_var = summary.get('bone_variance_mean', 0.0)
    
    warnings = []
    
    # Perspective-Aware Warning
    # If bone length is consistent (low variance) but asymmetrical (L != R),
    # it's likely just a camera angle issue.
    if bone_var < 0.01:
        if arm_asym > 0.15 or leg_asym > 0.15:
            warnings.append({
                'type': 'PERSPECTIVE_WARNING',
                'severity': 'low',
                'msg': f'Persistent asymmetry (arms:{arm_asym:.1%}, legs:{leg_asym:.1%}) with stable bone lengths suggests a perspective/camera angle offset.'
            })
    elif arm_asym > 0.3 or leg_asym > 0.3:
            warnings.append({
                'type': 'CRITICAL_ASYMMETRY',
                'severity': 'high',
                'msg': 'Extreme limb asymmetry detected. Possible tracking failure or subject partially out of frame.'
            })

    # 2. Jitter / Tracking Error Analysis
    jitter = summary.get('jitter_analysis', {})
    jitter_ratio = jitter.get('high_jitter_ratio', 0.0)
    
    if jitter_ratio > 0.1:
        warnings.append({
            'type': 'TRACKING_JITTER',
            'severity': 'medium',
            'msg': f'{jitter_ratio:.1%} of frames contain sudden large bone jumps. Tracking is unstable.'
        })

    # 3. Fragmentation Analysis
    loss_count = summary.get('tracking_loss_frame_count', 0)
    interp_count = summary.get('interpolated_frame_count', 0)
    total_frames = max(summary.get('frames_seen', 1), 1)
    fragmentation = (loss_count + interp_count) / total_frames
    
    if fragmentation > 0.2:
        warnings.append({
            'type': 'FRAGMENTED_TRACE',
            'severity': 'high',
            'msg': f'Fragmented bone trace: {fragmentation:.1%} of sequence is missing or interpolated. Joint data is unreliable.'
        })

    # 4. Final Reliability Score (0-100)
    # Base score 100, subtract penalties
    score = 100.0
    
    # Penalties
    score -= (jitter_ratio * 200)      # Max -20 penalty for 10% jitter
    score -= (fragmentation * 150)     # Max -30 penalty for 20% fragmentation
    
    # Only penalize asymmetry if variance is also high (tracking error)
    if bone_var > 0.02:
        score -= (max(arm_asym, leg_asym) * 100)
    else:
        # Perspective asymmetry is a minor penalty (-5)
        if arm_asym > 0.15 or leg_asym > 0.15:
            score -= 5

    # Coverage penalty
    coverage = summary.get('usable_pose_coverage', 0.0)
    if coverage < 0.9:
        score -= (1.0 - coverage) * 50
        
    final_score = max(0.0, min(100.0, score))
    
    return {
        'reliability_score': round(final_score, 2),
        'perspective_aware_warnings': warnings,
        'state': 'reliable' if final_score >= 70 else 'unreliable' if final_score < 40 else 'caution'
    }

if __name__ == '__main__':
    import sys
    import json
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
            print(json.dumps(calculate_reliability(data), indent=2))
