import json
import sys
import numpy as np

def inspect_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    events = data['events']
    print(f"Loaded {len(events)} events from {filepath}")
    
    has_dynamics = False
    has_wrist = False
    
    print("\n--- RANDOM SAMPLE INSPECTION ---")
    mid_idx = len(events) // 2
    for i in range(mid_idx, mid_idx + 5):
        e = events[i]
        wrist = e.get('wrist_target')
        wrist_str = f"[{wrist[0]:.1f}, {wrist[1]:.1f}]" if wrist else "None"
        print(f"T: {e['onset_sec']:.2f}s | Staff: {e['staff']} | Fingers: {e['fingering']} | Wrist: {wrist_str}")
        
        if wrist: has_wrist = True
        for n in e['notes']:
            if n['velocity'] != 64: has_dynamics = True

    print("\n--- DIAGNOSTICS ---")
    print(f"1. Dynamics?   {'YES' if has_dynamics else 'NO'}")
    print(f"2. Kinematics? {'YES' if has_wrist else 'NO (CRITICAL FAIL)'}")
    
    # Sanity Check
    if has_wrist:
        wrists = [e['wrist_target'] for e in events if e.get('wrist_target')]
        ws = np.array(wrists)
        print(f"   Wrist X Range: {ws[:,0].min():.1f} to {ws[:,0].max():.1f} mm")
        print(f"   Wrist Y Range: {ws[:,1].min():.1f} to {ws[:,1].max():.1f} mm")
        
        if ws[:,0].max() > 2000:
            print("   WARNING: Wrist X > 2 meters? Check piano geometry.")

if __name__ == "__main__":
    inspect_json(sys.argv[1])