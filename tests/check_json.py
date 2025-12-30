import json
import sys

def inspect_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    events = data['events']
    print(f"Loaded {len(events)} events from {filepath}")
    
    has_dynamics = False
    has_held_notes = False
    max_velocity = 0
    min_velocity = 127
    
    print("\n--- RANDOM SAMPLE INSPECTION ---")
    # Check middle of song where things get busy
    mid_idx = len(events) // 2
    for i in range(mid_idx, mid_idx + 10):
        e = events[i]
        print(f"Time: {e['onset_sec']:.2f}s | Vel: {e['notes'][0]['velocity']} | Held: {e['held_pitches']}")
        
        # Check Stats
        if len(e['held_pitches']) > 0: has_held_notes = True
        for n in e['notes']:
            v = n['velocity']
            if v != 64: has_dynamics = True
            max_velocity = max(max_velocity, v)
            min_velocity = min(min_velocity, v)

    print("\n--- DIAGNOSTICS ---")
    print(f"1. Dynamics Detected? {'YES' if has_dynamics else 'NO (Check inputs)'}")
    print(f"   Range: {min_velocity} - {max_velocity}")
    print(f"2. Polyphony Detected? {'YES' if has_held_notes else 'NO (Check logic)'}")
    print(f"3. Real-Time Conversion? {'YES' if events[-1]['onset_sec'] != events[-1]['onset'] else 'WARNING: Seconds == Quarters (120bpm default?)'}")

if __name__ == "__main__":
    inspect_json(sys.argv[1])