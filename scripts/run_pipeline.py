import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to sys.path so we can import src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ingest.parse_musicxml import extract_note_table, notes_to_events, m21
from src.fingering.dp_solver import DPSolver

def run(infile, outfile):
    print(f"--- 1. Ingesting: {infile} ---")
    score = m21.converter.parse(infile)
    note_table = extract_note_table(score)
    events = notes_to_events(note_table)
    print(f"Found {len(events)} events.")

    # Separation: Solve RH (staff 1) and LH (staff 2) separately
    rh_events = [e for e in events if e.staff == 1]
    lh_events = [e for e in events if e.staff == 2]

    solver = DPSolver()

    print(f"--- 2. Solving Fingerings ---")
    print(f"Right Hand: {len(rh_events)} events")
    rh_fingerings = solver.solve([vars(e) for e in rh_events])
    
    print(f"Left Hand:  {len(lh_events)} events")
    lh_fingerings = solver.solve([vars(e) for e in lh_events])

    # Inject fingerings back into the main event list
    # We use iterators to pop fingerings off as we encounter the events in the sorted list
    rh_iter = iter(rh_fingerings)
    lh_iter = iter(lh_fingerings)

    final_events = []
    for e in events:
        e_dict = vars(e)
        if e.staff == 1:
            fingers = next(rh_iter)
        else:
            fingers = next(lh_iter)
        
        e_dict['fingering'] = fingers
        final_events.append(e_dict)

    # Output
    payload = {
        "source": str(infile),
        "total_events": len(final_events),
        "events": final_events
    }

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    
    print(f"--- 3. Success ---")
    print(f"Saved annotated data to: {outfile}")
    print(f"Sample Event: Time {final_events[10]['onset_sec']}s -> Fingers {final_events[10]['fingering']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    args = parser.parse_args()
    
    run(args.infile, args.outfile)