import argparse
import json
import os
import time
from typing import Dict, Tuple

import numpy as np
import mujoco
import mujoco.viewer

# Allow "python -m scripts.play_plan_ghost" to import src/
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.planning.piano_geometry import PianoGeometry, KeyLocation


def build_ghost_maps(model: mujoco.MjModel) -> Dict[Tuple[str, int], int]:
    """
    Map (hand, finger_idx) -> mocap body index.

    hand: 'R' or 'L'
    finger_idx: 1..5
    """
    ghost_map: Dict[Tuple[str, int], int] = {}

    for hand_prefix, hand_code in (("ghost_r", "R"), ("ghost_l", "L")):
        for f in range(1, 6):
            body_name = f"{hand_prefix}{f}"
            try:
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
            except Exception:
                body_id = -1

            if body_id < 0:
                print(f"[WARN] Ghost body '{body_name}' not found in model.")
                continue

            mocap_id = model.body_mocapid[body_id]
            if mocap_id < 0:
                print(f"[WARN] Ghost body '{body_name}' has no mocap ID.")
                continue

            ghost_map[(hand_code, f)] = mocap_id

    print(f"Found {len(ghost_map)} ghost mocap bodies.")
    return ghost_map


def play_plan(xml_path: str, plan_path: str, realtime: bool = True) -> None:
    print(f"Loading MuJoCo model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print(f"Loading plan JSON from: {plan_path}")
    with open(plan_path, "r") as f:
        payload = json.load(f)

    events = payload["events"]
    events = sorted(events, key=lambda e: e["onset_sec"])
    print(f"Loaded {len(events)} events from plan.")

    piano = PianoGeometry()
    ghost_map = build_ghost_maps(model)

    # Neutral "rest" positions for each hand (off the keys but not in deep space)
    rest_pos_R = np.array([0.63, -0.25, 0.15])  # right hand hover
    rest_pos_L = np.array([0.20, -0.25, 0.15])  # left hand hover

    # How long after onset a note is "visually active"
    active_window = 0.35  # seconds

    # Active assignments: (hand, finger) -> {loc: KeyLocation, expiry: float}
    active: Dict[Tuple[str, int], Dict] = {}

    # Precompute which events are left/right hand
    # staff == 1 → Right hand; staff == 2 → Left hand
    def hand_from_staff(staff: int) -> str:
        return "R" if staff == 1 else "L"

    print("--- Starting ghost-plan visualization ---")
    dt = model.opt.timestep
    n_events = len(events)
    next_event_idx = 0
    sim_t = 0.0

    start_wall = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Time update
            if realtime:
                sim_t = time.time() - start_wall
            else:
                sim_t += dt

            # 1. Add any events whose onset_sec <= current time
            while next_event_idx < n_events and events[next_event_idx]["onset_sec"] <= sim_t:
                e = events[next_event_idx]
                staff = e.get("staff", 1)
                hand = hand_from_staff(staff)

                pitches = e.get("pitches", [])
                fingering = e.get("fingering", [])

                if len(pitches) != len(fingering):
                    # Fallback: ignore mismatched event
                    next_event_idx += 1
                    continue

                for pitch, finger in zip(pitches, fingering):
                    if finger is None:
                        continue
                    key = (hand, int(finger))
                    if key not in ghost_map:
                        continue

                    try:
                        k_loc: KeyLocation = piano.get_key_location(int(pitch))
                    except Exception:
                        # Note outside 88-key range: skip
                        continue

                    active[key] = {
                        "loc": k_loc,
                        "expiry": e["onset_sec"] + active_window,
                    }

                next_event_idx += 1

            # 2. Drop expired assignments
            to_delete = [k for k, v in active.items() if sim_t > v["expiry"]]
            for k in to_delete:
                del active[k]

            # 3. Update ghost mocap positions
            #    For each ghost body:
            #      - If it has an active assignment, float above its key
            #      - Else, park at a hand-specific rest pose
            for (hand, finger), mocap_id in ghost_map.items():
                if (hand, finger) in active:
                    k_loc = active[(hand, finger)]["loc"]
                    # Hover slightly above the key surface
                    z_hover = k_loc.center_z + 0.06
                    data.mocap_pos[mocap_id] = np.array(
                        [k_loc.center_x, k_loc.center_y, z_hover]
                    )
                else:
                    if hand == "R":
                        data.mocap_pos[mocap_id] = rest_pos_R
                    else:
                        data.mocap_pos[mocap_id] = rest_pos_L

            # 4. Advance sim (we only care about time + viewer, physics is trivial here)
            mujoco.mj_step(model, data)

            viewer.sync()

            # Maintain real-time if requested
            if realtime:
                time_until_next = step_start + dt - time.time()
                if time_until_next > 0:
                    time.sleep(time_until_next)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a CHORD/plan JSON as ghost fingertips over the piano."
    )
    parser.add_argument(
        "--xml",
        type=str,
        required=True,
        help="Path to MuJoCo XML with piano + ghost bodies (e.g. assets/piano.xml)",
    )
    parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="Path to plan JSON (output of run_pipeline.py)",
    )
    parser.add_argument(
        "--no_realtime",
        action="store_true",
        help="Run as fast as possible instead of wall-clock real time.",
    )
    args = parser.parse_args()

    play_plan(args.xml, args.plan, realtime=not args.no_realtime)


if __name__ == "__main__":
    main()
