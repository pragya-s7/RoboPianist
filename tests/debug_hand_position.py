#!/usr/bin/env python3
"""
Quick debug script to check hand initialization and first few steps.
"""
import sys
import os
import json
import numpy as np
import mujoco

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.planning.piano_geometry import PianoGeometry
from src.planning.kinematics import resolve_model_path

def debug_hand(json_file):
    print("="*60)
    print("CHORD Hand Position Debugger")
    print("="*60)

    # Load JSON
    with open(json_file) as f:
        data_json = json.load(f)
    events = data_json.get("events", [])
    print(f"\nLoaded {len(events)} events")

    if events:
        e0 = events[0]
        print(f"First event: t={e0.get('onset_sec')}s, pitches={e0.get('pitches')}")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(resolve_model_path())
    data = mujoco.MjData(model)

    # Piano geometry
    piano_geo = PianoGeometry()
    mid_c = piano_geo.get_key_location(60)

    print(f"\nPiano middle C (MIDI 60): x={mid_c.center_x:.3f}, y={mid_c.center_y:.3f}, z={mid_c.center_z:.3f}")

    # Set home position
    home_pose = np.array([mid_c.center_x, -0.22, 0.15, 0.0, 0.0, 0.0])
    print(f"\nHome pose (gantry): tx={home_pose[0]:.3f}, ty={home_pose[1]:.3f}, tz={home_pose[2]:.3f}")

    # Find gantry joints
    joint_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    qpos_ids = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_ids.append(model.jnt_qposadr[jid])

    # Reset and set home
    mujoco.mj_resetData(model, data)
    for i, qidx in enumerate(qpos_ids):
        data.qpos[qidx] = home_pose[i]

    mujoco.mj_forward(model, data)
    mujoco.mj_kinematics(model, data)

    # Check fingertip positions
    print("\n" + "="*60)
    print("Fingertip Positions After Home Initialization:")
    print("="*60)

    for f_idx in range(1, 6):
        site_name = f"tip_{f_idx}"
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid >= 0:
            pos = data.site_xpos[sid]
            print(f"Finger {f_idx} ({site_name:5s}): x={pos[0]:7.3f}, y={pos[1]:7.3f}, z={pos[2]:7.3f}")

    # Check palm
    palm_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_3")
    if palm_site >= 0:
        palm_pos = data.site_xpos[palm_site]
        print(f"\nHand center (tip_3): x={palm_pos[0]:.3f}, y={palm_pos[1]:.3f}, z={palm_pos[2]:.3f}")
        print(f"Distance above piano: {palm_pos[2] - mid_c.center_z:.3f}m ({(palm_pos[2] - mid_c.center_z)*100:.1f}cm)")

    # Check if this is reasonable
    if palm_pos[2] > 0.5:
        print("\n⚠️  WARNING: Hand is more than 50cm above piano! This will look wrong.")
    elif palm_pos[2] > 0.3:
        print("\n⚠️  WARNING: Hand is quite high (>30cm). Expected ~12-15cm.")
    else:
        print("\n✓ Hand height looks reasonable")

    # Simulate a few steps
    print("\n" + "="*60)
    print("Simulating 10 steps (10ms)...")
    print("="*60)

    for step in range(10):
        mujoco.mj_step(model, data)
        if step % 3 == 0:
            mujoco.mj_kinematics(model, data)
            pos = data.site_xpos[palm_site]
            vel = data.qvel[qpos_ids[:3]]  # tx, ty, tz velocities
            print(f"Step {step}: z={pos[2]:.4f}m, vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")

    final_pos = data.site_xpos[palm_site]
    displacement = final_pos - palm_pos
    print(f"\nDisplacement after 10 steps: dx={displacement[0]:.4f}, dy={displacement[1]:.4f}, dz={displacement[2]:.4f}")

    if np.linalg.norm(displacement) > 0.1:
        print("⚠️  WARNING: Hand moved >10cm without control! Physics unstable or gains too high.")
    else:
        print("✓ Hand stayed roughly in place")

    print("\n" + "="*60)
    print("Debug complete. Check values above for issues.")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/debug_hand_position.py <json_file>")
        sys.exit(1)
    debug_hand(sys.argv[1])
