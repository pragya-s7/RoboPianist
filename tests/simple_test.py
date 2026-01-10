#!/usr/bin/env python3
"""
Simple test: just move gantry to a fixed position without any IK or QP.
This tests if the basic physics/control is stable.
"""
import sys
import os
import numpy as np
import mujoco
import mujoco.viewer
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.planning.kinematics import resolve_model_path

def run_simple():
    model = mujoco.MjModel.from_xml_path(resolve_model_path())
    data = mujoco.MjData(model)

    # Get gantry actuator IDs
    gantry_acts = ["a_tx", "a_ty", "a_tz", "a_rx", "a_ry", "a_rz"]
    gantry_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in gantry_acts]

    # Get finger actuator IDs
    finger_acts = []
    use_shadow = False
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and name.startswith("rh_A_"):
            use_shadow = True
            break
    if use_shadow:
        for prefix in ["rh_A_TH", "rh_A_FF", "rh_A_MF", "rh_A_RF", "rh_A_LF"]:
            for i in range(model.nu):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if name and name.startswith(prefix):
                    finger_acts.append(i)
    else:
        for prefix in ["a_th", "a_ff", "a_mf", "a_rf", "a_lf"]:
            for i in range(4):
                name = f"{prefix}_abd" if i == 0 else f"{prefix}_j{i+1}"
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid >= 0:
                    finger_acts.append(aid)

    # Initial position: over middle C
    target = np.array([0.636, -0.15, 0.10, 0.0, 0.0, 0.0])

    mujoco.mj_resetData(model, data)

    # Set initial gantry position and control
    gantry_joints = ["tx", "ty", "tz", "rx", "ry", "rz"]
    for i, jname in enumerate(gantry_joints):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        qpos_idx = model.jnt_qposadr[jid]
        data.qpos[qpos_idx] = target[i]

    for i, aid in enumerate(gantry_act_ids):
        data.ctrl[aid] = target[i]

    # Set fingers to neutral
    for aid in finger_acts:
        data.ctrl[aid] = 0.0

    mujoco.mj_forward(model, data)

    print("=== SIMPLE GANTRY TEST ===")
    print(f"Target position: {target[:3]}")
    print("Press Ctrl+C to exit\n")

    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        if "mjpython" in str(e):
            print("Run with: mjpython tests/simple_test.py")
            return
        raise

    dt = model.opt.timestep
    sim_t = 0.0

    # Simple control: slowly move to different positions
    positions = [
        np.array([0.5, -0.15, 0.10, 0.0, 0.0, 0.0]),
        np.array([0.7, -0.15, 0.10, 0.0, 0.0, 0.0]),
        np.array([0.7, -0.15, 0.05, 0.0, 0.0, 0.0]),
        np.array([0.5, -0.15, 0.05, 0.0, 0.0, 0.0]),
    ]
    pos_idx = 0
    target = positions[0]

    with viewer as v:
        while v.is_running():
            step_start = time.time()

            # Change target every 2 seconds
            if int(sim_t) % 2 == 0 and int(sim_t) > 0:
                new_idx = (int(sim_t) // 2) % len(positions)
                if new_idx != pos_idx:
                    pos_idx = new_idx
                    target = positions[pos_idx]
                    print(f"[t={sim_t:.1f}] New target: {target[:3]}")

            # Simple position control: directly set control targets
            for i, aid in enumerate(gantry_act_ids):
                data.ctrl[aid] = target[i]

            mujoco.mj_step(model, data)
            sim_t += dt

            v.sync()

            # Real-time sync
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    print("Done.")


if __name__ == "__main__":
    run_simple()
