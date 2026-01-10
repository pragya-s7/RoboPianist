#!/usr/bin/env python3
"""
Diagnostic script to identify the cause of hand explosion in MuJoCo.
Runs headless and prints detailed state information.
"""
import sys
import os
import numpy as np
import mujoco

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.planning.kinematics import resolve_model_path

def diagnose():
    try:
        xml_path = resolve_model_path()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    print("=== MUJOCO SIMULATION DIAGNOSTIC ===\n")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print(f"Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    print(f"Timestep: {model.opt.timestep}")
    print(f"Integrator: {model.opt.integrator}")
    print()

    # Print all joints with their limits
    print("=== JOINT CONFIGURATION ===")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qpos_adr = model.jnt_qposadr[i]
        qvel_adr = model.jnt_dofadr[i]
        limited = model.jnt_limited[i]
        if limited:
            lo, hi = model.jnt_range[i]
            print(f"  {name:12s}: qpos[{qpos_adr}], qvel[{qvel_adr}], range=[{lo:.3f}, {hi:.3f}]")
        else:
            print(f"  {name:12s}: qpos[{qpos_adr}], qvel[{qvel_adr}], unlimited")
    print()

    # Print all actuators
    print("=== ACTUATOR CONFIGURATION ===")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        ctrl_limited = model.actuator_ctrllimited[i]
        if ctrl_limited:
            lo, hi = model.actuator_ctrlrange[i]
            print(f"  {name:12s}: ctrl range=[{lo:.1f}, {hi:.1f}]")
        else:
            print(f"  {name:12s}: unlimited ctrl")
    print()

    # Reset and set a simple initial pose
    mujoco.mj_resetData(model, data)

    # Find gantry joints
    gantry_joints = ["tx", "ty", "tz", "rx", "ry", "rz"]
    gantry_qpos = {}
    for name in gantry_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            gantry_qpos[name] = model.jnt_qposadr[jid]

    # Set initial position: middle C, behind keyboard, slightly above
    initial_pose = {"tx": 0.636, "ty": -0.22, "tz": 0.15, "rx": 0.0, "ry": 0.0, "rz": 0.0}
    for name, val in initial_pose.items():
        if name in gantry_qpos:
            data.qpos[gantry_qpos[name]] = val

    # Also set actuator targets to match
    gantry_acts = ["a_tx", "a_ty", "a_tz", "a_rx", "a_ry", "a_rz"]
    for i, aname in enumerate(gantry_acts):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        if aid >= 0:
            jname = gantry_joints[i]
            data.ctrl[aid] = initial_pose.get(jname, 0.0)

    mujoco.mj_forward(model, data)

    print("=== INITIAL STATE ===")
    print(f"qpos[:10] = {data.qpos[:10]}")
    print(f"qvel[:10] = {data.qvel[:10]}")
    print(f"ctrl[:10] = {data.ctrl[:10]}")
    print()

    # Run simulation for a few seconds and monitor
    print("=== RUNNING SIMULATION (5 seconds) ===")
    dt = model.opt.timestep
    n_steps = int(5.0 / dt)

    max_qvel = 0.0
    max_qpos_drift = 0.0
    initial_qpos = data.qpos.copy()
    nan_detected = False
    explosion_step = -1

    # Monitor finger joints specifically
    finger_joints = []
    finger_prefixes = ("th_", "ff_", "mf_", "rf_", "lf_", "rh_TH", "rh_FF", "rh_MF", "rh_RF", "rh_LF")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name.startswith(finger_prefixes):
            finger_joints.append((name, model.jnt_qposadr[i], model.jnt_dofadr[i]))

    for step in range(n_steps):
        mujoco.mj_step(model, data)

        # Check for NaN
        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            nan_detected = True
            explosion_step = step
            print(f"[STEP {step}] NaN DETECTED!")
            print(f"  qpos has NaN: {np.any(np.isnan(data.qpos))}")
            print(f"  qvel has NaN: {np.any(np.isnan(data.qvel))}")
            break

        # Check for explosion (very high velocities)
        curr_max_qvel = np.max(np.abs(data.qvel))
        if curr_max_qvel > max_qvel:
            max_qvel = curr_max_qvel

        if curr_max_qvel > 100:  # rad/s - very high
            explosion_step = step
            print(f"[STEP {step}] VELOCITY EXPLOSION: max |qvel| = {curr_max_qvel:.2f}")

            # Find which joints have high velocities
            for fj_name, qpos_idx, qvel_idx in finger_joints:
                if np.abs(data.qvel[qvel_idx]) > 10:
                    print(f"  {fj_name}: qvel = {data.qvel[qvel_idx]:.2f}")
            break

        # Print status every 1 second
        if step % int(1.0 / dt) == 0:
            t = step * dt
            print(f"[t={t:.1f}s] max|qvel|={curr_max_qvel:.3f}, max|qpos drift|={np.max(np.abs(data.qpos - initial_qpos)):.4f}")

    print()
    print("=== FINAL STATE ===")
    print(f"qpos[:10] = {data.qpos[:10]}")
    print(f"qvel[:10] = {data.qvel[:10]}")
    print(f"Max |qvel| observed: {max_qvel:.3f}")
    print()

    if nan_detected:
        print("RESULT: SIMULATION EXPLODED (NaN)")
    elif explosion_step > 0:
        print(f"RESULT: SIMULATION EXPLODED at step {explosion_step}")
    else:
        print("RESULT: SIMULATION STABLE")

    # Check finger joint states
    print()
    print("=== FINGER JOINT FINAL STATES ===")
    for fj_name, qpos_idx, qvel_idx in finger_joints[:8]:  # first 8
        q = data.qpos[qpos_idx]
        qd = data.qvel[qvel_idx]

        # Get joint limits
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fj_name)
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            in_range = lo <= q <= hi
            range_str = f"[{lo:.2f},{hi:.2f}]"
            status = "OK" if in_range else "OUT OF RANGE!"
        else:
            range_str = "unlimited"
            status = "OK"

        print(f"  {fj_name:10s}: q={q:7.3f}, qd={qd:7.3f}, range={range_str} {status}")


if __name__ == "__main__":
    diagnose()
