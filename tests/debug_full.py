#!/usr/bin/env python3
"""
Full debug of sim_loop to find the actual issue.
"""
import sys
import os
import json
import numpy as np
import mujoco

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.planning.piano_geometry import PianoGeometry
from src.control.state_machine import KeyStateMachine, ContactMode
from src.planning.kinematics import resolve_model_path

def debug_sim(json_file):
    with open(json_file, "r") as f:
        data_json = json.load(f)

    events_raw = data_json.get("events", data_json) if isinstance(data_json, dict) else data_json

    def get_onset(x):
        return x.get("onset_sec", x.get("onset", 0.0))

    events = sorted(events_raw, key=get_onset)

    model = mujoco.MjModel.from_xml_path(resolve_model_path())
    data = mujoco.MjData(model)
    piano_geo = PianoGeometry()

    # Get actuator indices
    gantry_act_names = ["a_tx", "a_ty", "a_tz", "a_rx", "a_ry", "a_rz"]
    gantry_act_ids = []
    for name in gantry_act_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        gantry_act_ids.append(aid)

    # Get joint indices
    gantry_joint_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    gantry_qpos_ids = []
    gantry_qvel_ids = []
    for name in gantry_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        gantry_qpos_ids.append(model.jnt_qposadr[jid])
        gantry_qvel_ids.append(model.jnt_dofadr[jid])

    # Initial pose
    mid_c = piano_geo.get_key_location(60)
    home_pose = np.array([mid_c.center_x, -0.22, 0.15, 0.0, 0.0, 0.0])

    print(f"=== DEBUG SIMULATION ===")
    print(f"Initial target: {home_pose}")
    print(f"Key 60 location: x={mid_c.center_x}, z={mid_c.center_z}")
    print()

    mujoco.mj_resetData(model, data)

    # Set initial position
    for i, q_idx in enumerate(gantry_qpos_ids):
        data.qpos[q_idx] = home_pose[i]

    # Set control targets
    for i, act_id in enumerate(gantry_act_ids):
        data.ctrl[act_id] = home_pose[i]

    mujoco.mj_forward(model, data)

    # Get fingertip site
    tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_2")

    print(f"Initial fingertip pos: {data.site_xpos[tip_site_id]}")
    print(f"Initial gantry qpos: {[data.qpos[i] for i in gantry_qpos_ids]}")
    print(f"Initial ctrl: {[data.ctrl[i] for i in gantry_act_ids]}")
    print()

    dt = model.opt.timestep
    sim_t = 0.0
    event_idx = 0
    active_machines = []

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    # Run for 2 seconds and print detailed info
    n_steps = int(2.0 / dt)

    for step in range(n_steps):
        # Spawn events
        while event_idx < len(events):
            e = events[event_idx]
            if get_onset(e) <= sim_t + 0.5:
                if e.get("staff") == 1:
                    contacts = e.get("contacts", [])
                    if contacts:
                        for c in contacts:
                            if int(c.get("staff", 1)) != 1:
                                continue
                            pitch = int(c["pitch"])
                            finger_idx = int(c["finger"])
                            strike = float(c["strike_time_sec"])
                            sm = KeyStateMachine(pitch, strike, finger_idx)
                            active_machines.append(sm)
                    else:
                        fingers = e.get("fingering", [])
                        pitches = e.get("pitches", [])
                        if fingers and pitches:
                            for p, f in zip(pitches, fingers):
                                active_machines.append(KeyStateMachine(p, get_onset(e), f))
                event_idx += 1
            else:
                break

        # Update state machines
        key_depths = {}
        for midi in range(21, 109):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"k{midi}")
            if jid >= 0:
                key_depths[midi] = data.qpos[model.jnt_qposadr[jid]]

        active_machines = [sm for sm in active_machines if not sm.done]
        for sm in active_machines:
            sm.update(sim_t, key_depths.get(sm.pitch, 0.0))

        # Find primary machine
        primary_sm = None
        for sm in active_machines:
            if sm.mode == ContactMode.STRIKE:
                primary_sm = sm
                break
        if primary_sm is None and active_machines:
            primary_sm = active_machines[0]

        # Compute control
        mujoco.mj_kinematics(model, data)

        q_dot_cmd = np.zeros(6)

        if primary_sm is not None:
            k_loc = piano_geo.get_key_location(primary_sm.pitch)
            key_surface_z = k_loc.center_z
            z_target, z_vmax, _ = primary_sm.get_target(key_surface_z)
            # Use actual key Y position
            target_pos = np.array([k_loc.center_x, k_loc.center_y, z_target])

            mujoco.mj_jacSite(model, data, jacp, jacr, tip_site_id)
            J_pos = jacp[:, gantry_qvel_ids]  # 3 x 6

            curr_pos = data.site_xpos[tip_site_id].copy()
            pos_err = target_pos - curr_pos

            Kp = 5.0  # Reduced gain
            ee_vel_des = pos_err * Kp

            # Clamp EE velocity
            EE_VEL_MAX = 0.5
            ee_vel_des = np.clip(ee_vel_des, -EE_VEL_MAX, EE_VEL_MAX)

            if z_vmax is not None:
                ee_vel_des[2] = np.clip(ee_vel_des[2], -z_vmax, z_vmax)

            try:
                q_dot_cmd, *_ = np.linalg.lstsq(J_pos, ee_vel_des, rcond=None)
            except:
                q_dot_cmd = np.zeros(6)

            # Clamp joint velocities
            JOINT_VEL_MAX = 1.0
            q_dot_cmd = np.clip(q_dot_cmd, -JOINT_VEL_MAX, JOINT_VEL_MAX)

        # Apply control - THIS IS THE KEY PART
        GANTRY_LIMITS = [
            (-0.5, 1.5),   # tx
            (-0.5, 0.5),   # ty
            (0.0, 0.6),    # tz
            (-0.5, 0.5),   # rx
            (-0.8, 0.8),   # ry
            (-0.8, 0.8),   # rz
        ]
        old_ctrl = np.array([data.ctrl[i] for i in gantry_act_ids])
        new_ctrl = old_ctrl + q_dot_cmd * dt

        # Clamp to joint limits
        for i, act_id in enumerate(gantry_act_ids):
            lo, hi = GANTRY_LIMITS[i]
            new_ctrl[i] = np.clip(new_ctrl[i], lo, hi)
            data.ctrl[act_id] = new_ctrl[i]

        # Step
        mujoco.mj_step(model, data)
        sim_t += dt

        # Print debug info every 100ms
        if step % int(0.1 / dt) == 0:
            gantry_pos = [data.qpos[i] for i in gantry_qpos_ids]
            gantry_vel = [data.qvel[i] for i in gantry_qvel_ids]
            ctrl = [data.ctrl[i] for i in gantry_act_ids]
            tip_pos = data.site_xpos[tip_site_id].copy()

            modes = [sm.mode.name for sm in active_machines]

            print(f"[t={sim_t:.2f}s]")
            print(f"  Modes: {modes}")
            print(f"  Gantry pos: [{gantry_pos[0]:.3f}, {gantry_pos[1]:.3f}, {gantry_pos[2]:.3f}]")
            print(f"  Gantry vel: [{gantry_vel[0]:.3f}, {gantry_vel[1]:.3f}, {gantry_vel[2]:.3f}]")
            print(f"  Ctrl:       [{ctrl[0]:.3f}, {ctrl[1]:.3f}, {ctrl[2]:.3f}]")
            print(f"  Tip pos:    [{tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f}]")
            if primary_sm:
                k_loc = piano_geo.get_key_location(primary_sm.pitch)
                z_target, _, _ = primary_sm.get_target(k_loc.center_z)
                print(f"  Target:     [{k_loc.center_x:.3f}, {k_loc.center_y:.3f}, {z_target:.3f}]")
            print(f"  q_dot_cmd:  [{q_dot_cmd[0]:.3f}, {q_dot_cmd[1]:.3f}, {q_dot_cmd[2]:.3f}]")
            print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/debug_full.py [json_path]")
    else:
        debug_sim(sys.argv[1])
