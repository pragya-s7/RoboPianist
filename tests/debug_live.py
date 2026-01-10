#!/usr/bin/env python3
"""
Debug the actual sim_loop behavior with detailed logging.
"""
import sys
import os
import json
import numpy as np
import mujoco
import mujoco.viewer
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.planning.piano_geometry import PianoGeometry
from src.control.state_machine import KeyStateMachine, ContactMode
from src.control.servo import ChordServo, RobotState, ConstraintBlock
from src.control.constraint_factory import ConstraintFactory
from src.planning.kinematics import resolve_model_path

def run_debug(json_file):
    with open(json_file, "r") as f:
        data_json = json.load(f)

    events_raw = data_json.get("events", data_json) if isinstance(data_json, dict) else data_json

    def get_onset(x):
        return x.get("onset_sec", x.get("onset", 0.0))

    events = sorted(events_raw, key=get_onset)

    model = mujoco.MjModel.from_xml_path(resolve_model_path())
    data = mujoco.MjData(model)
    piano_geo = PianoGeometry()

    # Get gantry actuator/joint indices
    gantry_act_names = ["a_tx", "a_ty", "a_tz", "a_rx", "a_ry", "a_rz"]
    gantry_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in gantry_act_names]

    gantry_joint_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    gantry_qpos_ids = []
    gantry_qvel_ids = []
    for name in gantry_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        gantry_qpos_ids.append(model.jnt_qposadr[jid])
        gantry_qvel_ids.append(model.jnt_dofadr[jid])

    n_gantry_dof = 6

    # Get finger actuators
    finger_act_map = {1: [], 2: [], 3: [], 4: [], 5: []}
    use_shadow = False
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and name.startswith("rh_A_"):
            use_shadow = True
            break
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not name:
            continue
        if use_shadow:
            if name.startswith("rh_A_TH"):
                finger_act_map[1].append(i)
            elif name.startswith("rh_A_FF"):
                finger_act_map[2].append(i)
            elif name.startswith("rh_A_MF"):
                finger_act_map[3].append(i)
            elif name.startswith("rh_A_RF"):
                finger_act_map[4].append(i)
            elif name.startswith("rh_A_LF"):
                finger_act_map[5].append(i)
        else:
            if name.startswith("a_th"):
                finger_act_map[1].append(i)
            elif name.startswith("a_ff"):
                finger_act_map[2].append(i)
            elif name.startswith("a_mf"):
                finger_act_map[3].append(i)
            elif name.startswith("a_rf"):
                finger_act_map[4].append(i)
            elif name.startswith("a_lf"):
                finger_act_map[5].append(i)

    # Fingertip site
    tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_2")

    # Initial pose
    mid_c = piano_geo.get_key_location(60)
    home_pose = np.array([mid_c.center_x, -0.15, 0.12, 0.0, 0.0, 0.0])

    mujoco.mj_resetData(model, data)

    for i, q_idx in enumerate(gantry_qpos_ids):
        data.qpos[q_idx] = home_pose[i]
    for i, act_id in enumerate(gantry_act_ids):
        data.ctrl[act_id] = home_pose[i]

    mujoco.mj_forward(model, data)

    print("=== DEBUG LIVE SIMULATION ===")
    print(f"Initial gantry: {home_pose[:3]}")
    print(f"Initial tip: {data.site_xpos[tip_site_id]}")
    print()

    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        if "mjpython" in str(e):
            print("Run with: mjpython tests/debug_live.py " + json_file)
            return
        raise

    dt = model.opt.timestep
    sim_t = 0.0
    event_idx = 0
    active_machines = []

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    GANTRY_LIMITS = [
        (-0.5, 1.5), (-0.5, 0.5), (0.0, 0.6),
        (-0.5, 0.5), (-0.8, 0.8), (-0.8, 0.8),
    ]

    with viewer as v:
        while v.is_running():
            step_start = time.time()

            # Read key depths
            key_depths = {}
            for midi in range(21, 109):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"k{midi}")
                if jid >= 0:
                    key_depths[midi] = data.qpos[model.jnt_qposadr[jid]]

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

            # Finger targets
            finger_targets = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
            for sm in active_machines:
                val = 0.0
                if sm.mode == ContactMode.STRIKE:
                    val = 1.0
                elif sm.mode == ContactMode.HOLD:
                    val = 0.6
                elif sm.mode == ContactMode.PRETOUCH:
                    val = 0.1
                finger_targets[sm.finger_idx] = max(finger_targets[sm.finger_idx], val)

            # Kinematics
            mujoco.mj_kinematics(model, data)

            # Current state
            curr_tip = data.site_xpos[tip_site_id].copy()
            curr_gantry = np.array([data.qpos[i] for i in gantry_qpos_ids])

            # Simple proportional control on gantry position
            # Target: move gantry so tip reaches key
            if primary_sm is not None:
                k_loc = piano_geo.get_key_location(primary_sm.pitch)
                key_surface_z = k_loc.center_z
                z_target, z_vmax, _ = primary_sm.get_target(key_surface_z)

                # Desired tip position
                tip_target = np.array([k_loc.center_x, k_loc.center_y, z_target])

                # Error in tip space
                tip_error = tip_target - curr_tip

                # Simple: move gantry by tip error (assuming roughly 1:1 mapping)
                # This is a huge simplification but should be stable
                Kp = 2.0  # Very low gain
                gantry_vel = np.zeros(6)
                gantry_vel[0] = tip_error[0] * Kp  # tx
                gantry_vel[1] = tip_error[1] * Kp  # ty
                gantry_vel[2] = tip_error[2] * Kp  # tz

                # Clamp velocities
                VEL_MAX = 0.3
                gantry_vel = np.clip(gantry_vel, -VEL_MAX, VEL_MAX)

                # Update control targets
                new_ctrl = data.ctrl[gantry_act_ids].copy() + gantry_vel * dt
                for i in range(6):
                    lo, hi = GANTRY_LIMITS[i]
                    new_ctrl[i] = np.clip(new_ctrl[i], lo, hi)
                    data.ctrl[gantry_act_ids[i]] = new_ctrl[i]

            # Apply finger control
            FINGER_CURL_MAX = {
                1: (0.0, 0.4, 0.7, 0.9),
                2: (0.0, 1.4, 1.4, 1.4),
                3: (0.0, 1.4, 1.4, 1.4),
                4: (0.0, 1.4, 1.4, 1.4),
                5: (0.0, 1.4, 1.4, 1.4),
            }
            for f_idx, curl in finger_targets.items():
                acts = finger_act_map.get(f_idx, [])
                curl_max = FINGER_CURL_MAX.get(f_idx, (0.0, 1.4, 1.4, 1.4))
                if len(acts) == 4:
                    data.ctrl[acts[0]] = curl * curl_max[0]
                    data.ctrl[acts[1]] = curl * curl_max[1]
                    data.ctrl[acts[2]] = curl * curl_max[2]
                    data.ctrl[acts[3]] = curl * curl_max[3]

            # Step
            mujoco.mj_step(model, data)
            sim_t += dt

            # Print debug every 0.5s
            if int(sim_t * 2) != int((sim_t - dt) * 2):
                modes = [sm.mode.name for sm in active_machines]
                gantry_pos = [data.qpos[i] for i in gantry_qpos_ids]
                ctrl = [data.ctrl[i] for i in gantry_act_ids]
                tip = data.site_xpos[tip_site_id]
                print(f"[t={sim_t:.1f}s] modes={modes}")
                print(f"  Gantry:  [{gantry_pos[0]:.3f}, {gantry_pos[1]:.3f}, {gantry_pos[2]:.3f}]")
                print(f"  Ctrl:    [{ctrl[0]:.3f}, {ctrl[1]:.3f}, {ctrl[2]:.3f}]")
                print(f"  Tip:     [{tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}]")
                if primary_sm:
                    k = piano_geo.get_key_location(primary_sm.pitch)
                    z_t, _, _ = primary_sm.get_target(k.center_z)
                    print(f"  Target:  [{k.center_x:.3f}, {k.center_y:.3f}, {z_t:.3f}]")
                print()

            v.sync()

            # Real-time
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mjpython tests/debug_live.py [json_path]")
    else:
        run_debug(sys.argv[1])
