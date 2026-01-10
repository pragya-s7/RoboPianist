#!/usr/bin/env python3
"""
Diagnose control loop issues.
This simulates what sim_loop does without the viewer.
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

FINGER_SUFFIX_ORDER = ["abd", "j2", "j3", "j4"]

def get_actuator_indices(model):
    target_gantry = ["a_tx", "a_ty", "a_tz", "a_rx", "a_ry", "a_rz"]
    gantry_ids = {}
    finger_map = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}

    use_shadow = False
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not name:
            continue
        if name.startswith("rh_A_"):
            use_shadow = True
            break

    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not name:
            continue
        if name in target_gantry:
            gantry_ids[name] = i
        elif use_shadow:
            if name.startswith("rh_A_TH") and name.endswith(("THJ4", "THJ3", "THJ2", "THJ1")):
                finger_map[1][name] = i
            elif name.startswith("rh_A_FF") and name.endswith(("FFJ3", "FFJ0")):
                finger_map[2][name] = i
            elif name.startswith("rh_A_MF") and name.endswith(("MFJ3", "MFJ0")):
                finger_map[3][name] = i
            elif name.startswith("rh_A_RF") and name.endswith(("RFJ3", "RFJ0")):
                finger_map[4][name] = i
            elif name.startswith("rh_A_LF") and name.endswith(("LFJ3", "LFJ0")):
                finger_map[5][name] = i
        else:
            if name.startswith("a_th_"):
                finger_map[1][name.split("a_th_", 1)[1]] = i
            elif name.startswith("a_ff_"):
                finger_map[2][name.split("a_ff_", 1)[1]] = i
            elif name.startswith("a_mf_"):
                finger_map[3][name.split("a_mf_", 1)[1]] = i
            elif name.startswith("a_rf_"):
                finger_map[4][name.split("a_rf_", 1)[1]] = i
            elif name.startswith("a_lf_"):
                finger_map[5][name.split("a_lf_", 1)[1]] = i

    ordered_finger_map = {}
    for f_idx, suffix_map in finger_map.items():
        if use_shadow:
            ordered = list(suffix_map.values())
            ordered_finger_map[f_idx] = ordered
        else:
            ordered = [suffix_map.get(s) for s in FINGER_SUFFIX_ORDER]
            if all(act is not None for act in ordered):
                ordered_finger_map[f_idx] = ordered
            else:
                ordered_finger_map[f_idx] = []

    ordered_gantry_ids = [gantry_ids[name] for name in target_gantry if name in gantry_ids]
    return ordered_gantry_ids, ordered_finger_map


def build_gantry_joint_maps(model):
    joint_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    qpos_indices = []
    qvel_indices = []

    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Gantry joint '{name}' not found in model.")
        qpos_indices.append(int(model.jnt_qposadr[jid]))
        qvel_indices.append(int(model.jnt_dofadr[jid]))

    return joint_names, qpos_indices, qvel_indices


def diagnose_control(json_file):
    with open(json_file, "r") as f:
        data_json = json.load(f)

    events_raw = data_json.get("events", data_json) if isinstance(data_json, dict) else data_json
    if not events_raw:
        raise RuntimeError(f"No 'events' field found in {json_file}")

    # Handle both "onset_sec" and "onset" formats
    def get_onset(x):
        return x.get("onset_sec", x.get("onset", 0.0))

    events = sorted(events_raw, key=get_onset)

    xml_path = resolve_model_path()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    piano_geo = PianoGeometry()

    gantry_act_ids, finger_act_map = get_actuator_indices(model)
    act_ctrl_ranges = model.actuator_ctrlrange.copy()
    act_ctrl_limited = model.actuator_ctrllimited.copy()
    act_trn_type = model.actuator_trntype.copy()
    act_trn_id = model.actuator_trnid.copy()
    jnt_limited = model.jnt_limited.copy()
    jnt_range = model.jnt_range.copy()
    actuator_meta = {}
    finger_gain_enabled = 10.0
    finger_bias_enabled = -10.0
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if not name:
            continue
        if name.startswith("rh_A_TH"):
            actuator_meta[aid] = (1, "flex")
        elif name.startswith("rh_A_FF"):
            actuator_meta[aid] = (2, "flex")
        elif name.startswith("rh_A_MF"):
            actuator_meta[aid] = (3, "flex")
        elif name.startswith("rh_A_RF"):
            actuator_meta[aid] = (4, "flex")
        elif name.startswith("rh_A_LF"):
            actuator_meta[aid] = (5, "flex")
        elif name.startswith("a_th_"):
            actuator_meta[aid] = (1, name.split("a_th_", 1)[1])
        elif name.startswith("a_ff_"):
            actuator_meta[aid] = (2, name.split("a_ff_", 1)[1])
        elif name.startswith("a_mf_"):
            actuator_meta[aid] = (3, name.split("a_mf_", 1)[1])
        elif name.startswith("a_rf_"):
            actuator_meta[aid] = (4, name.split("a_rf_", 1)[1])
        elif name.startswith("a_lf_"):
            actuator_meta[aid] = (5, name.split("a_lf_", 1)[1])
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        if name.startswith(("th_", "ff_", "mf_", "rf_", "lf_")) or name.startswith("rh_"):
            dof = model.jnt_dofadr[jid]
            model.dof_damping[dof] = max(model.dof_damping[dof], 12.0)
            model.dof_frictionloss[dof] = max(model.dof_frictionloss[dof], 1.0)
    joint_names, gantry_qpos_ids, gantry_qvel_ids = build_gantry_joint_maps(model)
    n_gantry_dof = len(gantry_qpos_ids)


    # Fingertip sites
    finger_site_ids = {}
    for f_idx in range(1, 6):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"tip_{f_idx}")
        finger_site_ids[f_idx] = sid

    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_yaw")
    if wrist_id < 0:
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
    if wrist_id < 0:
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    if wrist_id < 0:
        raise RuntimeError("Body 'wrist_yaw' or palm body not found. Check scene.xml.")

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    # Initial pose
    mid_c = piano_geo.get_key_location(60)
    home_pose = np.array([mid_c.center_x, -0.22, 0.15, 0.0, 0.0, 0.0])

    mujoco.mj_resetData(model, data)

    for i, q_idx in enumerate(gantry_qpos_ids):
        data.qpos[q_idx] = home_pose[i]

    if len(gantry_act_ids) == 6:
        data.ctrl[gantry_act_ids] = home_pose

    mujoco.mj_forward(model, data)

    finger_home_targets = {}
    actuator_home_ctrl = data.ctrl.copy()
    for acts in finger_act_map.values():
        for act_id in acts:
            if act_trn_type[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = act_trn_id[act_id, 0]
                qpos_addr = model.jnt_qposadr[jid]
                finger_home_targets[act_id] = float(data.qpos[qpos_addr])

    for act_id, home in finger_home_targets.items():
        if act_ctrl_limited[act_id]:
            lo, hi = act_ctrl_ranges[act_id]
            home = float(np.clip(home, lo, hi))
        data.ctrl[act_id] = home
        actuator_home_ctrl[act_id] = home

    print(f"=== CONTROL LOOP DIAGNOSTIC ===")
    print(f"JSON: {json_file}")
    print(f"Events: {len(events)}")
    print()

    dt = model.opt.timestep
    sim_t = 0.0
    event_idx = 0
    active_machines = []

    max_qvel = 0.0
    max_ctrl_change = 0.0
    max_curl = 0.0
    explosion_t = -1

    # Target smoothing (same as sim_loop.py)
    smooth_wrist_target = np.array([mid_c.center_x, -0.15, 0.12])
    TARGET_SMOOTHING_TAU_XY = 0.35
    TARGET_SMOOTHING_TAU_Z = 0.18
    smooth_finger_targets = {f: 0.0 for f in range(1, 6)}
    FINGER_SMOOTHING_TAU = 0.08
    # Allow full joint range so STRIKE can actually depress keys.
    MAX_FINGER_CURL = 1.25
    FINGER_FLEX_RANGE = 1.0

    GANTRY_KP = np.array([1.6, 1.6, 1.4, 0.8, 0.8, 0.8])
    GANTRY_KD = np.array([1.4, 1.4, 1.2, 0.6, 0.6, 0.6])
    GANTRY_VEL_MAX = np.array([0.25, 0.2, 0.18, 0.2, 0.2, 0.2])
    # Finger actuator rate limiting removed to keep joints stiff

    # Run for 30 seconds
    n_steps = int(30.0 / dt)

    for step in range(n_steps):
        # Read key depths
        key_depths = {}
        for midi in range(21, 109):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"k{midi}")
            if jid >= 0:
                addr = model.jnt_qposadr[jid]
                key_depths[midi] = data.qpos[addr]

        # Spawn events
        while event_idx < len(events):
            e = events[event_idx]
            if get_onset(e) <= sim_t + 0.5:
                if e["staff"] == 1:
                    contacts = e.get("contacts", [])
                    if contacts:
                        for c in contacts:
                            if int(c.get("staff", 1)) != 1:
                                continue
                            pitch = int(c["pitch"])
                            finger_idx = int(c["finger"])
                            strike = float(c["strike_time_sec"])
                            pretouch = float(c["pretouch_start_sec"])
                            hold_end = float(c["hold_end_sec"])
                            release_end = float(c["release_end_sec"])

                            sm = KeyStateMachine(pitch, strike, finger_idx)
                            sm.approach_lead = max(0.01, strike - pretouch)
                            sm.hold_duration = max(0.01, hold_end - strike)
                            sm.release_duration = max(0.01, release_end - hold_end)
                            active_machines.append(sm)
                    else:
                        # Fallback: legacy behavior using fingering + onset
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

        # Find primary machine
        primary_sm = None
        if active_machines:
            for sm in active_machines:
                if sm.mode == ContactMode.STRIKE:
                    primary_sm = sm
                    break
            if primary_sm is None:
                primary_sm = active_machines[0]

        q_dot_cmd = np.zeros(n_gantry_dof)

        # Find wrist target from events (same logic as sim_loop.py)
        wrist_target = None
        for e in reversed(events[:event_idx]):
            if e.get("staff") == 1 and "wrist_target" in e:
                event_time = e.get("onset_sec", e.get("onset", 0))
                # Only use events that have actually started
                if event_time <= sim_t and sim_t - event_time < 2.0:
                    wrist_target = np.array(e["wrist_target"], dtype=float)
                    break

        # Default wrist target if none found
        if wrist_target is None:
            wrist_target = np.array([piano_geo.get_key_location(60).center_x, -0.15, 0.12])

        if active_machines:
            mujoco.mj_kinematics(model, data)
            wrist_pos = data.xpos[wrist_id].copy()
            wrist_rot = data.xmat[wrist_id].reshape(3, 3)
            desired_wrist = []
            for sm in active_machines:
                if sm.mode not in (ContactMode.PRETOUCH, ContactMode.STRIKE, ContactMode.HOLD):
                    continue
                sid = finger_site_ids.get(sm.finger_idx, -1)
                if sid < 0:
                    continue
                k_loc = piano_geo.get_key_location(sm.pitch)
                key_surface_z = k_loc.center_z
                z_target, _z_vmax, _stiff = sm.get_target(key_surface_z)
                desired_tip = np.array([k_loc.center_x, k_loc.center_y, z_target])
                tip_pos = data.site_xpos[sid].copy()
                tip_offset_local = wrist_rot.T @ (tip_pos - wrist_pos)
                tip_offset_world = wrist_rot @ tip_offset_local
                desired_wrist.append(np.array([
                    desired_tip[0] - tip_offset_world[0],
                    desired_tip[1] - tip_offset_world[1],
                    desired_tip[2] - tip_offset_world[2],
                ]))
            if desired_wrist:
                wrist_target[:3] = np.mean(desired_wrist, axis=0)

        # Apply target smoothing
        alpha_xy = dt / (TARGET_SMOOTHING_TAU_XY + dt)
        alpha_z = dt / (TARGET_SMOOTHING_TAU_Z + dt)
        smooth_wrist_target[0] = alpha_xy * wrist_target[0] + (1 - alpha_xy) * smooth_wrist_target[0]
        smooth_wrist_target[1] = alpha_xy * wrist_target[1] + (1 - alpha_xy) * smooth_wrist_target[1]
        smooth_wrist_target[2] = alpha_z * wrist_target[2] + (1 - alpha_z) * smooth_wrist_target[2]
        smooth_wrist_target[0] = np.clip(smooth_wrist_target[0], -0.5, 1.5)
        smooth_wrist_target[1] = np.clip(smooth_wrist_target[1], -0.5, 0.5)
        smooth_wrist_target[2] = np.clip(smooth_wrist_target[2], 0.0, 0.6)

        curr_gantry = np.array([data.qpos[idx] for idx in gantry_qpos_ids])
        curr_gantry_vel = np.array([data.qvel[idx] for idx in gantry_qvel_ids])
        gantry_error = np.zeros(n_gantry_dof)
        gantry_error[0] = smooth_wrist_target[0] - curr_gantry[0]
        gantry_error[1] = smooth_wrist_target[1] - curr_gantry[1]
        gantry_error[2] = smooth_wrist_target[2] - curr_gantry[2]
        gantry_error[3] = -curr_gantry[3]
        gantry_error[4] = -curr_gantry[4]
        gantry_error[5] = -curr_gantry[5]

        q_dot_cmd = GANTRY_KP * gantry_error - GANTRY_KD * curr_gantry_vel
        q_dot_cmd = np.clip(q_dot_cmd, -GANTRY_VEL_MAX, GANTRY_VEL_MAX)

        # Apply gantry control
        GANTRY_LIMITS = [
            (-0.5, 1.5), (-0.5, 0.5), (0.0, 0.6),
            (-0.5, 0.5), (-0.8, 0.8), (-0.8, 0.8),
        ]
        old_ctrl = data.ctrl[gantry_act_ids].copy()
        if len(gantry_act_ids) == n_gantry_dof:
            new_ctrl = old_ctrl + q_dot_cmd * dt
            # Clamp to joint limits
            for i in range(n_gantry_dof):
                lo, hi = GANTRY_LIMITS[i]
                new_ctrl[i] = np.clip(new_ctrl[i], lo, hi)
            data.ctrl[gantry_act_ids] = new_ctrl
            ctrl_change = np.max(np.abs(new_ctrl - old_ctrl))
            if ctrl_change > max_ctrl_change:
                max_ctrl_change = ctrl_change

        # Apply finger control - move from home pose toward joint limits
        primary_finger = primary_sm.finger_idx if primary_sm is not None else None
        for aid, meta in actuator_meta.items():
            f_idx, suffix = meta
            if suffix == "abd" or (primary_finger is not None and f_idx != primary_finger):
                model.actuator_gainprm[aid, 0] = 0.0
                model.actuator_biasprm[aid, 1] = 0.0
                home = finger_home_targets.get(aid, 0.0)
                if act_ctrl_limited[aid]:
                    lo, hi = act_ctrl_ranges[aid]
                    home = float(np.clip(home, lo, hi))
                data.ctrl[aid] = home
            else:
                model.actuator_gainprm[aid, 0] = finger_gain_enabled
                model.actuator_biasprm[aid, 1] = finger_bias_enabled
        for f_idx, curl in finger_targets.items():
            if primary_finger is not None and f_idx != primary_finger:
                curl = 0.0
            f_alpha = dt / (FINGER_SMOOTHING_TAU + dt)
            smooth_finger_targets[f_idx] = (
                f_alpha * curl + (1 - f_alpha) * smooth_finger_targets[f_idx]
            )
            smooth_finger_targets[f_idx] = min(smooth_finger_targets[f_idx], MAX_FINGER_CURL)
            acts = finger_act_map.get(f_idx, [])
            if acts:
                for act_id in acts:
                    home = actuator_home_ctrl[act_id]
                    target = home
                    if act_ctrl_limited[act_id]:
                        lo, hi = act_ctrl_ranges[act_id]
                        target = home + smooth_finger_targets[f_idx] * FINGER_FLEX_RANGE * (hi - home)
                        target = np.clip(target, lo, hi)
                    data.ctrl[act_id] = float(target)
            if curl > max_curl:
                max_curl = curl

        # Step physics
        mujoco.mj_step(model, data)
        sim_t += dt

        # Check for problems
        curr_max_qvel = np.max(np.abs(data.qvel))
        if curr_max_qvel > max_qvel:
            max_qvel = curr_max_qvel

        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            print(f"[t={sim_t:.3f}s] NaN DETECTED!")
            explosion_t = sim_t
            break

        if curr_max_qvel > 50:
            print(f"[t={sim_t:.3f}s] HIGH VELOCITY: max|qvel|={curr_max_qvel:.1f}")
            # Find which joint
            for jid in range(model.njnt):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                qvel_idx = model.jnt_dofadr[jid]
                if np.abs(data.qvel[qvel_idx]) > 20:
                    print(f"  {name}: qvel={data.qvel[qvel_idx]:.1f}")
            explosion_t = sim_t
            break

        # Print status every 0.5 seconds
        if step % int(0.5 / dt) == 0:
            modes = [sm.mode.name for sm in active_machines]
            print(f"[t={sim_t:.2f}s] max|qvel|={curr_max_qvel:.2f}, modes={modes}, max_ctrl_change={max_ctrl_change:.4f}")

    print()
    print("=== SUMMARY ===")
    print(f"Max |qvel| observed: {max_qvel:.2f}")
    print(f"Max ctrl change per step: {max_ctrl_change:.4f}")
    print(f"Max curl value: {max_curl:.2f}")

    if explosion_t > 0:
        print(f"EXPLOSION at t={explosion_t:.3f}s")
    else:
        print("NO EXPLOSION")

    # Final finger joint states
    print()
    print("=== FINAL FINGER STATES ===")
    finger_joints = ["th_abd", "th_j2", "th_j3", "th_j4",
                     "ff_abd", "ff_j2", "ff_j3", "ff_j4",
                     "mf_abd", "mf_j2", "mf_j3", "mf_j4"]
    for jname in finger_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            qpos_idx = model.jnt_qposadr[jid]
            qvel_idx = model.jnt_dofadr[jid]
            lo, hi = model.jnt_range[jid]
            q = data.qpos[qpos_idx]
            qd = data.qvel[qvel_idx]
            in_range = "OK" if lo <= q <= hi else "OUT OF RANGE!"
            print(f"  {jname:10s}: q={q:7.3f} ({lo:.2f},{hi:.2f}) qd={qd:7.2f} {in_range}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/diagnose_control.py [json_path]")
    else:
        diagnose_control(sys.argv[1])
