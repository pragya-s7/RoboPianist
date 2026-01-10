# tests/sim_loop.py
import sys
import os
import json
from typing import Optional
import numpy as np
import time
import mujoco
import mujoco.viewer
from scipy import sparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.planning.piano_geometry import PianoGeometry
from src.control.state_machine import KeyStateMachine, ContactMode
from src.control.servo import ChordServo, RobotState, ConstraintBlock
from src.control.constraint_factory import ConstraintFactory
from src.control.key_progress import normalize_key_travel
from src.planning.kinematics import resolve_model_path

FINGER_SUFFIX_ORDER = ["abd", "j2", "j3", "j4"]


def get_actuator_indices(model):
    """
    STRICT NAME CHECKING FOR GANTRY AND FINGERS

    Returns:
        gantry_ids: [int] actuator indices for tx, ty, tz, rx, ry, rz
        finger_map: {finger_idx: [actuator_ids]}
    """
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
    """
    Map gantry joint names -> (qpos index, qvel index).

    Returns:
        joint_names: [str] in consistent order
        qpos_indices: [int]
        qvel_indices: [int]
    """
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


def select_primary_machine(active_machines, t):
    """
    Heuristic: choose the 'most urgent' contact to drive the gantry.

    Preference:
      1. Any STRIKE mode machine, closest in time to onset.
      2. Else PRETOUCH / HOLD.
      3. Else any remaining (HOVER / RELEASE) if nothing else.

    Returns:
        KeyStateMachine or None
    """
    if not active_machines:
        return None

    # Bucket by mode priority
    def mode_priority(m):
        if m.mode == ContactMode.STRIKE:
            return 0
        if m.mode in (ContactMode.PRETOUCH, ContactMode.HOLD):
            return 1
        return 2  # HOVER / RELEASE

    candidates = sorted(
        active_machines,
        key=lambda sm: (mode_priority(sm), abs(sm.onset - t)),
    )
    return candidates[0]


def _build_separation_block(
    factory: ConstraintFactory,
    finger_kinematics: dict[int, tuple[np.ndarray, np.ndarray]],
    d_min: float = 0.015,
) -> Optional[ConstraintBlock]:
    """
    Given per-finger kinematics:

        finger_kinematics[f] = (pos, J_pos) where:
            pos   : (3,)
            J_pos : (3, ndof)

    Build a combined separation ConstraintBlock enforcing a minimum
    3D separation d_min between each pair of active fingertips.

    We choose n as the unit vector along (p_i - p_j), so the constraint:

        nᵀ (J_i - J_j) q̇ + s_sep ≥ d_min - nᵀ (p_i - p_j)

    pushes the fingers apart along their current relative direction.
    """
    fingers = sorted(finger_kinematics.keys())
    if len(fingers) < 2:
        return None

    blocks = []

    for idx_i in range(len(fingers)):
        for idx_j in range(idx_i + 1, len(fingers)):
            fi = fingers[idx_i]
            fj = fingers[idx_j]

            p_i, J_pos_i = finger_kinematics[fi]
            p_j, J_pos_j = finger_kinematics[fj]

            diff = p_i - p_j
            dist = float(np.linalg.norm(diff))
            if dist < 1e-4:
                # Practically coincident; skip this pair
                continue

            n = diff / dist  # unit vector from finger j to i

            block = factory.create_separation_constraint(
                J_i=J_pos_i,
                J_j=J_pos_j,
                p_i=p_i,
                p_j=p_j,
                n=n,
                d_min=d_min,
            )
            if block.A.shape[0] > 0:
                blocks.append(block)

    if not blocks:
        return None

    A = sparse.vstack([b.A for b in blocks], format="csc")
    l = np.concatenate([b.l for b in blocks])
    u = np.concatenate([b.u for b in blocks])

    return ConstraintBlock(A=A, l=l, u=u)


def run_sim(json_file):
    with open(json_file, "r") as f:
        data_json = json.load(f)

    # Support both legacy and pipeline formats, but prefer pipeline:
    #   pipeline: {"source": ..., "total_events": ..., "events": [...]} 
    #   legacy:   {"events": [...]} 
    events_raw = data_json.get("events", data_json) if isinstance(data_json, dict) else data_json
    if not events_raw:
        raise RuntimeError(f"No 'events' field found in {json_file}")

    # Handle both "onset_sec" and "onset" formats
    def get_onset(x):
        return x.get("onset_sec", x.get("onset", 0.0))

    events = sorted(events_raw, key=get_onset)

    try:
        xml_path = resolve_model_path()
        print(f"Loaded Model Path: {xml_path}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    piano_geo = PianoGeometry()

    gantry_act_ids, finger_act_map = get_actuator_indices(model)
    use_shadow = any(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("rh_A_")
        for i in range(model.nu)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    )
    print(f"DEBUG: use_shadow = {use_shadow}")
    act_ctrl_ranges = model.actuator_ctrlrange.copy()
    act_ctrl_limited = model.actuator_ctrllimited.copy()
    act_trn_type = model.actuator_trntype.copy()
    act_trn_id = model.actuator_trnid.copy()
    jnt_limited = model.jnt_limited.copy()
    jnt_range = model.jnt_range.copy()
    # Finger actuator metadata + gain presets
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
    # Extra damping for finger joints to reduce oscillations
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        if name.startswith(("th_", "ff_", "mf_", "rf_", "lf_")) or name.startswith("rh_"):
            dof = model.jnt_dofadr[jid]
            model.dof_damping[dof] = max(model.dof_damping[dof], 12.0)
            model.dof_frictionloss[dof] = max(model.dof_frictionloss[dof], 1.0)

    # --- GANTRY JOINT MAPPING (qpos / qvel indices) ---
    joint_names, gantry_qpos_ids, gantry_qvel_ids = build_gantry_joint_maps(model)
    n_gantry_dof = len(gantry_qpos_ids)


    # --- SERVO SETUP (Layer A) ---
    servo = ChordServo(n_dof=n_gantry_dof)

    # fingertip sites: tip_1 .. tip_5
    finger_site_ids: dict[int, int] = {}
    for f_idx in range(1, 6):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"tip_{f_idx}")
        if sid < 0:
            print(f"WARNING: Site 'tip_{f_idx}' not found. Separation for that finger will be disabled.")
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

    # --- INITIAL POSE ---
    mid_c = piano_geo.get_key_location(60)
    # x = key center, y = -0.22 (back), z = 0.15 (hover), rx = 0, ry = 0, rz = 0
    base_y = 0.20 if use_shadow else -0.15
    base_z = 0.25 if use_shadow else 0.12
    SHADOW_PITCH = -0.35
    home_ry = SHADOW_PITCH if use_shadow else 0.0
    home_pose = np.array([mid_c.center_x, base_y, base_z, 0.0, home_ry, 0.0])
    print(f"DEBUG: home_pose = {home_pose}")
    print(f"DEBUG: wrist_id = {wrist_id} ({mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, wrist_id)})")

    mujoco.mj_resetData(model, data)

    # 1. Set Joint Positions (teleport robot to start)
    for i, q_idx in enumerate(gantry_qpos_ids):
        data.qpos[q_idx] = home_pose[i]

    # 2. Set Actuator Targets (hold robot at start)
    if len(gantry_act_ids) == 6:
        data.ctrl[gantry_act_ids] = home_pose

    mujoco.mj_forward(model, data)
    print(f"DEBUG: Initial Wrist after mj_forward: {data.xpos[wrist_id]}")

    # Finger home pose (used as neutral target for curls)
    finger_home_targets = {}
    actuator_home_ctrl = data.ctrl.copy()
    for acts in finger_act_map.values():
        for act_id in acts:
            if act_trn_type[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = act_trn_id[act_id, 0]
                qpos_addr = model.jnt_qposadr[jid]
                finger_home_targets[act_id] = float(data.qpos[qpos_addr])

    # Initialize finger controls to current joint positions to avoid large initial torques
    for act_id, home in finger_home_targets.items():
        if act_ctrl_limited[act_id]:
            lo, hi = act_ctrl_ranges[act_id]
            home = float(np.clip(home, lo, hi))
        data.ctrl[act_id] = home
        actuator_home_ctrl[act_id] = home

    # Headless run
    dt = model.opt.timestep
    sim_t = 0.0
    event_idx = 0
    active_machines: list[KeyStateMachine] = []

    # Phase 1: ConstraintFactory knows about ndof (gantry only)
    constraint_factory = ConstraintFactory(ndof=n_gantry_dof)

    # --- TARGET SMOOTHING ---
    smooth_wrist_target = np.array([mid_c.center_x, base_y, base_z])
    smooth_wrist_rot = np.array([0.0, 0.0, 0.0])
    SHADOW_PITCH = -0.35
    ROT_SMOOTHING_TAU = 0.25
    TARGET_SMOOTHING_TAU_XY = 0.35
    TARGET_SMOOTHING_TAU_Z = 0.18
    smooth_finger_targets = {f: 0.0 for f in range(1, 6)}
    FINGER_SMOOTHING_TAU = 0.12 if use_shadow else 0.08
    MAX_FINGER_CURL = 0.45 if use_shadow else 0.6
    FINGER_FLEX_RANGE = 0.35 if use_shadow else 0.5

    GANTRY_KP = np.array([20.0, 20.0, 20.0, 40.0, 40.0, 40.0])
    GANTRY_KD = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    GANTRY_VEL_MAX = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])

    print("Starting headless simulation for 5 seconds...")
    max_steps = int(5.0 / dt)
    
    for step in range(max_steps):
        # --- READ CURRENT KEY DEPTHS (for state machine & constraints) ---
        key_depths = {}
        for midi in range(21, 109):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"k{midi}")
            if jid >= 0:
                addr = model.jnt_qposadr[jid]
                key_depths[midi] = data.qpos[addr]

        # --- SPAWN EVENTS (KeyStateMachines), now using Layer C contacts ---
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
                            if use_shadow:
                                sm.hover_height = 0.03
                                sm.pretouch_clearance = 0.003
                                sm.strike_overtravel = 0.009

                            sm.approach_lead = max(0.01, strike - pretouch)
                            sm.hold_duration = max(0.01, hold_end - strike)
                            sm.release_duration = max(0.01, release_end - hold_end)

                            active_machines.append(sm)
                    else:
                        fingers = e.get("fingering", [])
                        pitches = e.get("pitches", [])
                        if fingers and pitches:
                            for p, f in zip(pitches, fingers):
                                sm = KeyStateMachine(p, get_onset(e), f)
                                if use_shadow:
                                    sm.hover_height = 0.03
                                    sm.pretouch_clearance = 0.003
                                    sm.strike_overtravel = 0.009
                                active_machines.append(sm)
                event_idx += 1
            else:
                break

        # --- UPDATE ALL STATE MACHINES ---
        active_machines = [sm for sm in active_machines if not sm.done]
        for sm in active_machines:
            sm.update(sim_t, key_depths.get(sm.pitch, 0.0))

        # --- FINGER TARGETS (curl) ---
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

        # --- SIMPLE GANTRY CONTROL USING WRIST TARGETS ---
        primary_sm = select_primary_machine(active_machines, sim_t)

        wrist_target = None
        for e in reversed(events[:event_idx]):
            if e.get("staff") == 1 and "wrist_target" in e:
                event_time = e.get("onset_sec", e.get("onset", 0))
                if event_time <= sim_t and sim_t - event_time < 2.0:
                    wrist_target = np.array(e["wrist_target"], dtype=float)
                    break

        if wrist_target is None:
            wrist_target = np.array([mid_c.center_x, base_y, base_z])

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
                if use_shadow:
                    desired_wrist.append(
                        np.array([
                            desired_tip[0] - tip_offset_world[0],
                            base_y,
                            desired_tip[2] - tip_offset_world[2],
                        ])
                    )
                else:
                    desired_wrist.append(desired_tip - tip_offset_world)
            if desired_wrist:
                wrist_target[:3] = np.mean(desired_wrist, axis=0)

        # --- APPLY TARGET SMOOTHING ---
        alpha_xy = dt / (TARGET_SMOOTHING_TAU_XY + dt)
        alpha_z = dt / (TARGET_SMOOTHING_TAU_Z + dt)
        smooth_wrist_target[0] = alpha_xy * wrist_target[0] + (1 - alpha_xy) * smooth_wrist_target[0]
        smooth_wrist_target[1] = alpha_xy * wrist_target[1] + (1 - alpha_xy) * smooth_wrist_target[1]
        smooth_wrist_target[2] = alpha_z * wrist_target[2] + (1 - alpha_z) * smooth_wrist_target[2]
        smooth_wrist_target[0] = np.clip(smooth_wrist_target[0], -0.5, 1.5)
        smooth_wrist_target[1] = np.clip(smooth_wrist_target[1], -0.5, 0.5)
        min_z = 0.08 if use_shadow else 0.0
        smooth_wrist_target[2] = np.clip(smooth_wrist_target[2], min_z, 0.6)

        desired_rot = np.array([0.0, SHADOW_PITCH, 0.0]) if use_shadow else np.zeros(3)
        rot_alpha = dt / (ROT_SMOOTHING_TAU + dt)
        smooth_wrist_rot = rot_alpha * desired_rot + (1 - rot_alpha) * smooth_wrist_rot

        # --- APPLY GANTRY CONTROL (Direct Position) ---
        
        target_pos = np.zeros(6)
        target_pos[0:3] = smooth_wrist_target
        target_pos[3:6] = smooth_wrist_rot

        GANTRY_LIMITS = [
            (-0.5, 1.5),   # tx
            (-0.5, 0.5),   # ty
            (0.0, 0.6),    # tz
            (-0.5, 0.5),   # rx
            (-0.8, 0.8),   # ry
            (-0.8, 0.8),   # rz
        ]
        
        if len(gantry_act_ids) == n_gantry_dof:
            for i in range(n_gantry_dof):
                lo, hi = GANTRY_LIMITS[i]
                val = np.clip(target_pos[i], lo, hi)
                data.ctrl[gantry_act_ids[i]] = val

        primary_finger = None if use_shadow else (primary_sm.finger_idx if primary_sm is not None else None)
        for aid, meta in actuator_meta.items():
            f_idx, suffix = meta
            if use_shadow:
                continue
            if suffix == "abd":
                model.actuator_gainprm[aid, 0] = 0.0
                model.actuator_biasprm[aid, 1] = 0.0
                home = finger_home_targets.get(aid, 0.0)
                if act_ctrl_limited[aid]:
                    lo, hi = act_ctrl_ranges[aid]
                    home = float(np.clip(home, lo, hi))
                data.ctrl[aid] = home
            elif (primary_finger is not None and f_idx != primary_finger):
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

        mujoco.mj_step(model, data)
        sim_t += dt

        if step % 100 == 0:
            wrist_pos = data.xpos[wrist_id]
            print(f"t={sim_t:.2f} | Wrist: {wrist_pos} | Contacts: {data.ncon} | Active Machines: {len(active_machines)}")

    # --- Post-run diagnostics summary ---
    print("\n=== CHORD-Ω Servo Diagnostics (Tier 1) ===")
    print(f"Max zone slack: {global_max_zone:.4e}")
    print(f"Max sep  slack: {global_max_sep:.4e}")
    print(f"Max key  slack: {global_max_key:.4e}")
    print("  Interpretation:")
    print("    • High key slack  → timing / authority / contact issues.")
    print("    • High zone slack → gantry reach / geometry issues.")
    print("    • High sep  slack → multi-finger collision / crowding.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tests.sim_loop [json_path]")
    else:
        run_sim(sys.argv[1])