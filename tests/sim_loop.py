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
from src.control.layer_b import LayerB
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
            if name.startswith("rh_A_TH") and name.endswith(("THJ4", "THJ3", "THJ2", "THJ1", "THJ5")):
                finger_map[1][name] = i
            elif name.startswith("rh_A_FF") and name.endswith(("FFJ4", "FFJ3", "FFJ0")):
                finger_map[2][name] = i
            elif name.startswith("rh_A_MF") and name.endswith(("MFJ4", "MFJ3", "MFJ0")):
                finger_map[3][name] = i
            elif name.startswith("rh_A_RF") and name.endswith(("RFJ4", "RFJ3", "RFJ0")):
                finger_map[4][name] = i
            elif name.startswith("rh_A_LF") and name.endswith(("LFJ5", "LFJ4", "LFJ3", "LFJ0")):
                finger_map[5][name] = i
        else:
            if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
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
    act_ctrl_ranges = model.actuator_ctrlrange.copy()
    act_ctrl_limited = model.actuator_ctrllimited.copy()
    act_trn_type = model.actuator_trntype.copy()
    act_trn_id = model.actuator_trnid.copy()
    jnt_limited = model.jnt_limited.copy()
    jnt_range = model.jnt_range.copy()
    # Finger actuator metadata + gain presets
    actuator_meta = {}
    finger_gain_enabled = 30.0 if use_shadow else 10.0
    finger_bias_enabled = 0.0 if use_shadow else -10.0
    shadow_finger_kp = 120.0
    shadow_finger_kd = 2.5
    shadow_finger_force = 35.0
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
        if use_shadow:
            # Add light damping on Shadow joints to prevent instability.
            if name in ("tx", "ty", "tz", "rx", "ry", "rz"):
                dof = model.jnt_dofadr[jid]
                model.dof_damping[dof] = max(model.dof_damping[dof], 12.0)
                model.dof_frictionloss[dof] = max(model.dof_frictionloss[dof], 1.0)
            elif name.startswith("rh_"):
                dof = model.jnt_dofadr[jid]
                model.dof_damping[dof] = max(model.dof_damping[dof], 1.5)
                model.dof_frictionloss[dof] = max(model.dof_frictionloss[dof], 0.2)
            continue
        if name.startswith(("th_", "ff_", "mf_", "rf_", "lf_")) or name.startswith("rh_"):
            dof = model.jnt_dofadr[jid]
            model.dof_damping[dof] = max(model.dof_damping[dof], 12.0)
            model.dof_frictionloss[dof] = max(model.dof_frictionloss[dof], 1.0)
    # Increase key joint damping to reduce contact chatter.
    for midi in range(21, 109):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"k{midi}")
        if jid >= 0:
            dof = model.jnt_dofadr[jid]
            model.dof_damping[dof] = max(model.dof_damping[dof], 2.0)
            model.dof_frictionloss[dof] = max(model.dof_frictionloss[dof], 0.5)

    # --- GANTRY JOINT MAPPING (qpos / qvel indices) ---
    joint_names, gantry_qpos_ids, gantry_qvel_ids = build_gantry_joint_maps(model)
    n_gantry_dof = len(gantry_qpos_ids)


    # --- SERVO SETUP (Layer A) ---
    servo = ChordServo(n_dof=n_gantry_dof)
    layer_b = LayerB(update_frequency=100.0)

    # fingertip sites: tip_1 .. tip_5
    finger_site_ids: dict[int, int] = {}
    for f_idx in range(1, 6):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"tip_{f_idx}")
        if sid < 0:
            print(f"WARNING: Site 'tip_{f_idx}' not found. Separation for that finger will be disabled.")
        finger_site_ids[f_idx] = sid
    tip_geom_ids = {}
    for f_idx in range(1, 6):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tip_geom_{f_idx}")
        if gid >= 0:
            tip_geom_ids[f_idx] = gid
    hand_geom_ids = []
    for gid in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
        if bname and bname.startswith("rh_"):
            hand_geom_ids.append(gid)
    key_geom_ids = {}
    for gid in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
        if bname and bname.startswith("key_"):
            try:
                midi = int(bname.split("_", 1)[1])
            except ValueError:
                continue
            key_geom_ids.setdefault(midi, []).append(gid)

    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_yaw")
    if wrist_id < 0:
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
    if wrist_id < 0:
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    if wrist_id < 0:
        raise RuntimeError("Body 'wrist_yaw' or palm body not found. Check scene.xml.")

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    # Shadow hand: infer which actuator direction actually curls the fingertip down.
    shadow_act_curl = {}
    if use_shadow:
        tmp_data = mujoco.MjData(model)

        def _tip_z_for_ctrl(act_id: int, ctrl_val: float, tip_id: int) -> float:
            mujoco.mj_resetData(model, tmp_data)
            tmp_data.ctrl[:] = 0.0
            tmp_data.ctrl[act_id] = ctrl_val
            for _ in range(200):
                mujoco.mj_step(model, tmp_data)
            return float(tmp_data.site_xpos[tip_id][2])

        finger_act_mag = {}
        for f_idx, acts in finger_act_map.items():
            finger_act_mag[f_idx] = {}
            tip_id = finger_site_ids.get(f_idx, -1)
            if tip_id is None or tip_id < 0:
                continue
            for act_id in acts:
                lo, hi = act_ctrl_ranges[act_id]
                if not act_ctrl_limited[act_id] or abs(hi - lo) < 1e-6:
                    continue
                z_lo = _tip_z_for_ctrl(act_id, lo, tip_id)
                z_hi = _tip_z_for_ctrl(act_id, hi, tip_id)
                p_lo = tmp_data.site_xpos[tip_id].copy()
                # Recompute for hi to get full delta vector.
                mujoco.mj_resetData(model, tmp_data)
                tmp_data.ctrl[:] = 0.0
                tmp_data.ctrl[act_id] = hi
                for _ in range(200):
                    mujoco.mj_step(model, tmp_data)
                p_hi = tmp_data.site_xpos[tip_id].copy()
                d = p_hi - p_lo
                dz = d[2]
                dxy = float((d[0] ** 2 + d[1] ** 2) ** 0.5)
                score = (-dz) - 0.3 * dxy
                direction = 1 if dz < 0.0 else -1
                shadow_act_curl[act_id] = (direction, score)
                finger_act_mag[f_idx][act_id] = score
        # Keep only the strongest curl actuators per finger (avoids extra joints).
        keep_per_finger = 2
        for f_idx, mags in finger_act_mag.items():
            if not mags:
                continue
            sorted_acts = sorted(mags.items(), key=lambda kv: kv[1], reverse=True)
            keep = {act_id for act_id, _ in sorted_acts[:keep_per_finger]}
            for act_id, mag in mags.items():
                if act_id not in keep or mag <= 0.0:
                    shadow_act_curl[act_id] = (0, mag)

    # --- INITIAL POSE ---
    mid_c = piano_geo.get_key_location(60)
    key_min_x = piano_geo.get_key_location(21).center_x
    key_max_x = piano_geo.get_key_location(108).center_x
    # Use a conservative home pose; precise targeting will use calibrated tip offsets.
    # Fingers point toward piano (+Y), so wrist needs to be pulled back (-Y)
    base_y = -0.55 if use_shadow else -0.15  # Further back to reach keys
    base_z = 0.13 if use_shadow else 0.11
    # No pitch needed - fingers curl down to press keys. Pitch causes finger tilt.
    SHADOW_PITCH = 0.0
    SHADOW_ROLL = 0.0
    home_rx = SHADOW_ROLL
    home_ry = SHADOW_PITCH
    home_pose = np.array([mid_c.center_x, base_y, base_z, home_rx, home_ry, 0.0])

    mujoco.mj_resetData(model, data)

    # 1. Set Joint Positions (teleport robot to start)
    for i, q_idx in enumerate(gantry_qpos_ids):
        data.qpos[q_idx] = home_pose[i]

    # 2. Set Actuator Targets (hold robot at start)
    if len(gantry_act_ids) == 6:
        data.ctrl[gantry_act_ids] = home_pose

    mujoco.mj_forward(model, data)

    # Cache key site positions at rest for more reliable targeting.
    key_site_ids = {}
    key_site_world = {}
    for midi in range(21, 109):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"s_{midi}")
        if sid >= 0:
            key_site_ids[midi] = sid
            key_site_world[midi] = data.site_xpos[sid].copy()
    if key_site_world:
        key_min_x = min(pos[0] for pos in key_site_world.values())
        key_max_x = max(pos[0] for pos in key_site_world.values())

    # Calibrate fingertip offsets in world frame relative to wrist body.
    wrist_pos_home = data.xpos[wrist_id].copy()
    finger_tip_offsets = {}
    for f_idx, sid in finger_site_ids.items():
        if sid is None or sid < 0:
            continue
        tip_pos = data.site_xpos[sid].copy()
        finger_tip_offsets[f_idx] = tip_pos - wrist_pos_home

    # AUTO-CALIBRATE: Compute GANTRY position to place fingertip at key
    # Account for kinematic chain offset between gantry and wrist
    mid_key_pos = key_site_world.get(60)  # Middle C (MIDI 60)

    # Compute gantry-to-wrist offset (due to kinematic chain + rotation)
    gantry_pos_init = np.array([data.qpos[q_idx] for q_idx in gantry_qpos_ids[:3]])
    gantry_to_wrist_offset = wrist_pos_home - gantry_pos_init

    # Use middle finger (3) as reference, fallback to index (2) or ring (4)
    ref_finger = 3
    ref_offset = finger_tip_offsets.get(ref_finger)
    if ref_offset is None:
        ref_finger = 2
        ref_offset = finger_tip_offsets.get(ref_finger)
    if ref_offset is None:
        ref_finger = 4
        ref_offset = finger_tip_offsets.get(ref_finger)

    if mid_key_pos is not None and ref_offset is not None:
        # Step 1: Where should fingertip be? (key position + hover clearance)
        hover_clearance = 0.02
        target_fingertip = np.array([mid_key_pos[0], mid_key_pos[1], mid_key_pos[2] + hover_clearance])

        # Step 2: Where should wrist be? (fingertip - fingertip_offset_from_wrist)
        target_wrist = target_fingertip - ref_offset

        # Step 3: Where should gantry be? (wrist - gantry_to_wrist_offset)
        target_gantry = target_wrist - gantry_to_wrist_offset

        base_y = float(np.clip(target_gantry[1], -0.8, 0.5))  # Extended Y range
        base_z = float(np.clip(target_gantry[2], 0.02, 0.55))

        print(f"[CALIBRATION] Gantry target: Y={base_y:.3f}, Z={base_z:.3f}")

        # APPLY calibrated position to gantry
        home_pose[1] = base_y
        home_pose[2] = base_z
        for i, q_idx in enumerate(gantry_qpos_ids):
            data.qpos[q_idx] = home_pose[i]
        if len(gantry_act_ids) == 6:
            data.ctrl[gantry_act_ids] = home_pose
        mujoco.mj_forward(model, data)

        # Verify calibration
        new_tip_pos = data.site_xpos[finger_site_ids[ref_finger]].copy()
        print(f"[CALIBRATION] Fingertip above key: {(new_tip_pos[2] - mid_key_pos[2])*100:.1f}cm")

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
    prev_finger_ctrl = {int(i): float(val) for i, val in enumerate(data.ctrl)}

    print(f"--- PLAYING (servo-driven gantry, Tier 1) : {json_file} ---")
    print("NOTE: For best results, generate this JSON via scripts/run_pipeline.py.\n" 
          "      (Fingerings, wrist targets, and Layer C contacts are used.)\n")
    debug_fingers = os.environ.get("DEBUG_FINGERS", "0") == "1"
    debug_next_t = 0.0

    # macOS Helper
    try:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        if "mjpython" in str(e):
            print("\n!!! macOS ERROR DETECTED !!!")
            print("Run this command instead:\n")
            print(f"    mjpython -m tests.sim_loop {json_file}\n")
            sys.exit(1)
        else:
            raise e

    # Diagnostics aggregation over entire run
    global_max_zone = 0.0
    global_max_sep = 0.0
    global_max_key = 0.0

    with viewer_ctx as viewer:
        dt = model.opt.timestep
        servo.dt = dt
        sim_t = 0.0
        event_idx = 0
        active_machines: list[KeyStateMachine] = []

        # Phase 1: ConstraintFactory knows about ndof (gantry only)
        constraint_factory = ConstraintFactory(ndof=n_gantry_dof)

        # --- TARGET SMOOTHING ---
        # Initialize smooth targets to ACTUAL current position to prevent initial jump
        current_gantry_pos = np.array([data.qpos[i] for i in gantry_qpos_ids])
        smooth_wrist_target = current_gantry_pos[0:3].copy()
        smooth_wrist_rot = current_gantry_pos[3:6].copy()
        
        smooth_desired_z = float(current_gantry_pos[2])
        SHADOW_PITCH = 0.0  # No pitch - fingers level, curling does the work
        ROT_SMOOTHING_TAU = 0.25
        TARGET_SMOOTHING_TAU_XY = 0.20  # Faster for responsive positioning
        TARGET_SMOOTHING_TAU_Z = 0.10  # Fast z response for key pressing
        # Alpha is computed per-step: alpha = dt / (tau + dt)
        # Smooth finger curl targets to prevent snap-to extremes
        smooth_finger_targets = {f: 0.0 for f in range(1, 6)}
        finger_targets_cmd = {f: 0.0 for f in range(1, 6)}
        last_finger_update = -1.0
        FINGER_SMOOTHING_TAU = 0.06 if use_shadow else 0.08  # faster Shadow response
        # Allow full joint range so STRIKE can actually depress keys.
        MAX_FINGER_CURL = 2.0
        FINGER_FLEX_RANGE = 1.8  # fraction of joint range used for curl

        # Gantry damping gains
        # Adjusted for stability (stiff rotation, damped translation)
        # Reduced rotation gains to prevent jitter, but kept strong enough to hold
        GANTRY_KP = np.array([20.0, 20.0, 20.0, 30.0, 30.0, 30.0])
        GANTRY_KD = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
        GANTRY_VEL_MAX = np.array([0.5, 1.0, 0.5, 1.0, 1.0, 1.0])
        servo.v_max = GANTRY_VEL_MAX
        servo.acc_limit = 8.0
        TIP_IK_KP = 6.0
        TIP_IK_DAMP = 0.05
        ROT_KP = np.array([10.0, 10.0, 10.0])
        ROT_KD = np.array([4.0, 4.0, 4.0])
        FORCE_GANTRY_POSE = True
        GANTRY_TRANSLATION_ONLY = True
        Y_LOCK = False
        # Allow gantry to move down for key pressing (fingertips need to reach key surface)
        Z_DOWN_RANGE = 0.04  # Allow 4cm downward movement for key pressing

        # Finger actuator rate limiting removed to keep joints stiff

        while viewer.is_running():
            step_start = time.time()



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
                # Look-ahead window 0.5s
                if get_onset(e) <= sim_t + 0.5:
                    # Right hand only for now (staff == 1)
                    if e["staff"] == 1:
                        contacts = e.get("contacts", [])
                        if contacts:
                            # Prefer Layer C contact schedule if available
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
                                    sm.hover_height = 0.025  # Lower hover for faster approach
                                    sm.pretouch_clearance = 0.002  # Closer pretouch
                                    sm.strike_overtravel = 0.020  # 20mm overtravel for reliable key press
                                    sm.strike_timeout = 0.35  # Give more time for key press

                                # Infer timing parameters from contact spec
                                sm.approach_lead = max(0.01, strike - pretouch)
                                sm.hold_duration = max(0.05, hold_end - strike)  # Minimum hold time
                                sm.release_duration = max(0.01, release_end - hold_end)

                                active_machines.append(sm)
                        else:
                            # Fallback: legacy behavior using fingering + onset
                            fingers = e.get("fingering", [])
                            pitches = e.get("pitches", [])
                            if fingers and pitches:
                                for p, f in zip(pitches, fingers):
                                    sm = KeyStateMachine(p, get_onset(e), f)
                                    if use_shadow:
                                        sm.hover_height = 0.025
                                        sm.pretouch_clearance = 0.002
                                        sm.strike_overtravel = 0.020
                                        sm.strike_timeout = 0.35
                                    active_machines.append(sm)
                    event_idx += 1
                else:
                    break

            # --- UPDATE ALL STATE MACHINES ---
            active_machines = [sm for sm in active_machines if not sm.done]
            for sm in active_machines:
                sm.update(sim_t, key_depths.get(sm.pitch, 0.0))


            # --- FINGER KINEMATICS (positions + Jacobians in gantry coords) ---
            finger_kinematics = {}
            active_fingers = {sm.finger_idx for sm in active_machines}
            for f_idx in active_fingers:
                sid = finger_site_ids.get(f_idx, -1)
                if sid is None or sid < 0:
                    continue
                mujoco.mj_jacSite(model, data, jacp, jacr, sid)
                pos = data.site_xpos[sid].copy()
                J_pos_full = jacp.copy()
                J_pos = J_pos_full[:, gantry_qvel_ids]
                finger_kinematics[f_idx] = (pos, J_pos)

            # --- STABLE GANTRY CONTROL ---
            # Key insight: Compute wrist target from key geometry ONCE, not every frame.
            # Don't use feedback IK based on current finger positions (causes oscillation).

            active_press_machines = [
                sm for sm in active_machines
                if sm.mode in (ContactMode.PRETOUCH, ContactMode.STRIKE, ContactMode.HOLD)
            ]
            primary_sm = select_primary_machine(active_press_machines, sim_t)
            # Enable fingertip collisions for fingers that are in PRETOUCH, STRIKE, or HOLD modes.
            # This allows physical contact with keys for pressing.
            active_fingers = set()
            active_pitches = set()
            for sm in active_machines:
                if sm.mode in (ContactMode.PRETOUCH, ContactMode.STRIKE, ContactMode.HOLD):
                    active_fingers.add(sm.finger_idx)
                    active_pitches.add(sm.pitch)

            # Disable all hand collisions first, then selectively enable
            for gid in hand_geom_ids:
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = 0

            # Enable collisions for active fingertips
            for f_idx in active_fingers:
                gid = tip_geom_ids.get(f_idx)
                if gid is not None:
                    model.geom_contype[gid] = 1
                    model.geom_conaffinity[gid] = 1

            # Enable collisions for active keys
            for midi, gids in key_geom_ids.items():
                enabled = midi in active_pitches
                for gid in gids:
                    model.geom_contype[gid] = 1 if enabled else 0
                    model.geom_conaffinity[gid] = 1 if enabled else 0

            # --- FINGER TARGETS (curl) ---
            finger_mode_map = {}
            for sm in active_machines:
                mode = sm.mode
                prev = finger_mode_map.get(sm.finger_idx)
                if prev is None:
                    finger_mode_map[sm.finger_idx] = mode
                else:
                    if prev != ContactMode.STRIKE and mode == ContactMode.STRIKE:
                        finger_mode_map[sm.finger_idx] = mode
                    elif prev == ContactMode.HOVER and mode in (ContactMode.PRETOUCH, ContactMode.HOLD):
                        finger_mode_map[sm.finger_idx] = mode
            if sim_t - last_finger_update >= 0.01:
                finger_targets = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
                for sm in active_machines:
                    val = 0.0
                    if sm.mode == ContactMode.STRIKE:
                        val = 1.8
                    elif sm.mode == ContactMode.HOLD:
                        val = 1.4
                    elif sm.mode == ContactMode.PRETOUCH:
                        val = 0.8
                    finger_targets[sm.finger_idx] = max(finger_targets[sm.finger_idx], val)
                    if sm.mode == ContactMode.STRIKE:
                        key_depth = key_depths.get(sm.pitch, 0.0)
                        press_thresh = sm._key_down_threshold()
                        if key_depth < press_thresh:
                            boost = min(0.4, 2.5 * (press_thresh - key_depth))
                            finger_targets[sm.finger_idx] = max(
                                finger_targets[sm.finger_idx], 1.2 + boost
                            )
                if primary_sm is not None:
                    finger_targets[primary_sm.finger_idx] = max(
                        finger_targets[primary_sm.finger_idx], 1.2
                    )
                # Slight curl on non-active fingers to keep them off the keys.
                default_curl = 0.15 if use_shadow else 0.1
                for f_idx in finger_targets:
                    if finger_targets[f_idx] <= 0.0:
                        finger_targets[f_idx] = default_curl
                finger_targets_cmd = finger_targets
                last_finger_update = sim_t

            if active_machines:
                # Compute wrist targets from calibrated fingertip offsets.
                target_wrist_positions = []
                for sm in active_machines:
                    if sm.mode in (ContactMode.PRETOUCH, ContactMode.STRIKE, ContactMode.HOLD):
                        tip_offset = finger_tip_offsets.get(sm.finger_idx)
                        if tip_offset is None:
                            continue
                        key_pos = key_site_world.get(sm.pitch)
                        if key_pos is None:
                            k_loc = piano_geo.get_key_location(sm.pitch)
                            key_pos = np.array([k_loc.center_x, k_loc.center_y, k_loc.center_z])
                        z_target, _, _ = sm.get_target(key_pos[2])
                        tip_target = np.array([key_pos[0], key_pos[1], z_target])
                        target_wrist = tip_target - tip_offset
                        target_wrist_positions.append(target_wrist)

                if target_wrist_positions:
                    wrist_target = np.mean(target_wrist_positions, axis=0)
                    wrist_target[0] = float(np.clip(wrist_target[0], key_min_x, key_max_x))
                else:
                    wrist_target = np.array([mid_c.center_x, base_y, base_z])
            else:
                wrist_target = np.array([mid_c.center_x, base_y, base_z])

            # --- APPLY TARGET SMOOTHING ---
            # Use slower smoothing for stable, non-oscillating motion
            alpha_xy = dt / (TARGET_SMOOTHING_TAU_XY + dt)
            alpha_z = dt / (TARGET_SMOOTHING_TAU_Z + dt)
            smooth_wrist_target[0] = alpha_xy * wrist_target[0] + (1 - alpha_xy) * smooth_wrist_target[0]
            smooth_wrist_target[1] = alpha_xy * wrist_target[1] + (1 - alpha_xy) * smooth_wrist_target[1]
            smooth_wrist_target[2] = alpha_z * wrist_target[2] + (1 - alpha_z) * smooth_wrist_target[2]

            # Clamp to gantry limits
            smooth_wrist_target[0] = np.clip(smooth_wrist_target[0], -0.5, 1.5)
            smooth_wrist_target[1] = np.clip(smooth_wrist_target[1], -0.8, 0.5)  # Extended Y range
            smooth_wrist_target[2] = np.clip(smooth_wrist_target[2], 0.04, 0.25)

            arch_pitch = -0.12 if active_press_machines else 0.0
            if use_shadow:
                desired_rot = np.array([0.0, SHADOW_PITCH + arch_pitch, 0.0])
            else:
                desired_rot = np.array([0.0, -0.1 + arch_pitch, 0.0])
            rot_alpha = dt / (ROT_SMOOTHING_TAU + dt)
            smooth_wrist_rot = rot_alpha * desired_rot + (1 - rot_alpha) * smooth_wrist_rot

            # --- LAYER B UPDATE (Timing & Contact Linearization) ---
            layer_b_output = {}
            if layer_b.should_update(sim_t):
                layer_b_output = layer_b.update(
                    current_time=sim_t,
                    state_machines=active_machines,
                    key_depths=key_depths,
                    finger_jacobians=finger_kinematics,
                    qvel_current=data.qvel[gantry_qvel_ids].copy(),
                )

            gantry_qpos = np.array([data.qpos[q_idx] for q_idx in gantry_qpos_ids])
            gantry_qvel = data.qvel[gantry_qvel_ids].copy()

            first_strike_time = min((sm.onset for sm in active_machines), default=None)
            idle_hold = (not active_press_machines) and (
                first_strike_time is None or sim_t < first_strike_time
            )

            # --- BUILD KEY SUCCESS CONSTRAINTS (Layer A) ---
            key_blocks = []
            for sm in active_machines:
                if sm.mode != ContactMode.STRIKE:
                    continue
                kin = finger_kinematics.get(sm.finger_idx)
                if kin is None:
                    continue
                _, J_pos = kin
                z_jac_row = J_pos[2, :]
                if layer_b_output.get("updated", False):
                    deadline = layer_b.get_adapted_deadline(sm.pitch, sm.finger_idx, sm.onset)
                else:
                    deadline = sm.onset
                dt_rem = deadline - sim_t
                k_current = normalize_key_travel(key_depths.get(sm.pitch, 0.0))
                key_blocks.append(
                    constraint_factory.create_key_success_constraint(
                        z_jac_row=z_jac_row,
                        dt_rem=dt_rem,
                        k_current=k_current,
                    )
                )

            key_constraints = None
            if key_blocks:
                A = sparse.vstack([b.A for b in key_blocks], format="csc")
                l = np.concatenate([b.l for b in key_blocks])
                u = np.concatenate([b.u for b in key_blocks])
                key_constraints = ConstraintBlock(A=A, l=l, u=u)

            # --- BUILD ZONE CONSTRAINTS (prevent deep penetration) ---
            zone_blocks = []
            tau_list = [0.5 * dt, dt]
            # Gantry translation workspace constraint (soft via s_zone)
            workspace_block = constraint_factory.create_gantry_workspace_constraint(
                gantry_qpos_translation=gantry_qpos[:3],
                gantry_qvel_indices=[0, 1, 2],
                workspace_min=np.array([-0.5, -0.5, 0.0]),
                workspace_max=np.array([1.5, 0.5, 0.6]),
                max_gantry_speed=0.5,
                dt=dt,
            )
            zone_blocks.append(workspace_block)
            for sm in active_machines:
                if sm.mode not in (ContactMode.PRETOUCH, ContactMode.STRIKE, ContactMode.HOLD):
                    continue
                kin = finger_kinematics.get(sm.finger_idx)
                if kin is None:
                    continue
                pos, J_pos = kin
                k_loc = piano_geo.get_key_location(sm.pitch)
                key_surface_z = k_loc.center_z
                z_min = key_surface_z - 0.002  # allow slight penetration
                a = np.array([0.0, 0.0, -1.0])
                b = -z_min
                zone_blocks.append(
                    constraint_factory.create_zone_constraint(
                        a=a,
                        b=b,
                        y=pos,
                        J_pos=J_pos,
                        tau_list=tau_list,
                    )
                )

            zone_constraints = None
            if zone_blocks:
                A = sparse.vstack([b.A for b in zone_blocks], format="csc")
                l = np.concatenate([b.l for b in zone_blocks])
                u = np.concatenate([b.u for b in zone_blocks])
                zone_constraints = ConstraintBlock(A=A, l=l, u=u)

            # --- SEPARATION CONSTRAINTS ---
            sep_constraints = _build_separation_block(
                factory=constraint_factory,
                finger_kinematics=finger_kinematics,
                d_min=0.015,
            )

            # --- NOMINAL GANTRY VELOCITY (for QP) ---
            target_pos = np.zeros(6)
            target_pos[0:3] = smooth_wrist_target
            target_pos[3:6] = smooth_wrist_rot

            q_dot_nom = GANTRY_KP * (target_pos - gantry_qpos) - GANTRY_KD * gantry_qvel

            # Hold position until the first strike time is reached to prevent early drift.
            if idle_hold:
                q_dot_nom[:3] = 0.0
            elif primary_sm is not None:
                kin = finger_kinematics.get(primary_sm.finger_idx)
                if kin is not None:
                    p_curr, _ = kin
                    key_pos = key_site_world.get(primary_sm.pitch)
                    if key_pos is None:
                        k_loc = piano_geo.get_key_location(primary_sm.pitch)
                        key_pos = np.array([k_loc.center_x, k_loc.center_y, k_loc.center_z])
                    z_target, _, _ = primary_sm.get_target(key_pos[2])
                    p_target = np.array([key_pos[0], key_pos[1], z_target])
                    p_err = p_target - p_curr
                    # Only drive gantry translations from fingertip error; keep rotations via PD.
                    q_dot_nom[:3] = 0.0
                    q_dot_nom[0] = TIP_IK_KP * p_err[0]
                    q_dot_nom[2] = TIP_IK_KP * p_err[2]
                    q_dot_nom[2] = np.clip(q_dot_nom[2], -0.2, 0.2)
            desired_wrist_z = base_z
            if primary_sm is not None and primary_sm.mode == ContactMode.STRIKE:
                # Drop wrist more aggressively during STRIKE to help fingertip reach key
                # Key travel is 12mm, so we need fingertip to move at least that much
                desired_wrist_z = base_z - 0.025
            elif primary_sm is not None and primary_sm.mode == ContactMode.PRETOUCH:
                # Slight drop during PRETOUCH to prepare for strike
                desired_wrist_z = base_z - 0.01
            smooth_desired_z = alpha_z * desired_wrist_z + (1 - alpha_z) * smooth_desired_z
            if idle_hold:
                q_dot_nom[2] = 0.0
            else:
                q_dot_nom[2] += 4.0 * (smooth_desired_z - gantry_qpos[2])

            rot_err = smooth_wrist_rot - gantry_qpos[3:6]
            q_dot_nom[3:6] += ROT_KP * rot_err - ROT_KD * gantry_qvel[3:6]
            q_dot_nom[3:6] = np.clip(q_dot_nom[3:6], -0.4, 0.4)

            if layer_b_output.get("updated", False) and layer_b_output.get("velocity_scales"):
                max_scale = max(layer_b_output["velocity_scales"].values(), default=1.0)
                q_dot_nom *= max_scale

            q_dot_nom = np.clip(q_dot_nom, -GANTRY_VEL_MAX, GANTRY_VEL_MAX)

            # --- LAYER A QP SOLVE ---
            if idle_hold:
                q_dot_cmd = np.zeros(n_gantry_dof, dtype=float)
                diag = {"status": "hold", "max_zone_slack": 0.0, "max_sep_slack": 0.0, "max_key_slack": 0.0}
            else:
                state = RobotState(
                    q=gantry_qpos,
                    q_dot=gantry_qvel,
                    J=None,
                    ee_pos=smooth_wrist_target,
                    ee_rot=np.eye(3),
                )
                q_dot_cmd, diag = servo.step(
                    state=state,
                    q_dot_nom=q_dot_nom,
                    key_constraints=key_constraints,
                    zone_constraints=zone_constraints,
                    sep_constraints=sep_constraints,
                )

            if diag.get("status") == "opt":
                global_max_zone = max(global_max_zone, diag.get("max_zone_slack", 0.0))
                global_max_sep = max(global_max_sep, diag.get("max_sep_slack", 0.0))
                global_max_key = max(global_max_key, diag.get("max_key_slack", 0.0))

            # --- APPLY GANTRY CONTROL (velocity -> position target) ---
            gantry_cmd = gantry_qpos + q_dot_cmd * dt
            if Y_LOCK:
                gantry_cmd[1] = base_y
            gantry_cmd[2] = smooth_desired_z
            gantry_cmd[3:6] = smooth_wrist_rot
            # Allow larger Z range: can drop up to Z_DOWN_RANGE below base, or rise 3cm above
            gantry_cmd[2] = float(np.clip(gantry_cmd[2], max(0.02, base_z - Z_DOWN_RANGE), base_z + 0.03))
            GANTRY_LIMITS = [
                (-0.5, 1.5), (-0.8, 0.5), (0.0, 0.6),  # Extended Y range to -0.8
                (-0.5, 0.5), (-0.8, 0.8), (-0.8, 0.8),
            ]
            if len(gantry_act_ids) == n_gantry_dof:
                for i in range(n_gantry_dof):
                    lo, hi = GANTRY_LIMITS[i]
                    val = float(np.clip(gantry_cmd[i], lo, hi))
                    data.ctrl[gantry_act_ids[i]] = val
                    if FORCE_GANTRY_POSE and (not GANTRY_TRANSLATION_ONLY or i < 3):
                        if Y_LOCK and i == 1:
                            val = base_y
                        data.qpos[gantry_qpos_ids[i]] = val
                        data.qvel[gantry_qvel_ids[i]] = 0.0

            # --- APPLY FINGER CONTROL (curl) ---
            primary_finger = None if use_shadow else (primary_sm.finger_idx if primary_sm is not None else None)
            # Enable finger actuators with appropriate gains
            # Shadow Hand needs stronger gains for position control
            for aid, meta in actuator_meta.items():
                f_idx, suffix = meta
                if use_shadow:
                    # Shadow Hand needs stronger position gains to achieve visible curl.
                    # Position actuator: force = kp * (ctrl - qpos) - kd * qvel
                    model.actuator_gainprm[aid, 0] = shadow_finger_kp
                    model.actuator_biasprm[aid, 1] = -shadow_finger_kp
                    model.actuator_biasprm[aid, 2] = -shadow_finger_kd
                    model.actuator_forcerange[aid] = np.array(
                        [-shadow_finger_force, shadow_finger_force], dtype=float
                    )
                    if act_ctrl_limited[aid]:
                        lo, hi = act_ctrl_ranges[aid]
                        data.ctrl[aid] = float(np.clip(data.ctrl[aid], lo, hi))
                elif suffix == "abd":
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
            for f_idx, curl in finger_targets_cmd.items():
                if primary_finger is not None and f_idx != primary_finger:
                    curl = 0.0
                # Smooth curl targets to avoid impulsive jumps (skip for active press)
                if curl >= 1.0:
                    smooth_finger_targets[f_idx] = curl
                else:
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
                    mode = finger_mode_map.get(f_idx, ContactMode.HOVER)
                    if use_shadow:
                        # Use data-driven curl direction; skip joints that don't move the tip down.
                        direction, _ = shadow_act_curl.get(act_id, (0, 0.0))
                        if direction == 0:
                            target = home
                        else:
                            if direction > 0:
                                span = act_ctrl_ranges[act_id][1] - home
                            else:
                                span = home - act_ctrl_ranges[act_id][0]
                            if smooth_finger_targets[f_idx] >= 1.0:
                                target = home + direction * span
                            else:
                                target = home + direction * smooth_finger_targets[f_idx] * FINGER_FLEX_RANGE * span
                    else:
                        if smooth_finger_targets[f_idx] >= 1.0:
                            target = act_ctrl_ranges[act_id][1]
                        else:
                            target = home + smooth_finger_targets[f_idx] * FINGER_FLEX_RANGE * (act_ctrl_ranges[act_id][1] - home)
                    if act_ctrl_limited[act_id]:
                        lo, hi = act_ctrl_ranges[act_id]
                        target = np.clip(target, lo, hi)
                        # Rate limit finger movements for stability
                        prev = prev_finger_ctrl.get(act_id, target)
                        if mode == ContactMode.STRIKE:
                            max_delta = 0.12  # Fast for striking keys
                        elif mode in (ContactMode.PRETOUCH, ContactMode.HOLD):
                            max_delta = 0.06
                        else:
                            max_delta = 0.02  # Slow for idle fingers
                        target = float(np.clip(target, prev - max_delta, prev + max_delta))
                        prev_finger_ctrl[act_id] = target
                        data.ctrl[act_id] = target
                        if debug_fingers and sim_t >= debug_next_t and f_idx in (2, 3):
                            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
                            qpos_val = None
                            if act_trn_type[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                                jid = act_trn_id[act_id, 0]
                                qpos_val = float(data.qpos[model.jnt_qposadr[jid]])
                            print(f"[t={sim_t:.3f}] finger {f_idx} {name} target={target:.3f} qpos={qpos_val} mode={finger_mode_map.get(f_idx)}")
                            debug_next_t = sim_t + 0.5

            # --- STEP PHYSICS ---
            mujoco.mj_step(model, data)
            if FORCE_GANTRY_POSE:
                for i in range(n_gantry_dof):
                    if GANTRY_TRANSLATION_ONLY and i >= 3:
                        continue
                    val = float(np.clip(gantry_cmd[i], GANTRY_LIMITS[i][0], GANTRY_LIMITS[i][1]))
                    if Y_LOCK and i == 1:
                        val = base_y
                    data.qpos[gantry_qpos_ids[i]] = val
                    data.qvel[gantry_qvel_ids[i]] = 0.0
                mujoco.mj_forward(model, data)
            sim_t += dt

            # Viewer update
            viewer.sync()

            # Maintain real-time
            time_until_next = step_start + dt - time.time()
            if time_until_next > 0:
                time.sleep(time_until_next)

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
