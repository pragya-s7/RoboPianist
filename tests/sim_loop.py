# tests/sim_loop.py
import sys
import os
import json
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


def get_actuator_indices(model):
    """
    STRICT NAME CHECKING FOR GANTRY AND FINGERS

    Returns:
        gantry_ids: [int] actuator indices for tx, ty, tz, rx, ry, rz
        finger_map: {finger_idx: [actuator_ids]}
    """
    target_gantry = ["a_tx", "a_ty", "a_tz", "a_rx", "a_ry", "a_rz"]

    gantry_ids = []
    finger_map = {1: [], 2: [], 3: [], 4: [], 5: []}

    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not name:
            continue

        if name in target_gantry:
            gantry_ids.append(i)
        elif name.startswith("a_th"):
            finger_map[1].append(i)
        elif name.startswith("a_ff"):
            finger_map[2].append(i)
        elif name.startswith("a_mf"):
            finger_map[3].append(i)
        elif name.startswith("a_rf"):
            finger_map[4].append(i)
        elif name.startswith("a_lf"):
            finger_map[5].append(i)

    return gantry_ids, finger_map


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
) -> ConstraintBlock | None:
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
    events_raw = data_json.get("events", [])
    if not events_raw:
        raise RuntimeError(f"No 'events' field found in {json_file}")

    events = sorted(events_raw, key=lambda x: x["onset_sec"])

    xml_path = "assets/scene.xml"
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} missing.")
        return

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    piano_geo = PianoGeometry()

    gantry_act_ids, finger_act_map = get_actuator_indices(model)

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

    # For primary servo target we still use RH index finger (2)
    tip_site_id_primary = finger_site_ids.get(2, -1)
    if tip_site_id_primary < 0:
        raise RuntimeError("Primary site 'tip_2' not found. Check scene.xml fingertip site names.")

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    # --- INITIAL POSE ---
    mid_c = piano_geo.get_key_location(60)
    # x = key center, y = -0.22 (back), z = 0.15 (hover), rx = 0, ry = 0, rz = 0
    home_pose = np.array([mid_c.center_x, -0.22, 0.15, 0.0, 0.0, 0.0])

    mujoco.mj_resetData(model, data)

    # 1. Set Joint Positions (teleport robot to start)
    for i, q_idx in enumerate(gantry_qpos_ids):
        data.qpos[q_idx] = home_pose[i]

    # 2. Set Actuator Targets (hold robot at start)
    if len(gantry_act_ids) == 6:
        data.ctrl[gantry_act_ids] = home_pose

    mujoco.mj_forward(model, data)

    print(f"--- PLAYING (servo-driven gantry, Tier 1) : {json_file} ---")
    print("NOTE: For best results, generate this JSON via scripts/run_pipeline.py.\n"
          "      (Fingerings, wrist targets, and Layer C contacts are used.)\n")

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
        sim_t = 0.0
        event_idx = 0
        active_machines: list[KeyStateMachine] = []

        # Phase 1: ConstraintFactory knows about ndof (gantry only)
        constraint_factory = ConstraintFactory(ndof=n_gantry_dof)

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
                if e["onset_sec"] <= sim_t + 0.5:
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

                                # Infer timing parameters from contact spec
                                sm.approach_lead = max(0.01, strike - pretouch)
                                sm.hold_duration = max(0.01, hold_end - strike)
                                sm.release_duration = max(0.01, release_end - hold_end)

                                active_machines.append(sm)
                        else:
                            # Fallback: legacy behavior using fingering + onset_sec
                            fingers = e.get("fingering", [])
                            pitches = e.get("pitches", [])
                            if fingers and pitches:
                                for p, f in zip(pitches, fingers):
                                    active_machines.append(KeyStateMachine(p, e["onset_sec"], f))
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

            # --- SERVO (Layer A) FOR GANTRY USING PRIMARY CONTACT ---
            # Default: no motion if no active machine
            q_dot_nom = np.zeros(n_gantry_dof)
            zone_block = None
            key_block = None
            sep_block = None

            primary_sm = select_primary_machine(active_machines, sim_t)

            # Kinematics once per step
            mujoco.mj_kinematics(model, data)

            # --- Per-finger kinematics for separation constraints ---
            finger_kinematics: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for sm in active_machines:
                f_idx = sm.finger_idx
                site_id = finger_site_ids.get(f_idx, -1)
                if site_id < 0:
                    continue

                # Shared buffers for jacobian
                mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                J_full = np.vstack([jacp, jacr])  # 6 x nv
                gantry_dof_ids = gantry_qvel_ids
                J_active = J_full[:, gantry_dof_ids]
                J_pos = J_active[:3, :]  # 3 x n_dof

                pos = data.site_xpos[site_id].copy()
                finger_kinematics[f_idx] = (pos, J_pos)

            # --- Primary fingertip kinematics for target tracking ---
            J_pos_primary = None
            if primary_sm is not None:
                site_id = tip_site_id_primary
                mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                J_full_primary = np.vstack([jacp, jacr])  # 6 x nv
                gantry_dof_ids = gantry_qvel_ids
                J_active_primary = J_full_primary[:, gantry_dof_ids]
                J_pos_primary = J_active_primary[:3, :]

                curr_pos = data.site_xpos[site_id].copy()
            else:
                # If no primary, still need a pose for state
                curr_pos = data.site_xpos[tip_site_id_primary].copy()
                J_full_primary = np.vstack([jacp, jacr])  # stale, but unused

            ee_rot = data.site_xmat[tip_site_id_primary].reshape(3, 3).copy()

            # Build RobotState (gantry only)
            q = np.array([data.qpos[idx] for idx in gantry_qpos_ids])
            q_dot = np.array([data.qvel[idx] for idx in gantry_qvel_ids])
            state = RobotState(q=q, q_dot=q_dot, J=None, ee_pos=curr_pos, ee_rot=ee_rot)

            if primary_sm is not None and J_pos_primary is not None:
                # Geometry for the active pitch
                k_loc = piano_geo.get_key_location(primary_sm.pitch)

                # Key surface z = key top (we use body center z as proxy)
                key_surface_z = k_loc.center_z

                # Contact-mode-based z target and max allowed velocity
                z_target, z_vmax, _stiff = primary_sm.get_target(key_surface_z)

                # Heuristic Y offset (thumb vs other fingers)
                y_off = -0.18 if primary_sm.finger_idx == 1 else -0.22

                target_pos = np.array(
                    [k_loc.center_x, y_off, z_target], dtype=float
                )

                # --- Nominal EE velocity (resolved-rate control) ---
                pos_err = target_pos - curr_pos
                Kp = 30.0
                ee_vel_des = pos_err * Kp

                # Clamp vertical velocity according to state machine suggestion
                if z_vmax is not None:
                    ee_vel_des[2] = np.clip(ee_vel_des[2], -z_vmax, z_vmax)

                # Only use translational part for nominal velocity
                J_pos = J_pos_primary  # 3 x n_dof
                try:
                    # Least-squares solution: minimize || J_pos qdot - ee_vel_des ||
                    q_dot_nom, *_ = np.linalg.lstsq(J_pos, ee_vel_des, rcond=None)
                except np.linalg.LinAlgError:
                    q_dot_nom = np.zeros(n_gantry_dof)

                # --- Zone constraint: halfspace with two-point collocation ---
                # Halfspace normal along the line from current_pos to target_pos:
                d = target_pos - curr_pos
                dist = float(np.linalg.norm(d))
                if dist > 1e-6:
                    a = d / dist
                    # Require fingertip not to overshoot beyond target along a:
                    # aᵀ y <= aᵀ target_pos
                    b = float(a @ target_pos)
                    tau_list = [dt * 0.5, dt]
                    zone_block = constraint_factory.create_zone_constraint(
                        a=a,
                        b=b,
                        y=curr_pos,
                        J_pos=J_pos,
                        tau_list=tau_list,
                    )
                else:
                    zone_block = None

                # --- Key-success constraint (spec-correct) ---
                if primary_sm.mode == ContactMode.STRIKE:
                    # Remaining time to this note's onset (deadline)
                    dt_rem = primary_sm.onset - sim_t

                    # Current key depth for this pitch (meters) → normalize to k ∈ [0, 1]
                    key_depth_m = key_depths.get(primary_sm.pitch, 0.0)
                    k_norm = normalize_key_travel(key_depth_m, k_down_m=0.012)

                    # Fingertip z-row of the position Jacobian
                    z_row = J_pos[2, :]  # shape (ndof,)

                    key_block = constraint_factory.create_key_constraint(
                        z_jac_row=z_row,
                        dt_rem=dt_rem,
                        k_current=k_norm,
                        k80=0.8,
                    )
                else:
                    key_block = None

            # --- Separation constraints between active fingertips (Tier 1) ---
            if finger_kinematics:
                sep_block = _build_separation_block(
                    factory=constraint_factory,
                    finger_kinematics=finger_kinematics,
                    d_min=0.015,  # 1.5 cm minimum separation
                )
            else:
                sep_block = None

            # Solve QP for gantry velocities
            q_dot_cmd, diag = servo.step(
                state=state,
                q_dot_nom=q_dot_nom,
                key_constraints=key_block,
                zone_constraints=zone_block,
                guard_constraints=None,
                sep_constraints=sep_block,
            )

            # Aggregate diagnostics for failure attribution
            if diag.get("status") == "opt":
                global_max_zone = max(global_max_zone, diag.get("max_zone_slack", 0.0))
                global_max_sep = max(global_max_sep, diag.get("max_sep_slack", 0.0))
                global_max_key = max(global_max_key, diag.get("max_key_slack", 0.0))

            # --- APPLY GANTRY CONTROL ---
            # Convert q_dot_cmd (velocities) into position targets for position actuators.
            # Simple Euler integration on actuator targets.
            if len(gantry_act_ids) == n_gantry_dof:
                # base target = current actuator commands
                current_ctrl = data.ctrl[gantry_act_ids].copy()
                new_ctrl = current_ctrl + q_dot_cmd * dt
                data.ctrl[gantry_act_ids] = new_ctrl

            # --- APPLY FINGER CONTROL (curl) ---
            for f_idx, curl in finger_targets.items():
                acts = finger_act_map.get(f_idx, [])
                if len(acts) == 4:
                    # Simple mapping: proximal joints mostly drive curl
                    data.ctrl[acts[0]] = 0.0
                    data.ctrl[acts[1]] = curl * 1.2
                    data.ctrl[acts[2]] = curl * 1.2
                    data.ctrl[acts[3]] = curl * 0.8

            # --- STEP PHYSICS ---
            mujoco.mj_step(model, data)
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
