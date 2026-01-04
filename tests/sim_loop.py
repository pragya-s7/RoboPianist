import sys
import os
import json
import numpy as np
import time
import mujoco
import mujoco.viewer
from scipy import sparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.control.servo import ChordServo, RobotState, ConstraintBlock
from src.control.constraint_factory import ConstraintFactory
from src.planning.kinematics import HandKinematics
from src.planning.piano_geometry import PianoGeometry
from src.control.state_machine import KeyStateMachine, ContactMode

def get_piano_state(model, data):
    states = {}
    for midi in range(21, 109):
        addr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"k{midi}")
        if addr >= 0:
            states[midi] = data.qpos[model.jnt_qposadr[addr]]
    return states

def get_robot_config(model):
    """ Separates Arm indices from Finger indices """
    arm_q_idx = []
    finger_q_idx = []
    arm_act_idx = []
    finger_act_idx = []

    for i in range(model.nq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name.startswith('j'):
            if name.startswith('jf'): # Finger joint
                finger_q_idx.append(model.jnt_qposadr[i])
            else: # Arm joint
                arm_q_idx.append(model.jnt_qposadr[i])

    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name.startswith('vf'): # Finger (velocity) or pf (position)
            finger_act_idx.append(i)
        else:
            arm_act_idx.append(i)
            
    return np.array(arm_q_idx), np.array(finger_q_idx), np.array(arm_act_idx), np.array(finger_act_idx)

def run_sim(json_file):
    with open(json_file, 'r') as f:
        events = json.load(f)['events']
    
    kinematics = HandKinematics(model_path="assets/scene.xml") 
    piano_geo = PianoGeometry()
    model = kinematics.model
    data = kinematics.data

    # Config
    arm_q, finger_q, arm_act, finger_act = get_robot_config(model)
    n_arm = len(arm_q)
    print(f"Arm DOFs: {n_arm}, Finger DOFs: {len(finger_q)}")
    
    # Servo controls ARM ONLY
    servo = ChordServo(n_dof=n_arm)

    # --- HOME POSE ---
    # Arm: slightly bent
    data.qpos[arm_q[1]] = 0.3  # Lift shoulder
    data.qpos[arm_q[2]] = 1.2  # Bend elbow
    data.qpos[arm_q[4]] = 0.5  # Wrist Pitch down
    
    # Fingers: Straight (0.0)
    for idx in finger_q: data.qpos[idx] = 0.0
        
    mujoco.mj_forward(model, data) 

    active_machines = [] 
    next_event_idx = 0
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "debug_target")
    
    print("Launching Viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_t = 0.0
        dt = 0.002
        
        while viewer.is_running():
            loop_start = time.time()
            
            # Read State
            q_all = data.qpos[:]
            q_vel_all = data.qvel[:]
            q_arm = q_all[arm_q]
            q_dot_arm = q_vel_all[arm_q]
            
            key_states = get_piano_state(model, data)

            # Spawn Events
            while next_event_idx < len(events):
                e = events[next_event_idx]
                if e['onset_sec'] < sim_t + 0.5:
                    fingers = e['fingering']
                    pitches = e['pitches']
                    if fingers and len(fingers) == len(pitches):
                        for p, f in zip(pitches, fingers):
                            if e['staff'] == 1: 
                                active_machines.append(KeyStateMachine(p, e['onset_sec'], f))
                    next_event_idx += 1
                else:
                    break

            machines_to_keep = []
            constraint_blocks = []
            
            # Finger Commands (Default: Open)
            finger_cmds = np.zeros(len(finger_act)) 
            
            # Visual Target
            final_target_viz = np.array([0.636, -0.15, 0.25])

            if not active_machines:
                # Hover Home: Use finger 3 (Middle)
                # Need Jacobian of Finger 3 w.r.t Arm Joints
                ee_pos, _, J_full = kinematics.get_finger_state(q_all, 3) 
                J_arm = J_full[:, arm_q] 
                zones = ConstraintFactory.create_zone_constraint(J_arm, ee_pos, final_target_viz, radius=0.05)
                constraint_blocks.append(zones)

            for sm in active_machines:
                k_travel = key_states.get(sm.pitch, 0.0)
                sm.update(sim_t, k_travel)

                if not sm.done:
                    machines_to_keep.append(sm)
                    
                    loc = piano_geo.get_key_location(sm.pitch)
                    key_x = (loc.center_x / 1000.0) + 0.636 
                    key_y = (loc.center_y / 1000.0) - 0.1 
                    key_base_z = (loc.center_z / 1000.0) 

                    # State Machine Targets
                    z_target, _, _ = sm.get_target(key_base_z)
                    target_pos = np.array([key_x, key_y, z_target])
                    final_target_viz = target_pos

                    # Finger Control Logic
                    # If STRIKE/HOLD, curl finger (positive angle)
                    f_idx_local = sm.finger_idx - 1 # 1-based to 0-based
                    if 0 <= f_idx_local < len(finger_cmds):
                        if sm.mode in [ContactMode.STRIKE, ContactMode.HOLD]:
                            finger_cmds[f_idx_local] = 0.8 # Curl down
                        else:
                            finger_cmds[f_idx_local] = 0.0 # Straight

                    # Arm Kinematics
                    f_pos, _, J_full = kinematics.get_finger_state(q_all, sm.finger_idx)
                    J_arm = J_full[:, arm_q]

                    # Relaxed radius for arm, fingers do the precision work
                    rad = 0.02 
                    zones = ConstraintFactory.create_zone_constraint(J_arm, f_pos, target_pos, radius=rad)
                    constraint_blocks.append(zones)
            
            active_machines = machines_to_keep

            # Update Debug Viz
            if mocap_id >= 0: data.mocap_pos[0] = final_target_viz

            # Solve Arm QP
            active_zone = None
            if constraint_blocks:
                full_A = sparse.vstack([b.A for b in constraint_blocks])
                full_l = np.concatenate([b.l for b in constraint_blocks])
                full_u = np.concatenate([b.u for b in constraint_blocks])
                active_zone = ConstraintBlock(full_A, full_l, full_u)

            r_state = RobotState(q=q_arm, q_dot=q_dot_arm, J=np.zeros((6, n_arm)), ee_pos=np.zeros(3), ee_rot=np.eye(3))
            
            # Arm Homing (Soft)
            q_target_arm = np.zeros(n_arm)
            q_target_arm[1] = 0.3
            q_target_arm[2] = 1.2
            q_target_arm[4] = 0.5
            
            q_dot_nom = -2.0 * (q_arm - q_target_arm) - 0.5 * q_dot_arm
            q_cmd, diag = servo.step(r_state, q_dot_nom=q_dot_nom, zone_constraints=active_zone)
            
            if np.any(np.isnan(q_cmd)) or np.max(np.abs(q_cmd)) > 20.0:
                q_cmd = np.zeros_like(q_cmd)
            q_cmd = np.clip(q_cmd, -5.0, 5.0)

            # Apply Controls
            data.ctrl[arm_act] = q_cmd       # Velocity for Arm
            data.ctrl[finger_act] = finger_cmds # Position for Fingers

            mujoco.mj_step(model, data)
            sim_t += dt
            viewer.sync()
            
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mjpython tests/sim_loop.py <json_path>")
    else:
        run_sim(sys.argv[1])