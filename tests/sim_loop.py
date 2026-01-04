import sys
import os
import json
import numpy as np
import time
import mujoco
import mujoco.viewer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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

def run_sim(json_file):
    with open(json_file, 'r') as f:
        events = json.load(f)['events']
    
    kinematics = HandKinematics(model_path="assets/scene.xml") 
    piano_geo = PianoGeometry()
    model = kinematics.model
    data = kinematics.data

    # Map Actuators
    arm_act_ids = []
    finger_act_ids = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name.startswith('p') and not name.startswith('pf'):
            arm_act_ids.append(i)
        elif name.startswith('pf'):
            finger_act_ids.append(i)

    # --- INITIAL POSE ---
    mujoco.mj_resetData(model, data)
    
    # Ready Pose (Pitch joints aligned)
    # j1 (pan): 0
    # j2 (shoulder): -0.5 (lean forward)
    # j3 (elbow): -1.0 (bend forward)
    # j4 (forearm): 0
    # j5 (wrist pitch): 0
    # j6 (wrist yaw): 0
    target_q = np.array([0, -0.5, -1.0, 0, 0, 0])
    
    q_arm_indices = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j{i+1}")] for i in range(6)]
    for i, q_idx in enumerate(q_arm_indices):
        data.qpos[q_idx] = target_q[i]

    # Settle physics
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
            key_states = get_piano_state(model, data)

            # --- EVENTS ---
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
            finger_ctrl = np.zeros(len(finger_act_ids))
            target_sum = np.zeros(3)
            active_count = 0

            for sm in active_machines:
                k_travel = key_states.get(sm.pitch, 0.0)
                sm.update(sim_t, k_travel)

                if not sm.done:
                    machines_to_keep.append(sm)
                    loc = piano_geo.get_key_location(sm.pitch)
                    # Target world coords
                    t_pos = np.array([(loc.center_x / 1000.0) + 0.636, (loc.center_y / 1000.0) - 0.1, 0.1])
                    target_sum += t_pos
                    active_count += 1
                    
                    # Finger Curl
                    curl = 0.0
                    if sm.mode in [ContactMode.STRIKE, ContactMode.HOLD]: curl = 0.9
                    elif sm.mode == ContactMode.PRETOUCH: curl = 0.3
                    
                    prefix = f"pf{sm.finger_idx}"
                    for i, aid in enumerate(finger_act_ids):
                        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                        if aname.startswith(prefix):
                            finger_ctrl[i] = curl
            
            active_machines = machines_to_keep

            # --- ARM CONTROL ---
            if active_count > 0:
                avg_target = target_sum / active_count
                if mocap_id >= 0: data.mocap_pos[0] = avg_target
                
                # Heuristic IK:
                # 1. Rotate Waist (j1) to match X
                dx = avg_target[0] - 0.63
                dy = avg_target[1] - (-0.6)
                target_q[0] = np.arctan2(dx, dy)
                
                # 2. Reach Forward (Shoulder j2 + Elbow j3)
                # Simple look-up table logic: further keys = straighter arm
                # dist = np.sqrt(dx**2 + dy**2)
                # For now, keep fixed height to test stability
                target_q[1] = -0.4
                target_q[2] = -0.8
            else:
                # Home
                target_q[0] = 0.0
                target_q[1] = -0.5
                target_q[2] = -1.0

            data.ctrl[arm_act_ids] = target_q
            data.ctrl[finger_act_ids] = finger_ctrl

            mujoco.mj_step(model, data)
            sim_t += dt
            
            if int(sim_t / dt) % 20 == 0:
                viewer.sync()
            
            elapsed = time.time() - loop_start
            if elapsed < dt:
                pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mjpython tests/sim_loop.py <json_path>")
    else:
        run_sim(sys.argv[1])