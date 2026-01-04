# FILE: tests/sim_loop.py
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.control.servo import ChordServo, RobotState

def run_sim(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract Right Hand Targets
    targets = []
    times = []
    for e in data['events']:
        if e['staff'] == 1 and e['wrist_target']:
            targets.append(e['wrist_target'])
            times.append(e['onset_sec'])
            
    if not targets:
        print("No Right Hand targets found.")
        return

    # Simulation Setup
    servo = ChordServo(n_dof=3) # Simplified 3-DOF robot (x,y,z slide)
    
    # Initial State
    q = np.array(targets[0]) # Start exactly at target
    dt = 0.001
    max_time = times[-1] + 1.0
    
    sim_t = []
    sim_pos = []
    target_idx = 0
    
    current_time = times[0]
    
    print(f"Simulating {len(targets)} targets over {max_time:.1f}s...")
    
    while current_time < max_time:
        # 1. Get current target (Hold last target if finished)
        while target_idx < len(times) - 1 and times[target_idx+1] < current_time:
            target_idx += 1
        
        target = np.array(targets[target_idx])
        
        # 2. Mock Jacobian (Identity for a Cartesian slider robot)
        # In a real arm, this comes from URDF
        J = np.eye(3) 
        
        # 3. Step Servo
        state = RobotState(q=q, q_dot=np.zeros(3), ee_pose=q)
        q_dot = servo.step(state, target, J)
        
        # 4. Integrate Physics
        q = q + q_dot * dt
        current_time += dt
        
        # Log
        if int(current_time * 1000) % 10 == 0: # Downsample
            sim_t.append(current_time)
            sim_pos.append(q.copy())

    # Plot
    sim_pos = np.array(sim_pos)
    target_x = [t[0] for t in targets]
    target_time = times
    
    plt.figure(figsize=(12, 6))
    plt.plot(target_time, target_x, 'ro', markersize=2, label='Planner Targets')
    plt.plot(sim_t, sim_pos[:, 0], 'b-', linewidth=1, label='Servo Trajectory')
    plt.title('Layer A Simulation: Servo Tracking')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (mm)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_sim(sys.argv[1])