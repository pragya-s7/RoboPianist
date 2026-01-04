import numpy as np
import mujoco
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os

from src.planning.piano_geometry import PianoGeometry

@dataclass
class IKSolution:
    q: np.ndarray
    converged: bool
    error: float

class HandKinematics:
    '''
    mujoco based kinematics oracle
    calculates FK and Jacobians by setting robot state directly in a separate 'mjData' struct
    '''
    def __init__(self, model_path: str='assets/scene.xml', ee_body_name: str = "end_effector"):
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Creating dummy 7-DOF arm...")
            self._create_dummy_scene(model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.n_dof = self.model.nq
        
        # map logical finger indices to MuJoCo site names
        self.finger_site_names = {
            1: "tip_1",
            2: "tip_2",
            3: "tip_3",
            4: "tip_4",
            5: "tip_5"
        }

        # cache site IDs
        self.site_ids = {}
        for f_idx, name in self.finger_site_names.items():
            try:
                self.site_ids[f_idx] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            except Exception:
                print(f"Warning: Site {name} not found")
        
        # buffers for Jacboian calculation
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.piano = PianoGeometry()

    def get_finger_state(self, q: np.ndarray, finger_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''returns (pos, rot, J) for a SPECIFIC finger tip'''
        # teleport robot
        if len(q) != self.n_dof:
            q_safe = np.zeros(self.n_dof)
            q_safe[:min(len(q), self.n_dof)] = q[:min(len(q), self.n_dof)]
            self.data.qpos[:] = q_safe
        else:
            self.data.qpos[:] = q

        mujoco.mj_kinematics(self.model, self.data)

        # get site ID
        site_id = self.site_ids.get(finger_idx)
        if site_id is None:
            # fallback to last body if finger not found
            return np.zeros(3), np.eye(3), np.zeros((6, self.n_dof))
        
        # read data
        pos = self.data.site_xpos[site_id].copy()
        mat = self.data.site_xmat[site_id].reshape(3, 3).copy()

        # compute Jacobian
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, site_id)
        J = np.vstack([self.jacp, self.jacr])

        return pos, mat, J

    def solve_wrist_target(self, midi_pitch: int, finger: int, is_left_hand: bool) -> np.ndarray:
        # wit hspider hand, wrist target calculations need to be roughly aligned with key
        # IK handles precise finger placement, for now return eky loc
        loc = self.piano.get_key_location(midi_pitch)
        return np.array([loc.center_x, loc.center_y, loc.center_z + 20.0])
    
    def generate_trajectory(self, annotated_events: List[Dict]) -> List[Dict]:
        '''
        adds 'wrist_target' to events based on geometry
        real time execution will refine this using the state machine
        '''
        processed = []
        for e in annotated_events:
            new_e = e.copy()
            fingers = new_e.get('fingering', [])
            pitches = new_e.get('pitches', [])

            if fingers and pitches:
                # heuristic: target based on first note of chord
                wt = self.solve_wrist_target(pitches[0], fingers[0], new_e.get('staff')==2)
                new_e['wrist_target'] = wt.tolist()
            else:
                new_e['wrist_target'] = None

            processed.append(new_e)
        return processed