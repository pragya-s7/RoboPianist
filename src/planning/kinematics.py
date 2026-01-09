import numpy as np
import mujoco
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os

from src.planning.piano_geometry import PianoGeometry, KeyLocation

class HandKinematics:
    def __init__(self, model_path: str='assets/scene.xml'):
        if not os.path.exists(model_path): 
            # Fallback for relative paths
            model_path = os.path.join(os.getcwd(), 'assets/scene.xml')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.n_dof = self.model.nq
        self.piano = PianoGeometry()
        self.finger_site_names = {1: "tip_1", 2: "tip_2", 3: "tip_3", 4: "tip_4", 5: "tip_5"}
        self.site_ids = {}
        for f_idx, name in self.finger_site_names.items():
            self.site_ids[f_idx] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

    def get_finger_state(self, q: np.ndarray, finger_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(q) != self.n_dof:
            q_safe = np.zeros(self.n_dof)
            q_safe[:min(len(q), self.n_dof)] = q[:min(len(q), self.n_dof)]
            self.data.qpos[:] = q_safe
        else:
            self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        site_id = self.site_ids.get(finger_idx)
        if site_id is None or site_id < 0: return np.zeros(3), np.eye(3), np.zeros((6, self.n_dof))
        pos = self.data.site_xpos[site_id].copy()
        mat = self.data.site_xmat[site_id].reshape(3, 3).copy()
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, site_id)
        J = np.vstack([self.jacp, self.jacr])
        return pos, mat, J

    def solve_wrist_target(self, midi_pitch: int, finger: int, is_left_hand: bool) -> np.ndarray:
        k_loc = self.piano.get_key_location(midi_pitch)
        x = k_loc.center_x
        # Heuristics for Gantry base position
        y = -0.18 if finger == 1 else -0.22
        z = 0.12 
        return np.array([x, y, z])

    def generate_trajectory(self, annotated_events: List[Dict]) -> List[Dict]:
        processed = []
        for e in annotated_events:
            new_e = e.copy()
            fingers = new_e.get('fingering', [])
            pitches = new_e.get('pitches', [])
            if fingers and pitches:
                wt = self.solve_wrist_target(pitches[0], fingers[0], is_left_hand=(new_e.get('staff') == 2))
                new_e['wrist_target'] = wt.tolist()
            else:
                new_e['wrist_target'] = None
            processed.append(new_e)
        return processed
