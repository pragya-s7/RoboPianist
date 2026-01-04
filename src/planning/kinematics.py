import numpy as np
from typing import List, Dict
from src.planning.piano_geometry import PianoGeometry

class HandKinematics:
    def __init__(self):
        self.piano = PianoGeometry()

        # right hand topology (mm relative to wrist center)
        # simplified arc layout
        self.rh_offsets = {
            1: np.array([-40.0, -20.0, -20.0]), # thumb: left, in, down
            2: np.array([-10.0, 60.0, -20.0]), # index: center-left, out, down
            3: np.array([10.0, 70.0, -20.0]), # middle: center, out, down
            4: np.array([30.0, 60.0, -20.0]), # ring: center-right, out, down
            5: np.array([50.0, 40.0, -20.0]), # pinky: right, out, down
        }

    def _get_finger_offset(self, finger:int, is_left_hand:bool) -> np.ndarray:
        base = self.rh_offsets.get(finger, np.array([0.0, 0.0, 0.0])).copy()
        if is_left_hand:
            base[0] = -base[0]
        return base

    def solve_wrist_target(self, midi_pitch: int, finger: int, is_left_hand: bool) -> np.ndarray:
        '''
        inverse kinematics (lite): given target key and finger, where should wrist be?
        P_key = P_wrist + P_finger_offset => P_key - P_finger_offset
        '''
        # get key location (world frame)
        loc = self.piano.get_key_location(midi_pitch)
        p_key = np.array([loc.center_x, loc.center_y, loc.center_z])

        # get finger offset (hand frame)
        p_finger = self._get_finger_offset(finger, is_left_hand)

        # solve wrist
        # assume hand frmae aligns with world frame for MVP. in full version we optimize yaw as well
        p_wrist = p_key - p_finger

        # adjust height (hover v. strike), target pre-touch height of 20 mm above key
        p_wrist[2] += 20.0

        return p_wrist
    
    def generate_trajectory(self, annotated_events: List[Dict]) -> List[Dict]:
        ''' takes events with fingering and adds wrist target [x, y, z]'''
        processed = []

        for e in annotated_events:
            # create a clean copy
            new_e = e.copy()
            fingers = e.get('fingering', [])
            pitches = e.get('pitches', [])
            staff = e.get('staff', 1)
            is_lh = (staff == 2) # simple heuristic for MVP

            if not fingers or not pitches:
                new_e['wrist_target'] = None
                processed.append(new_e)
                continue
        
            # simple heuristic: center wrist based on "average" active finger
            wrist_votes = []
            for p, f in zip(pitches, fingers):
                w_pos = self.solve_wrist_target(p, f, is_lh)
                wrist_votes.append(w_pos)

            if wrist_votes:
                avg_wrist = np.mean(wrist_votes, axis=0)
                new_e['wrist_target'] = avg_wrist.tolist()
            else:
                new_e['wrist_target'] = None

            processed.append(new_e)
        return processed
    
