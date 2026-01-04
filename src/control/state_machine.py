from enum import Enum
import numpy as np
from typing import Tuple

class ContactMode(Enum):
    HOVER = 0
    PRETOUCH = 1
    STRIKE = 2
    HOLD = 3
    RELEASE = 4

class KeyStateMachine:
    def __init__(self, midi_pitch: int, onset_time: float, finger_idx: int):
        self.pitch = midi_pitch
        self.onset = onset_time
        self.finger_idx = finger_idx
        self.mode = ContactMode.HOVER
        self.done = False

        # timings
        self.t_pretouch = 0.15 # seconds before onset to start lowering
        self.t_hold = 0.1 # seconds to hold key down

    def update(self, t: float, key_travel: float):
        time_to_onset = self.onset - t

        # state transitions
        if self.mode == ContactMode.HOVER:
            if time_to_onset < self.t_pretouch:
                self.mode = ContactMode.PRETOUCH

        elif self.mode == ContactMode.PRETOUCH:
            #  commit to strike 20ms before
            if time_to_onset < 0.02:
                self.mode = ContactMode.STRIKE

        elif self.mode == ContactMode.STRIKE:
            # if key is pressed (> 8mm) OR we are past time
            if key_travel > 0.008 or time_to_onset < -0.05:
                self.mode = ContactMode.HOLD

        elif self.mode == ContactMode.HOLD:
            if t > (self.onset + self.t_hold):
                self.mode = ContactMode.RELEASE

        elif self.mode == ContactMode.RELEASE:
            if t > (self.onset + self.t_hold + 0.2):
                self.done = True
    
    def get_target(self, key_surface_z: float) -> Tuple[float, float, float]:
        '''returns (z_target, z_max_velocity, stiffness_weight)'''
        if self.mode == ContactMode.HOVER:
            # high hover (2cm up)
            return key_surface_z + 0.02, 1.0, 1.0

        elif self.mode == ContactMode.PRETOUCH:
            # low hover (5mm up), precise positioning
            return key_surface_z + 0.005, 0.5, 10.0
        
        elif self.mode == ContactMode.STRIKE:
            # target BELOW key (-1cm), high velocity allowed
            return key_surface_z - 0.015, 2.0, 100.0
        
        elif self.mode == ContactMode.HOLD:
            # hold key down (-5mm)
            return key_surface_z - 0.005, 0.1, 50.0
        
        elif self.mode == ContactMode.RELEASE:
            # lift up
            return key_surface_z + 0.02, 1.0, 1.0
        
        return key_surface_z + 0.05, 1.0, 1.0