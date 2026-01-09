# scripts/debug_layout.py
import numpy as np

from src.planning.kinematics import HandKinematics
from src.planning.piano_geometry import PianoGeometry

hk = HandKinematics("assets/scene.xml")
pg = PianoGeometry()

# 1) Neutral pose
q0 = np.zeros(hk.n_dof)
pos, _, _ = hk.get_finger_state(q0, finger_idx=1)
print("Neutral fingertip (world, m):", pos)

# 2) Middle C-ish key (e.g. MIDI 60)
k = pg.get_key_location(60)
print("Key 60 center:", k)

# 3) Where your *wrist* target for that key is:
from src.planning.kinematics import HandKinematics
wrist_target = hk.solve_wrist_target(60, finger=1, is_left_hand=False)
print("Wrist target for MIDI 60:", wrist_target)

print("Δ fingertip -> key center (X,Y,Z):",
      pos - np.array([k.center_x, k.center_y, k.center_z]))
