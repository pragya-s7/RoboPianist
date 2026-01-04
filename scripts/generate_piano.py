# scripts/generate_piano.py
# Run this once to generate assets/piano.xml
import numpy as np

header = """
<mujoco model="piano_88">
  <default>
    <geom rgba="0.9 0.9 0.9 1" type="box"/>
    <!-- Keys are slide joints with spring return -->
    <joint type="slide" axis="0 0 -1" limited="true" range="0 0.012" damping="0.5" stiffness="100" armature="0.001"/>
  </default>
  <worldbody>
"""

footer = """
  </worldbody>
</mujoco>
"""

def generate():
    xml = header
    # Standard piano geometry
    white_width = 0.0236
    octave_width = 0.164
    
    # Offsets for 12 notes in octave relative to C
    # W=White, B=Black. 
    # C, C#, D, D#, E, F, F#, G, G#, A, A#, B
    is_white = [True, False, True, False, True, True, False, True, False, True, False, True]
    # Local offsets approx (in meters)
    offsets = [0, 0.014, 0.024, 0.041, 0.048, 0.072, 0.083, 0.096, 0.108, 0.120, 0.133, 0.144]

    # Start at A0 (MIDI 21). A0 is the 9th note in the octave sequence (index 9)
    # MIDI 21 is A0.
    start_offset = -0.02 # Shift piano to center
    
    for midi in range(21, 109):
        rel_note = (midi - 21 + 9) % 12
        octave = (midi - 21 + 9) // 12
        base_x = (octave * octave_width) + offsets[rel_note] + start_offset
        
        # Adjust for first partial octave (A0, A#0, B0) correction
        # (Simplified geometry for MVP)
        
        if is_white[rel_note]:
            # White Key
            xml += f"""
            <body name="key_{midi}" pos="{base_x} -0.1 0">
                <joint name="k{midi}"/>
                <geom size="0.011 0.075 0.01" pos="0 0 0"/>
                <site name="s_{midi}" pos="0 -0.06 0.01" size="0.005" rgba="1 0 0 0.5"/>
            </body>"""
        else:
            # Black Key
            xml += f"""
            <body name="key_{midi}" pos="{base_x} -0.05 0.02">
                <joint name="k{midi}"/>
                <geom size="0.006 0.05 0.01" rgba="0.1 0.1 0.1 1"/>
                <site name="s_{midi}" pos="0 -0.04 0.01" size="0.005" rgba="1 0 0 0.5"/>
            </body>"""

    xml += footer
    with open("assets/piano.xml", "w") as f:
        f.write(xml)
    print("Generated assets/piano.xml")

if __name__ == "__main__":
    generate()