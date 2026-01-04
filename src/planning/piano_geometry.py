import math
from dataclasses import dataclass

@dataclass
class KeyLocation:
    center_x: float # mm relative to middle c (MIDI 60)
    center_y: float ## mm (depth) - white keys ~0, black keys ~50
    center_z: float # mm (height) - usually 0
    is_black: bool

class PianoGeometry:
    def __init__(self):
        # standard piano dimensions
        self.octave_width = 164.0
        self.white_key_width = 23.6
        self.black_key_width = 13.7

        # offset pattern from C within an octave
        # C, D< E, F, G, A B
        self.white_indices = {0:0, 2:1, 4:2, 5:3, 7:4, 9:5, 11:6}

        # black key offsets relative to the left white key center
        # C#, D#, F#, G#, A#
        self.black_offsets= {1: 14.0, 3: 41.0, 6: 13.0, 8: 40.0, 10: 66.0}

    def get_key_location(self, midi_pitch: int) -> KeyLocation:
        # normalize to moddle C (60)
        rel_pitch = midi_pitch - 60
        octave = rel_pitch // 12
        note_in_octave = rel_pitch % 12

        # calculate octave offset
        base_x = octave * self.octave_width

        # calculate local offset
        if note_in_octave in self.white_indices:
            # white key
            idx = self.white_indices[note_in_octave]
            x = base_x + (idx * self.white_key_width)
            y = 0.0
            z = 0.0
            is_black = False
        else:
            # black key
            # simple heuristic: place based on white key structure
            if note_in_octave in [1, 3]: # C# or D#
                base_white = 0 # start at C
                offset_map = {1: 1, 3: 2}
                shift = 13.0 if note_in_octave == 1 else 43.0
                x = base_x + shift
            else: # F#, G#, A#
                shift = (3 * self.white_key_width) + (13.0 if note_in_octave == 6 else (42.0 if note_in_octave == 8 else 68.0))
                x = base_x + shift
            y = 45.0
            z = 12.0
            is_black = True

        return KeyLocation(x, y, z, is_black)
    
    