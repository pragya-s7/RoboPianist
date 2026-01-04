import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class FingerState:
    '''represents state of hand at a givewn moment'''
    pitches: Tuple[int, ...] # MIDI pitches involved
    fingers: Tuple[int, ...] # fingers used

    @property
    def center_pitch(self) -> float:
        '''estimate hand's center of mass based on active notes'''
        if not self.pitches: return 0.0
        return sum(self.pitches) / len(self.pitches)

class BiomechanicalHand:
    def __init__(self):
        # physical geometry (semitones)
        self.max_reach = {
            (1, 2): 10,
            (1, 3): 14,
            (1, 4): 17,
            (1, 5): 19, # octave + 5th
            (2, 3): 5,
            (2, 4): 8,
            (2, 5): 10,
            (3, 4): 4,
            (3, 5): 7,
            (4, 5): 4
        }
    
        self.w_stretch = 1.5 # cost per semitone of uncomfortable stretch
        self.w_crossing = 3.0 # cost for thumb-under/finger-over
        self.w_black_thumb = 4.0 # cost for putting thumb on black key
        self.w_velocity = 0.5 # cost for moving hand center quickly
        self.w_same_finger = 10.0 # penalty for repeating same finger on diff note
        self.w_weak_finger = 0.2 # slight penalty for 4/5 to prefer strong fingers

    def is_black_key (self, pitch: int) -> bool:
        return (pitch % 12) in [1, 3, 6, 8, 10]
    
    def get_valid_configurations(self, pitches: List[int]) -> List[FingerState]:
        '''
        generator that returns all biomechanically valid fingerings for set of pitches
        handles chords by checking internal consistency
        '''
        if not pitches:
            return [FingerState(tuple(), tuple())]
        pitches = sorted(pitches)
        valid_states = []

        # recursive helper to assign fingers to notes
        def _assign(idx, current_fingers):
            if idx == len(pitches):
                valid_states.append(FingerState (tuple(pitches), tuple(current_fingers)))
                return
            
            start_f = current_fingers[-1] + 1 if current_fingers else 1
            for f in range(start_f, 6): # fingers 1- 5
                # pruning: check reach with prev. finger in this chord
                if current_fingers:
                    prev_f = current_fingers[-1]
                    prev_p = pitches[idx-1]
                    curr_p = pitches[idx]

                    # fingers must be ordered (crossings happen over time, not within chord)
                    if f <= prev_f: continue

                    # check span constraints
                    pair = (prev_f, f)
                    dist = abs(curr_p - prev_p)
                    limit = self.max_reach.get(pair, 12)
                    if dist > limit: continue

                _assign(idx + 1, current_fingers + [f])
        
        _assign(0, [])
        return valid_states
    
    def static_cost(self, state: FingerState) -> float:
        '''cost of holding a specific shape (node cost)'''
        cost = 0.0
        for p, f in zip(state.pitches, state.fingers):
            # black key thumb penalty
            if f == 1 and self.is_black_key(p):
                # only penalize if its not the only way (static here, but usually context dependent)
                cost += self.w_black_thumb
            
            # weak finger penalty
            if f in [4, 5]:
                 cost += self.w_weak_finger

        return cost

    def transition_cost(self, prev: FingerState, curr: FingerState, dt: float) -> float:
        ''' cost of moving from atate prev to to curr over dt seconds'''
        cost = 0.0

        # handle rests/ firsst note
        if not prev.pitches or not curr.pitches:
            return 0.0
        
        # hand displacement (velocity proxy): penalize moving center of hand too fast
        # if dt is large, jumps are cheap, if dt is smalll, jumps expensive
        shift = abs(curr.center_pitch - prev.center_pitch)
        if dt > 0.05:
            velocity = shift / dt
            cost += velocity * self.w_velocity * 0.01
        else: 
            cost += shift * 2.0 

        # finger level transitions: compare melody line or highest notes for crossings
        # simplified for polyphony: compare avg or bounding box
        # find pivot fingers
        # if pitch goes up but we switch from higher finger to lower --> crossing
        if len(prev.fingers) == 1 and len(curr.fingers) == 1:
            p1, f1 = prev.pitches[0], prev.fingers[0]
            p2, f2 = curr.pitches[0], curr.fingers[0]

            diff_p = p2 - p1

            # same note, diff finger (substitution) --> good for repeating notes, bad for holds
            if diff_p == 0 and f1 != f2:
                cost += 1.0

            # same note, same finger -> fast repeat
            elif diff_p == 0 and f1 == f2:
                if dt < 0.15: cost += self.w_same_finger

            # ascending
            elif diff_p > 0:
                if f2 < f1: # crossover
                    if f2 == 1:
                        if f1 == 3:
                            cost += 0.0 # standard scale tuck
                        elif f1 == 4:
                            cost += 0.5 # standard arpeggio tuck
                        elif f1 == 2:
                            cost += 1.0 # chromatic scale tuck , okay but tight
                        elif f1 == 5: # very hard
                            cost += 3.0
                    else: # non thumb crossings --> generally very bad
                        cost += self.w_crossing * 2.0

                        
            # descednding
            elif diff_p < 0:
                if f2 > f1: # crossover 
                    cost += self.w_crossing
        
        return cost