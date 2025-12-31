from dataclasses import dataclass

@dataclass
class HandState:
    # 5 fingers representing MIDI pitch held by each finger; none means free finger
    fingers: tuple

class HandConstraints:
    def __init__(self, hand_span_octaves=1.2):
        self.max_stretch = {
            (0, 1): 12, # 1-2
            (0, 2): 15, # 1-3
            (0, 3): 17, # 1-4
            (0, 4): 19, # 1-5 (thumb-pinky max reach)
            (1, 2): 5, # 2-3
            (1, 3): 7, # 2-4
            (1, 4): 10, # 2-5
            (2, 3): 4, # 3-4
            (2, 4): 7, # 3-5
            (3, 4): 5 # 4-5
        }

    def check_chord_feasible(self, pitches, finger_indices):
        # sort by pitch to match physical hand geometry
        pairs = sorted(zip(pitches, finger_indices), key=lambda x: x[0])
        sorted_pitches = [p for p, f in pairs]
        sorted_fingers = [f for p, f in pairs]

        for i in range(len(sorted_fingers) - 1):
            if sorted_fingers[i] >= sorted_fingers[i+1]:
                return False
            
        for i in range(len(pairs) - 1):
            p1, f1 = pairs[i]
            p2, f2 = pairs[i+1]
            distance = abs(p2 - p1)
            finger_pair = tuple(sorted((f1, f2)))
            limit = self.max_stretch.get(finger_pair, 12)
            if distance > limit:
                return False
            
        return True
    
    def cost_transition(self, prev_state, current_state, onset_delta):
        '''
        calculate cost of moving hand from A to B -- euclidean distance of active fingers + penalty for awk crossings
        '''
        cost = 0.0

        # legato vs. staccato logic for later !!MVP, needs full impl. later
        return cost
