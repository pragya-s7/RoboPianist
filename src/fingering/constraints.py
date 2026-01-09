import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class FingerState:
    """Represents state of hand at a given moment."""
    pitches: Tuple[int, ...]  # MIDI pitches involved
    fingers: Tuple[int, ...]  # fingers used

    @property
    def center_pitch(self) -> float:
        """Estimate hand's center of mass based on active notes."""
        if not self.pitches:
            return 0.0
        return sum(self.pitches) / len(self.pitches)


class BiomechanicalHand:
    def __init__(self):
        # Physical geometry (semitones)
        self.max_reach = {
            (1, 2): 10,
            (1, 3): 14,
            (1, 4): 17,
            (1, 5): 19,  # octave + 5th
            (2, 3): 5,
            (2, 4): 8,
            (2, 5): 10,
            (3, 4): 4,
            (3, 5): 7,
            (4, 5): 4,
        }

        self.w_stretch = 1.5    # cost per semitone of uncomfortable stretch
        self.w_crossing = 3.0   # cost for thumb-under/finger-over
        self.w_black_thumb = 4.0  # cost for putting thumb on black key
        self.w_velocity = 0.5   # cost for moving hand center quickly
        self.w_same_finger = 10.0  # penalty for repeating same finger on diff note
        self.w_weak_finger = 0.2   # slight penalty for 4/5 to prefer strong fingers

        # Ideal pitch offsets (in semitones) relative to center for each finger
        self.finger_preferred_offset = {
            1: -4.0,
            2: -2.0,
            3: 0.0,
            4: 2.0,
            5: 4.0,
        }

        # Template and thumb-center penalties
        self.w_template = 0.15
        self.w_thumb_center = 0.8

    def is_black_key(self, pitch: int) -> bool:
        return (pitch % 12) in [1, 3, 6, 8, 10]

    def get_valid_configurations(self, pitches: List[int]) -> List[FingerState]:
        """
        Generate all biomechanically valid fingerings for a set of pitches.
        Handles chords by checking internal consistency (no intra-chord crossings,
        reach limits respected, fingers strictly increasing).
        """
        if not pitches:
            return [FingerState(tuple(), tuple())]

        pitches = sorted(pitches)
        valid_states: List[FingerState] = []

        # Recursive helper to assign fingers to notes
        def _assign(idx: int, current_fingers: List[int]) -> None:
            if idx == len(pitches):
                valid_states.append(
                    FingerState(tuple(pitches), tuple(current_fingers))
                )
                return

            start_f = current_fingers[-1] + 1 if current_fingers else 1
            for f in range(start_f, 6):  # fingers 1–5
                # Pruning: check reach with previous finger in this chord
                if current_fingers:
                    prev_f = current_fingers[-1]
                    prev_p = pitches[idx - 1]
                    curr_p = pitches[idx]

                    # Fingers must be ordered (crossings happen over time, not within chord)
                    if f <= prev_f:
                        continue

                    # Check span constraints
                    pair = (prev_f, f)
                    dist = abs(curr_p - prev_p)
                    limit = self.max_reach.get(pair, 12)
                    if dist > limit:
                        continue

                _assign(idx + 1, current_fingers + [f])

        _assign(0, [])
        return valid_states

    def static_cost(self, state: FingerState) -> float:
        """
        Cost of holding a specific shape (node cost).

        This is evaluated per chord and includes:
          - Black-key thumb penalty
          - Weak finger penalty
          - Template penalty (how closely each finger matches its ideal offset)
          - Thumb/5th near-center penalty

        NOTE: previous version double-counted chord-level terms by nesting loops;
        this version applies them exactly once per chord.
        """
        if not state.pitches:
            return 0.0

        cost = 0.0
        center = state.center_pitch

        # Per-note penalties
        for p, f in zip(state.pitches, state.fingers):
            # Black key thumb penalty
            if f == 1 and self.is_black_key(p):
                cost += self.w_black_thumb

            # Weak finger penalty
            if f in (4, 5):
                cost += self.w_weak_finger

        # Chord-level template penalty (for chords / polyphony)
        if len(state.pitches) >= 2:
            for p, f in zip(state.pitches, state.fingers):
                ideal = self.finger_preferred_offset.get(f, 0.0)
                actual = p - center
                cost += self.w_template * (actual - ideal) ** 2

        # Thumb / 5th near center penalty
        for p, f in zip(state.pitches, state.fingers):
            if f in (1, 5) and abs(p - center) <= 1:
                cost += self.w_thumb_center

        return cost

    def transition_cost(self, prev: FingerState, curr: FingerState, dt: float) -> float:
        """Cost of moving from state prev to curr over dt seconds."""
        cost = 0.0

        # Handle rests / first note
        if not prev.pitches or not curr.pitches:
            return 0.0

        # Hand displacement (velocity proxy): penalize moving center of hand too fast.
        shift = abs(curr.center_pitch - prev.center_pitch)
        if dt > 0.05:
            velocity = shift / max(dt, 1e-6)
            cost += velocity * self.w_velocity * 0.01
        else:
            cost += shift * 2.0

        # Finger-level transitions: compare single-note transitions (simplified)
        if len(prev.fingers) == 1 and len(curr.fingers) == 1:
            p1, f1 = prev.pitches[0], prev.fingers[0]
            p2, f2 = curr.pitches[0], curr.fingers[0]

            diff_p = p2 - p1
            span = abs(diff_p)

            # Same note, different finger (substitution)
            if diff_p == 0 and f1 != f2:
                cost += 1.0

            # Same finger on different pitch
            if f1 == f2 and span > 0:
                scale = 1.0 + 0.2 * span
                if dt < 0.25:
                    scale *= 1.5
                cost += self.w_same_finger * scale

            # Ascending line
            elif diff_p > 0:
                if f2 < f1:  # crossover
                    if f2 == 1:
                        if f1 == 3:
                            cost += 0.0  # standard scale tuck
                        elif f1 == 4:
                            cost += 0.5  # standard arpeggio tuck
                        elif f1 == 2:
                            cost += 1.0  # chromatic scale tuck
                        elif f1 == 5:  # very hard
                            cost += 3.0
                    else:  # non-thumb crossings: very bad
                        cost += self.w_crossing * 2.0

            # Descending line
            elif diff_p < 0:
                if f2 > f1:  # crossover
                    cost += self.w_crossing

        return cost
