from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from src.planning.piano_geometry import PianoGeometry


@dataclass
class ContactTiming:
    """timing for a single key press by a given finger"""
    pretouch_start_sec: float # when to start lowering toward the key
    strike_time_sec: float # nominal onset time of the note
    hold_end_sec: float # when to end "HOLD" state
    release_end_sec: float # when to be fully back in HOVER


@dataclass
class ContactSpec:
    """
    full spec of one finger-key contact
    """
    pitch: int
    finger: int
    staff: int
    onset_sec: float
    velocity: int

    # geometry
    key_center: List[float] # x, y, z
    is_black: bool

    # contact mode schedule
    pretouch_start_sec: float
    strike_time_sec: float
    hold_end_sec: float
    release_end_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pitch": int(self.pitch),
            "finger": int(self.finger),
            "staff": int(self.staff),
            "onset_sec": float(self.onset_sec),
            "velocity": int(self.velocity),
            "key_center": [float(c) for c in self.key_center],
            "is_black": bool(self.is_black),
            "pretouch_start_sec": float(self.pretouch_start_sec),
            "strike_time_sec": float(self.strike_time_sec),
            "hold_end_sec": float(self.hold_end_sec),
            "release_end_sec": float(self.release_end_sec),
        }


class LayerCPlanner:
    """
    CHORD Layer C:
      - takes symbolic + timed events w/ fingerings
      - adds per-finger contact schedules and exact key geometry

    Input event (per `run_pipeline`):
      {
        'onset_sec': float,
        'staff': int,
        'pitches': [int, ...],
        'fingering': [int, ...],
        'notes': [...],
        ... (other fields are passed through untouched)
      }

    Output event:
      same dict + new field:
        'contacts': [ContactSpec_dict, ...]
    """

    def __init__(
        self,
        t_pretouch: float = 0.15,
        t_hold: float = 0.10,
        t_release: float = 0.20,
    ) -> None:
        """
        t_pretouch: seconds before onset to begin PRETOUCH.
        t_hold:     duration to stay in HOLD after strike.
        t_release:  duration from end of HOLD until fully RELEASED.
        """
        self.t_pretouch = float(t_pretouch)
        self.t_hold = float(t_hold)
        self.t_release = float(t_release)
        self.piano = PianoGeometry()

    # --------- public API ---------

    def annotate_events(self, annotated_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point for Layer C.

        annotated_events: events that already have:
          - pitches: List[int]
          - staff: int
          - onset_sec: float
          - fingering: List[int]
        Returns a *new* list of dicts, with 'contacts' attached.
        """
        out: List[Dict[str, Any]] = []
        for e in annotated_events:
            e_new = dict(e)  # shallow copy, we never mutate the input

            contacts = self._build_contacts_for_event(e_new)
            e_new["contacts"] = [c.to_dict() for c in contacts]

            out.append(e_new)
        return out

    # --------- internal helpers ---------

    def _build_contacts_for_event(self, event: Dict[str, Any]) -> List[ContactSpec]:
        pitches = event.get("pitches", [])
        fingers = event.get("fingering", [])
        staff = int(event.get("staff", 1))
        onset_sec = float(event.get("onset_sec", 0.0))
        notes_meta = event.get("notes", [])

        if not pitches or not fingers or len(pitches) != len(fingers):
            return []

        # Build a pitch → velocity map from notes[]
        vel_map: Dict[int, int] = {}
        for note in notes_meta:
            p = int(note.get("pitch_midi"))
            v = int(note.get("velocity", 64))
            vel_map[p] = v

        contacts: List[ContactSpec] = []

        for pitch, finger in zip(pitches, fingers):
            pitch = int(pitch)
            finger = int(finger)

            k_loc = self.piano.get_key_location(pitch)

            pretouch_start = max(0.0, onset_sec - self.t_pretouch)
            strike_time = onset_sec
            hold_end = onset_sec + self.t_hold
            release_end = hold_end + self.t_release

            vel = vel_map.get(pitch, 64)

            spec = ContactSpec(
                pitch=pitch,
                finger=finger,
                staff=staff,
                onset_sec=onset_sec,
                velocity=vel,
                key_center=[k_loc.center_x, k_loc.center_y, k_loc.center_z],
                is_black=k_loc.is_black,
                pretouch_start_sec=pretouch_start,
                strike_time_sec=strike_time,
                hold_end_sec=hold_end,
                release_end_sec=release_end,
            )
            contacts.append(spec)

        return contacts
