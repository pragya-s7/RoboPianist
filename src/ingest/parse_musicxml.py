from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
import bisect

import music21 as m21

@dataclass
class NoteInfo:
    pitch_midi: int
    onset: float # symbolic (quarterlengths)
    onset_sec: float # real-time (seconds)
    duration: float # symbolic
    duration_sec: float #real- time
    velocity: int # MIDI velocity (0 - 127)
    staff: int
    voice: Optional[str]
    measure: Optional[int]
    part_idx: int
    note_id: str # debug id

@dataclass
class Event:
    onset: float
    onset_sec: float 
    staff: int
    event_type: str # either "note" or "chord"
    pitches: List[int]
    held_pitches: List[int] # notes started alr that are not yet finished
    notes: List[Dict[str, Any]]

def _build_velocity_map(part: m21.stream.Part) -> List[Tuple[float, int]]:
    '''
    scans Part for dyanmic objects (p, f, ff) returns sorted list of (offset, midi_velocity)
    '''
    dynamic_map = {'ppp': 16, 'pp': 33, 'p': 49, 'mp': 64, 'mf': 80, 'f': 97, 'ff': 112, 'fff': 126}
    dynamics_list = []
    
    for el in part.recurse():
        if isinstance(el, m21.dynamics.Dynamic):
            vel = dynamic_map.get(el.value, 64)
            try:
                offset = float(el.getOffsetInHierarchy(part))
                dynamics_list.append((offset, int(vel)))
            except:
                continue

    dynamics_list.sort(key=lambda x: x[0])
    if not dynamics_list or dynamics_list[0][0] > 0.0:
        dynamics_list.insert(0, (0.0, 64))
    return dynamics_list

def _get_velocity_at_offset(offset: float, velocity_map: List[Tuple[float, int]]) -> int:
    '''
    binary search to find active velocity 
    '''
    idx = bisect.bisect_right(velocity_map, (offset, 128)) - 1
    if idx < 0:
        return 64
    return velocity_map[idx][1]

def _safe_measure_number(n: m21.base.Music21Object) -> Optional[int]:
    try:
        return n.measureNumber
    except:
        return None
    
def _infer_staff(n: Any, pitch_midi: int) -> int:
    ''' 
    MVP staff inference:
    1) if music21 exposes n.staff, we use that value
    2) else if pitch >= middle C --> staff 1. else staff 2. 
    this is current mvp implementation, can always update for later @MVP
    '''
    if hasattr(n, "staff") and n.staff is not None:
        try:
            s = int(n.staff)
            if s in (1, 2):
                return s
        except Exception:
            print("NOTE: no staff from music21")
            pass
    return 1 if pitch_midi >= 60 else 2

def _is_attack(n) -> bool:
    if getattr(n, "tie", None) is None:
        return True
    return n.tie.type == 'start'


def extract_note_table(score: m21.stream.Score) -> List[NoteInfo]:
    note_infos: List[NoteInfo] = []

    for p_idx, part in enumerate(score.parts):
        vel_map = _build_velocity_map(part)
        flat_part = part.flatten()
        for n in flat_part.notes:
            onset = float(n.offset)
            measure_num = _safe_measure_number(n)
            try:
                onset_sec = float(n.seconds)
                dur_sec = float(n.duration.seconds)
            except: # fallback to 120 bpm
                onset_sec = onset * 0.5
                dur_sec = float(n.duration.quarterLength) * 0.5
            # dynamic lookup
            if n.volume.velocity is not None:
                base_velocity = int(n.volume.velocity)
            else:
                base_velocity = _get_velocity_at_offset(onset, vel_map)
            # expand chords to individual NoteInfo rows
            if isinstance(n, m21.chord.Chord):
                chord_dur = float(n.duration.quarterLength)
                voice = getattr(n, "voice", None)
                for j, p in enumerate(n.pitches):
                    midi = int(p.midi)
                    staff = _infer_staff(n, midi)
                    note_infos.append(NoteInfo(
                        pitch_midi=midi,
                        onset=onset,
                        onset_sec=onset_sec,
                        duration=chord_dur,
                        duration_sec = dur_sec,
                        velocity=base_velocity,
                        staff=staff,
                        voice=str(voice) if voice else None,
                        measure=measure_num,
                        part_idx=p_idx,
                        note_id=f"p{p_idx}_o{onset:.3f}_c{j}",
                    ))

            else:
                # skip non-attacks if tied
                if not _is_attack(n):
                    continue
                dur = float(n.duration.quarterLength)
                voice = getattr(n, "voice", None)
                midi = int(n.pitch.midi)
                staff = _infer_staff(n, midi)
                note_infos.append(NoteInfo(
                    pitch_midi=midi,
                    onset=onset,
                    onset_sec=onset_sec,
                    duration=dur,
                    duration_sec=dur_sec,
                    velocity=base_velocity,
                    staff=staff,
                    voice=str(voice) if voice else None,
                    measure=measure_num,
                    part_idx=p_idx,
                    note_id=f"p{p_idx}_o{onset:.3f}_n",
                ))

    note_infos.sort(key=lambda x: (x.onset, x.staff, x.pitch_midi))
    return note_infos

def notes_to_events(note_infos: List[NoteInfo], onset_eps: float = 1e-6) -> List[Event]:
    ''' 
    groups NoteInfo entries into attack events by (onset, staff); tracks held pitches for polyphony
    '''
    buckets: Dict[Tuple[float, int], List[NoteInfo]] = defaultdict(list)
    def quantize(t: float) -> float:
        return round(t / onset_eps) * onset_eps if onset_eps > 0 else t
    for ni in note_infos:
        key = (quantize(ni.onset), ni.staff)
        buckets[key].append(ni)

    sorted_keys = sorted(buckets.keys())
    events: List[Event] = []

    # track active notes: format - {staff_id: [(pitch, end_time_symbolic), ... ]}
    active_notes: Dict[int, List[Tuple[int, float]]] = {1: [], 2: []}
    for (onset, staff) in sorted_keys:
        current_notes = buckets[(onset, staff)]
        current_notes.sort(key=lambda x: x.pitch_midi)
        # clean up notes that ended before this onset; ignoring legato for now, if endtime< onset then finger is free
        active_notes[staff] = [
            (p, end) for (p, end) in active_notes[staff]
            if end > onset + onset_eps
        ]

        attacking_pitches = [x.pitch_midi for x in current_notes]
        held_pitches = [p for (p, end) in active_notes[staff]]
        events.append(Event(
            onset=onset,
            onset_sec=current_notes[0].onset_sec,
            staff=staff,
            event_type="note" if len(attacking_pitches) == 1 else "chord",
            pitches=attacking_pitches,
            held_pitches=sorted(held_pitches),
            notes=[asdict(n) for n in current_notes],
        ))

        for n in current_notes:
            active_notes[staff].append((n.pitch_midi, n.onset + n.duration))

    events.sort(key=lambda e: (e.onset, e.staff))
    return events
    
def musicxml_to_events_json(musicxml_path: str, out_json_path: str) -> None:
    score = m21.converter.parse(musicxml_path)
    print(f"Parsing: {musicxml_path}")
    note_table = extract_note_table(score)
    events = notes_to_events(note_table)

    payload = {
        "source": musicxml_path,
        "num_notes": len(note_table),
        "num_events": len(events),
        "events": [asdict(e) for e in events],
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json_path} with {len(events)} events.")

def debug_print_events(events: List[Event], k: int = 20) -> None:
    for e in events[:k]:
        print(f"onset={e.onset:.3f} staff={e.staff} type={e.event_type} pitches={e.pitches} "
              f"measures={[n.get('measure') for n in e.notes]}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="Path to .musicxml or .mxl")
    parser.add_argument("--outjson", required=True, help="Output JSON path")
    parser.add_argument("--print_k", type=int, default=20, help="Print first k events")
    args = parser.parse_args()

    score = m21.converter.parse(args.infile)
    note_table = extract_note_table(score)
    events = notes_to_events(note_table)

    debug_print_events(events, k=args.print_k)

    payload = {
        "source": args.infile,
        "num_notes": len(note_table),
        "num_events": len(events),
        "events": [asdict(e) for e in events],
    }

    with open(args.outjson, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {args.outjson} with {len(events)} events.")
