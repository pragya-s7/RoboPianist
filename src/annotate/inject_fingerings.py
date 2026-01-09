"""
src/annotate/inject_fingerings.py

Utility to inject CHORD DP fingerings into a score representation.

Supports two workflows:

1) MusicXML -> events + fingerings JSON
   -----------------------------------
   Input:  a .mxl / .musicxml / .xml file
   Output: a JSON file:
       {
         "source": "<input path>",
         "num_events": N,
         "events": [
           {
             "onset": ...,
             "onset_sec": ...,
             "staff": 1 or 2,
             "event_type": "note" | "chord",
             "pitches": [...],
             "held_pitches": [...],
             "notes": [...],
             "fingering": [int, ...]   # injected by this script
           },
           ...
         ]
       }

2) events JSON -> events + fingerings JSON
   ---------------------------------------
   Input:  a JSON file that already has "events" in the format produced by
           src.ingest.parse_musicxml.notes_to_events (or compatible).
   Output: the same structure, but with a "fingering" list attached per event.

Fingerings are computed using src.fingering.dp_solver.DPSolver, separately
for right-hand (staff == 1) and left-hand (staff == 2) event streams.

Example usage:

    # From MusicXML:
    python -m src.annotate.inject_fingerings \
        --infile data/raw/liebestraum-no-3-in-a-major-dream-of-love.mxl \
        --outfile data/processed/liebestraum_events_with_fingerings.json

    # From existing events JSON:
    python -m src.annotate.inject_fingerings \
        --infile data/processed/liebestraum_events.json \
        --outfile data/processed/liebestraum_events_with_fingerings.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Union

from src.fingering.dp_solver import DPSolver
from src.ingest import parse_musicxml


JsonEvent = Dict[str, Any]


def _normalize_events_sequence(
    events: Sequence[Union[JsonEvent, parse_musicxml.Event]]
) -> List[JsonEvent]:
    """
    Convert a sequence of Event dataclasses or dicts into a clean
    list of plain dicts. This lets us handle both:

      - Events created by notes_to_events (dataclass)
      - Events loaded from an existing JSON file

    Fields are preserved as-is.
    """
    out: List[JsonEvent] = []
    for e in events:
        if isinstance(e, dict):
            out.append(dict(e))
        else:
            # Assume it's a dataclass-compatible object
            out.append(asdict(e))
    return out


def _inject_fingerings_into_events(events: List[JsonEvent]) -> List[JsonEvent]:
    """
    Core logic: given a list of event dicts, compute and attach DP fingerings.

    We:
      - Split events by staff (1 = right hand, 2 = left hand; default to 1).
      - Run DPSolver separately on each hand's sequence.
      - Reconstruct a single timeline, preserving the original order, with
        a "fingering" list added per event.

    Events are not modified in-place; a new list of dicts is returned.
    """
    solver = DPSolver()

    # Separate by staff, preserving indices
    rh_indices: List[int] = []
    lh_indices: List[int] = []

    for idx, e in enumerate(events):
        staff = int(e.get("staff", 1))
        if staff == 2:
            lh_indices.append(idx)
        else:
            rh_indices.append(idx)

    # Prepare hand-specific event sequences in time order
    rh_events = [events[i] for i in rh_indices]
    lh_events = [events[i] for i in lh_indices]

    # Solve fingerings for each hand
    # DPSolver expects each event dict to contain at least:
    #   - "pitches": List[int]
    #   - "onset_sec": float
    # Extra fields are ignored.
    rh_fingerings: List[List[int]] = solver.solve(rh_events) if rh_events else []
    lh_fingerings: List[List[int]] = solver.solve(lh_events) if lh_events else []

    # Re-inject fingerings into a new event list, preserving original order
    result: List[JsonEvent] = []
    rh_iter = iter(rh_fingerings)
    lh_iter = iter(lh_fingerings)

    for idx, e in enumerate(events):
        e_new = dict(e)  # shallow copy

        staff = int(e_new.get("staff", 1))
        pitches = e_new.get("pitches", [])

        if not pitches:
            # Rest or malformed event: no fingering
            e_new["fingering"] = []
        else:
            if staff == 2 and lh_events:
                fingers = next(lh_iter)
            else:
                fingers = next(rh_iter)

            # Safety: ensure length matches number of attacking pitches
            # (we do not attempt to finger held_pitches).
            if len(fingers) != len(pitches):
                # Fallback: truncate or pad with None to avoid crashes.
                L = min(len(fingers), len(pitches))
                trimmed = list(fingers[:L])
                if L < len(pitches):
                    trimmed.extend([None] * (len(pitches) - L))
                e_new["fingering"] = trimmed
            else:
                e_new["fingering"] = list(fingers)

        result.append(e_new)

    return result


def _load_from_musicxml(path: str) -> Dict[str, Any]:
    """
    Load a MusicXML/.mxl score and convert it into an events JSON-like payload
    (without fingerings yet).

    Uses:
      - parse_musicxml.extract_note_table
      - parse_musicxml.notes_to_events
    """
    print(f"[inject_fingerings] Parsing MusicXML: {path}")
    score = parse_musicxml.m21.converter.parse(path)
    note_table = parse_musicxml.extract_note_table(score)
    events_dataclass = parse_musicxml.notes_to_events(note_table)

    events = _normalize_events_sequence(events_dataclass)

    payload: Dict[str, Any] = {
        "source": os.path.abspath(path),
        "num_notes": len(note_table),
        "num_events": len(events),
        "events": events,
    }
    return payload


def _load_from_events_json(path: str) -> Dict[str, Any]:
    """
    Load an existing events JSON file. It is expected to contain:
        {
          "events": [ ... ]
        }

    Other top-level fields are preserved and passed through unchanged.
    """
    print(f"[inject_fingerings] Loading events JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError(f"JSON at '{path}' does not contain an 'events' list.")

    payload["events"] = _normalize_events_sequence(events)
    return payload


def _detect_input_type(path: str) -> str:
    """
    Return 'musicxml' or 'json' based on file extension.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".mxl", ".musicxml", ".xml"]:
        return "musicxml"
    elif ext in [".json"]:
        return "json"
    else:
        raise ValueError(
            f"Unsupported input extension '{ext}'. Expected .mxl/.musicxml/.xml or .json"
        )


def inject_fingerings(infile: str, outfile: str) -> None:
    """
    High-level entry point:

      - Load score/events depending on infile type.
      - Inject DP fingerings.
      - Write outfile JSON with updated events.
    """
    mode = _detect_input_type(infile)

    if mode == "musicxml":
        payload = _load_from_musicxml(infile)
    else:
        payload = _load_from_events_json(infile)

    events = payload.get("events", [])
    if not isinstance(events, list):
        raise ValueError("Payload['events'] must be a list.")

    print(f"[inject_fingerings] Found {len(events)} events. Solving fingerings...")
    events_with_fingerings = _inject_fingerings_into_events(events)
    payload["events"] = events_with_fingerings
    payload["num_events"] = len(events_with_fingerings)
    payload["has_fingerings"] = True

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[inject_fingerings] Wrote {len(events_with_fingerings)} events with fingerings to: {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject CHORD DP fingerings into a MusicXML score or events JSON."
    )
    parser.add_argument(
        "--infile",
        type=str,
        required=True,
        help="Input file (.mxl/.musicxml/.xml or events .json).",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output JSON path with injected 'fingering' lists.",
    )

    args = parser.parse_args()
    inject_fingerings(args.infile, args.outfile)


if __name__ == "__main__":
    main()
