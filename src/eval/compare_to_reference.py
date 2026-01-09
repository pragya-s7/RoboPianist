# src/eval/compare_to_reference.py
"""
CLI tool to compare a CHORD-generated plan JSON against a reference JSON.

Example usage:

    python -m src.eval.compare_to_reference \
        --reference data/processed/liebestraum_annotated.json \
        --candidate data/processed/liebestraum_events.json

Assumptions:
  - Both JSON files have the same basic structure:
        {
          "source": "...",
          "events": [ ... ]
        }
  - Each event in "events" follows the format described in src.eval.metrics.

This script:
  1) Loads both JSONs.
  2) Aligns events by (staff, pitches, onset_sec within a tolerance).
  3) Computes fingering accuracy and timing error metrics.
  4) Prints a human-readable summary to stdout.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from .metrics import compute_all_metrics


def load_events(path: str) -> List[Dict[str, Any]]:
    """
    Load events from a JSON file.

    The file is expected to contain:
        {
          "events": [ ... ]
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("events")
    if not isinstance(events, list):
        raise ValueError(f"JSON at '{path}' does not contain an 'events' list.")
    return events


def pretty_print_metrics(metrics: Dict[str, Any], onset_tolerance: float) -> None:
    """
    Print metrics in a human-readable format.
    """
    align = metrics["alignment"]
    fing = metrics["fingering"]
    tim = metrics["timing"]

    print("=== CHORD Evaluation ===")
    print(f"- Onset alignment tolerance: ±{onset_tolerance * 1000:.1f} ms\n")

    print("1) Alignment:")
    print(f"   Reference events:  {align['num_ref_events']}")
    print(f"   Candidate events:  {align['num_cand_events']}")
    print(f"   Matched pairs:     {align['num_matches']}")
    print(f"   Unmatched (ref):   {align['num_ref_unmatched']}")
    print(f"   Unmatched (cand):  {align['num_cand_unmatched']}")
    print()

    print("2) Fingering Accuracy:")
    ne = fing["num_events_compared"]
    print(f"   Events compared (with fingering on both sides): {ne}")
    if ne > 0:
        print(f"   Event-level exact matches:  {fing['num_events_exact']} "
              f"({fing['event_exact_fraction'] * 100:.1f}%)")
    else:
        print("   Event-level exact matches:  N/A (no comparable fingerings)")

    nn = fing["num_notes_compared"]
    if nn > 0:
        print(f"   Note positions compared:    {nn}")
        print(f"   Note-level exact matches:   {fing['num_notes_exact']} "
              f"({fing['note_exact_fraction'] * 100:.1f}%)")
    else:
        print("   Note-level exact matches:   N/A (no comparable note positions)")
    print()

    print("3) Timing Error (on matched events):")
    n_t = tim["num_events"]
    if n_t > 0:
        print(f"   Events used:          {n_t}")
        print(f"   Mean abs error:       {tim['mae'] * 1000:.2f} ms")
        print(f"   Median abs error:     {tim['median_abs_error'] * 1000:.2f} ms")
        print(f"   RMSE:                 {tim['rmse'] * 1000:.2f} ms")
        print(f"   Max abs error:        {tim['max_abs_error'] * 1000:.2f} ms")
    else:
        print("   No matched events; timing metrics are undefined.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a CHORD-generated plan JSON to a reference JSON."
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference JSON (ground-truth annotations).",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Path to candidate JSON (CHORD output from run_pipeline.py).",
    )
    parser.add_argument(
        "--onset_tolerance",
        type=float,
        default=0.03,
        help="Onset alignment tolerance in seconds (default: 0.03 = 30 ms).",
    )

    args = parser.parse_args()

    ref_events = load_events(args.reference)
    cand_events = load_events(args.candidate)

    metrics = compute_all_metrics(
        ref_events=ref_events,
        cand_events=cand_events,
        onset_tolerance=args.onset_tolerance,
    )

    pretty_print_metrics(metrics, onset_tolerance=args.onset_tolerance)


if __name__ == "__main__":
    main()
