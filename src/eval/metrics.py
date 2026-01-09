# src/eval/metrics.py
"""
Evaluation utilities for CHORD plans.

This module provides functions to:
  - Align events from a reference JSON and a candidate JSON
  - Compute fingering accuracy metrics
  - Compute onset timing error statistics
  - Aggregate everything into a single metrics dict suitable for logs / tables

Assumptions about JSON event structure (both reference and candidate):
  Each file is a dict with key "events" -> List[dict], where each event dict
  contains at least:

    - "onset_sec": float      (attack time in seconds)
    - "staff": int            (1 = RH, 2 = LH)
    - "pitches": List[int]    (ascending MIDI pitches for this event)
    - optionally "fingering": List[int] of same length as "pitches"

This matches the output of src.ingest.parse_musicxml.notes_to_events
and src.fingering.dp_solver.DPSolver, as well as the annotated JSONs.

Typical usage:
    from src.eval.metrics import (
        align_events,
        compute_fingering_metrics,
        compute_timing_metrics,
        compute_all_metrics,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math


@dataclass
class AlignmentResult:
    """
    Result of aligning reference and candidate events.

    Attributes:
        matches:
            List of (ref_idx, cand_idx) pairs indicating matched events.
        ref_unmatched:
            List of indices (into ref_events) that had no match.
        cand_unmatched:
            List of indices (into cand_events) that had no match.
    """
    matches: List[Tuple[int, int]]
    ref_unmatched: List[int]
    cand_unmatched: List[int]


def _event_key(e: Dict[str, Any]) -> Tuple[int, Tuple[int, ...]]:
    """
    Key used to group events by (staff, pitches tuple).

    Pitches are treated as an ordered tuple. We assume both reference and
    candidate outputs keep pitches sorted ascending, which is consistent
    with notes_to_events.
    """
    staff = int(e.get("staff", 1))
    pitches = tuple(int(p) for p in e.get("pitches", []))
    return staff, pitches


def align_events(
    ref_events: List[Dict[str, Any]],
    cand_events: List[Dict[str, Any]],
    onset_tolerance: float = 0.03,
) -> AlignmentResult:
    """
    Align reference and candidate events using onset time, staff, and pitches.

    Strategy:
        - Group events by (staff, pitches).
        - Within each group, sort by onset_sec.
        - Greedily match each candidate event to the closest-in-time reference
          event in the same group whose onset_sec differs by <= onset_tolerance.
        - Any leftover events are reported as unmatched.

    Args:
        ref_events: List of reference event dicts.
        cand_events: List of candidate event dicts.
        onset_tolerance: Max allowed |t_ref - t_cand| to consider a match (seconds).

    Returns:
        AlignmentResult with index pairs and unmatched indices.
    """
    # Pre-group indices by (staff, pitches)
    ref_groups: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
    cand_groups: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}

    for i, e in enumerate(ref_events):
        k = _event_key(e)
        ref_groups.setdefault(k, []).append(i)

    for j, e in enumerate(cand_events):
        k = _event_key(e)
        cand_groups.setdefault(k, []).append(j)

    matches: List[Tuple[int, int]] = []
    ref_used = set()
    cand_used = set()

    # For each group key that appears in either ref or cand
    all_keys = set(ref_groups.keys()).union(cand_groups.keys())

    for key in sorted(all_keys):
        ref_idx_list = ref_groups.get(key, [])
        cand_idx_list = cand_groups.get(key, [])

        if not ref_idx_list or not cand_idx_list:
            # Nothing to match in this group
            continue

        # Sort by onset_sec within groups (ascending)
        ref_idx_list = sorted(
            ref_idx_list,
            key=lambda i: float(ref_events[i].get("onset_sec", 0.0)),
        )
        cand_idx_list = sorted(
            cand_idx_list,
            key=lambda j: float(cand_events[j].get("onset_sec", 0.0)),
        )

        # Greedy match: for each cand, find nearest-in-time unused ref
        r_ptr = 0
        for c_idx in cand_idx_list:
            t_c = float(cand_events[c_idx].get("onset_sec", 0.0))

            best_r = None
            best_dt = None

            # Scan forward on the ref list from current pointer; we allow
            # going backward a little (to catch slightly earlier matches),
            # but keep it simple for now and just check unused refs.
            for r_idx in ref_idx_list:
                if r_idx in ref_used:
                    continue
                t_r = float(ref_events[r_idx].get("onset_sec", 0.0))
                dt = abs(t_r - t_c)
                if dt > onset_tolerance:
                    continue

                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_r = r_idx

            if best_r is not None:
                matches.append((best_r, c_idx))
                ref_used.add(best_r)
                cand_used.add(c_idx)

    # Any ref/cand indices not in used sets are unmatched
    ref_unmatched = [i for i in range(len(ref_events)) if i not in ref_used]
    cand_unmatched = [j for j in range(len(cand_events)) if j not in cand_used]

    return AlignmentResult(
        matches=sorted(matches, key=lambda ij: ij[0]),
        ref_unmatched=ref_unmatched,
        cand_unmatched=cand_unmatched,
    )


@dataclass
class FingeringMetrics:
    """
    Fingering accuracy metrics.

    Attributes:
        num_events_compared:
            Number of matched event pairs where both had non-empty “fingering”.
        num_events_exact:
            Number of those for which the entire fingering list matched exactly.
        event_exact_fraction:
            num_events_exact / num_events_compared (0 if denominator is 0).

        num_notes_compared:
            Total number of individual note positions compared across all
            matched events (only positions where both ref and cand have a
            fingering index).
        num_notes_exact:
            Number of note positions with identical finger index.
        note_exact_fraction:
            num_notes_exact / num_notes_compared (0 if denominator is 0).
    """
    num_events_compared: int
    num_events_exact: int
    event_exact_fraction: float

    num_notes_compared: int
    num_notes_exact: int
    note_exact_fraction: float


def compute_fingering_metrics(
    ref_events: List[Dict[str, Any]],
    cand_events: List[Dict[str, Any]],
    matches: List[Tuple[int, int]],
) -> FingeringMetrics:
    """
    Compute fingering accuracy over matched event pairs.

    We compare the "fingering" fields on each event, which are expected to be
    List[int] of the same length as "pitches". For robustness, we only compare
    up to the min length of the two lists.

    Args:
        ref_events: List of reference events.
        cand_events: List of candidate events.
        matches: List of (ref_idx, cand_idx) pairs from align_events.

    Returns:
        FingeringMetrics with per-event and per-note accuracy.
    """
    num_events_compared = 0
    num_events_exact = 0

    num_notes_compared = 0
    num_notes_exact = 0

    for ref_idx, cand_idx in matches:
        ref_f = ref_events[ref_idx].get("fingering")
        cand_f = cand_events[cand_idx].get("fingering")

        if not ref_f or not cand_f:
            # Skip if either side lacks fingering info
            continue

        # Ensure list-like
        ref_list = list(ref_f)
        cand_list = list(cand_f)

        L = min(len(ref_list), len(cand_list))
        if L == 0:
            continue

        num_events_compared += 1

        # Per-note comparison
        exact_notes_this_event = 0
        for i in range(L):
            num_notes_compared += 1
            if int(ref_list[i]) == int(cand_list[i]):
                num_notes_exact += 1
                exact_notes_this_event += 1

        # Per-event exactness
        if exact_notes_this_event == L and len(ref_list) == len(cand_list):
            num_events_exact += 1

    if num_events_compared > 0:
        event_exact_fraction = num_events_exact / float(num_events_compared)
    else:
        event_exact_fraction = 0.0

    if num_notes_compared > 0:
        note_exact_fraction = num_notes_exact / float(num_notes_compared)
    else:
        note_exact_fraction = 0.0

    return FingeringMetrics(
        num_events_compared=num_events_compared,
        num_events_exact=num_events_exact,
        event_exact_fraction=event_exact_fraction,
        num_notes_compared=num_notes_compared,
        num_notes_exact=num_notes_exact,
        note_exact_fraction=note_exact_fraction,
    )


@dataclass
class TimingMetrics:
    """
    Onset timing error statistics over matched events.

    Attributes:
        num_events:
            Number of matched event pairs used.
        mae:
            Mean absolute error |t_cand - t_ref|.
        rmse:
            Root-mean-square error.
        max_abs_error:
            Maximum absolute timing error.
        median_abs_error:
            Median of absolute timing errors.
    """
    num_events: int
    mae: float
    rmse: float
    max_abs_error: float
    median_abs_error: float


def compute_timing_metrics(
    ref_events: List[Dict[str, Any]],
    cand_events: List[Dict[str, Any]],
    matches: List[Tuple[int, int]],
) -> TimingMetrics:
    """
    Compute onset timing error statistics over matched events.

    Args:
        ref_events: List of reference events.
        cand_events: List of candidate events.
        matches: List of (ref_idx, cand_idx) pairs from align_events.

    Returns:
        TimingMetrics with MAE, RMSE, max, and median absolute error.
    """
    errors: List[float] = []

    for ref_idx, cand_idx in matches:
        t_ref = float(ref_events[ref_idx].get("onset_sec", 0.0))
        t_cand = float(cand_events[cand_idx].get("onset_sec", 0.0))
        errors.append(abs(t_cand - t_ref))

    n = len(errors)
    if n == 0:
        return TimingMetrics(
            num_events=0,
            mae=0.0,
            rmse=0.0,
            max_abs_error=0.0,
            median_abs_error=0.0,
        )

    mae = sum(errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    max_abs = max(errors)

    sorted_errs = sorted(errors)
    mid = n // 2
    if n % 2 == 1:
        median = sorted_errs[mid]
    else:
        median = 0.5 * (sorted_errs[mid - 1] + sorted_errs[mid])

    return TimingMetrics(
        num_events=n,
        mae=mae,
        rmse=rmse,
        max_abs_error=max_abs,
        median_abs_error=median,
    )


def compute_all_metrics(
    ref_events: List[Dict[str, Any]],
    cand_events: List[Dict[str, Any]],
    onset_tolerance: float = 0.03,
) -> Dict[str, Any]:
    """
    Convenience wrapper to:
        1) Align events
        2) Compute fingering and timing metrics
        3) Aggregate into a single dict

    Returns:
        metrics: dict with keys:
            - "alignment": {
                  "num_ref_events",
                  "num_cand_events",
                  "num_matches",
                  "num_ref_unmatched",
                  "num_cand_unmatched",
              }
            - "fingering": FingeringMetrics as dict
            - "timing": TimingMetrics as dict
    """
    alignment = align_events(ref_events, cand_events, onset_tolerance=onset_tolerance)

    fingering = compute_fingering_metrics(ref_events, cand_events, alignment.matches)
    timing = compute_timing_metrics(ref_events, cand_events, alignment.matches)

    metrics: Dict[str, Any] = {
        "alignment": {
            "num_ref_events": len(ref_events),
            "num_cand_events": len(cand_events),
            "num_matches": len(alignment.matches),
            "num_ref_unmatched": len(alignment.ref_unmatched),
            "num_cand_unmatched": len(alignment.cand_unmatched),
        },
        "fingering": {
            "num_events_compared": fingering.num_events_compared,
            "num_events_exact": fingering.num_events_exact,
            "event_exact_fraction": fingering.event_exact_fraction,
            "num_notes_compared": fingering.num_notes_compared,
            "num_notes_exact": fingering.num_notes_exact,
            "note_exact_fraction": fingering.note_exact_fraction,
        },
        "timing": {
            "num_events": timing.num_events,
            "mae": timing.mae,
            "rmse": timing.rmse,
            "max_abs_error": timing.max_abs_error,
            "median_abs_error": timing.median_abs_error,
        },
    }

    return metrics
