# src/render/export_pdf.py
"""
Export CHORD plans to publication-ready PDF visualizations.

Given a plan JSON (e.g., output of scripts/run_pipeline.py or
src.annotate.inject_fingerings), this module produces a PDF containing:

  1) Piano-roll style plot:
       - x-axis: time (seconds)
       - y-axis: MIDI pitch
       - color: staff / hand (RH vs LH)
       - optional finger numbers drawn on each note if "fingering" is present

  2) Wrist trajectory plots:
       - x-axis: time (seconds)
       - y-axis: wrist X (lateral along keyboard) and Y (depth)
       - separate lines for right hand and left hand

Example usage (CLI):

    python -m src.render.export_pdf \
        --plan data/processed/liebestraum_events_dp.json \
        --out  outputs/liebestraum_plan.pdf

The JSON is expected to have the structure produced by run_pipeline.py:

    {
      "source": "...",
      "total_events": N,
      "events": [
        {
          "onset_sec": float,
          "staff": 1 or 2,
          "pitches": [int, ...],
          "fingering": [int or None, ...],  # optional
          "wrist_target": [x, y, z] or None,  # optional
          ...
        },
        ...
      ]
    }

If "wrist_target" is missing, the wrist trajectory plots will simply be empty.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


# ---------- Core Helpers ----------

def _load_plan(plan_path: str) -> Dict[str, Any]:
    """Load a CHORD plan JSON and return the parsed dict."""
    with open(plan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "events" not in data or not isinstance(data["events"], list):
        raise ValueError(f"Plan JSON at '{plan_path}' does not contain an 'events' list.")
    return data


def _collect_piano_roll_data(events: List[Dict[str, Any]]):
    """
    Extract piano-roll data from events.

    Returns:
        rh_points   : list of (t, pitch, finger_str or None)
        lh_points   : same but for left hand
        pitch_min   : min MIDI pitch seen (or 60 if none)
        pitch_max   : max MIDI pitch seen (or 60 if none)
    """
    rh_points = []
    lh_points = []

    pitch_min = None
    pitch_max = None

    for e in events:
        t = float(e.get("onset_sec", 0.0))
        staff = int(e.get("staff", 1))
        pitches = e.get("pitches", [])
        fingering = e.get("fingering", [])

        # Normalize fingering to list of same length as pitches where possible
        if isinstance(fingering, list):
            fingers = fingering
        else:
            fingers = []

        for idx, p in enumerate(pitches):
            p_int = int(p)

            if pitch_min is None or p_int < pitch_min:
                pitch_min = p_int
            if pitch_max is None or p_int > pitch_max:
                pitch_max = p_int

            finger_str: Optional[str] = None
            if idx < len(fingers) and fingers[idx] is not None:
                try:
                    f_val = int(fingers[idx])
                    finger_str = str(f_val)
                except Exception:
                    finger_str = None

            point = (t, p_int, finger_str)

            if staff == 2:
                lh_points.append(point)
            else:
                rh_points.append(point)

    if pitch_min is None:
        pitch_min = 60
    if pitch_max is None:
        pitch_max = 60

    return rh_points, lh_points, pitch_min, pitch_max


def _collect_wrist_data(events: List[Dict[str, Any]]):
    """
    Extract wrist trajectory data from events.

    Returns:
        rh_t, rh_x, rh_y : lists of times, X, Y for RH events with wrist_target
        lh_t, lh_x, lh_y : same for LH
    """
    rh_t, rh_x, rh_y = [], [], []
    lh_t, lh_x, lh_y = [], [], []

    for e in events:
        wt = e.get("wrist_target")
        if not wt or not isinstance(wt, list) or len(wt) < 2:
            continue

        t = float(e.get("onset_sec", 0.0))
        x = float(wt[0])
        y = float(wt[1])
        staff = int(e.get("staff", 1))

        if staff == 2:
            lh_t.append(t)
            lh_x.append(x)
            lh_y.append(y)
        else:
            rh_t.append(t)
            rh_x.append(x)
            rh_y.append(y)

    return rh_t, rh_x, rh_y, lh_t, lh_x, lh_y


# ---------- Plotting ----------

def plot_plan_to_pdf(plan_path: str, pdf_path: str, title: Optional[str] = None) -> None:
    """
    High-level function: load plan JSON, render plots, and save to PDF.

    Args:
        plan_path: path to input plan JSON.
        pdf_path:  path to output PDF file.
        title:     optional title string to display at top.
    """
    data = _load_plan(plan_path)
    events: List[Dict[str, Any]] = data["events"]

    if title is None:
        # Try to derive a short title from source / filename
        src = data.get("source") or os.path.basename(plan_path)
        title = f"CHORD Plan: {os.path.basename(src)}"

    rh_points, lh_points, pitch_min, pitch_max = _collect_piano_roll_data(events)
    rh_t, rh_x, rh_y, lh_t, lh_x, lh_y = _collect_wrist_data(events)

    # Create figure with three subplots:
    #  1) Piano-roll
    #  2) Wrist X over time
    #  3) Wrist Y over time
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    ax_roll, ax_x, ax_y = axes

    fig.suptitle(title, fontsize=14)

    # --- 1) Piano-roll plot ---

    # Right hand notes (red-ish)
    if rh_points:
        t_r = [p[0] for p in rh_points]
        pitch_r = [p[1] for p in rh_points]
        ax_roll.scatter(t_r, pitch_r, marker="o", s=18, alpha=0.7, label="RH notes")

        # Draw finger labels on top
        for t, p, f in rh_points:
            if f is not None:
                ax_roll.text(t, p + 0.2, f, fontsize=7, ha="center", va="bottom", color="red")

    # Left hand notes (blue-ish)
    if lh_points:
        t_l = [p[0] for p in lh_points]
        pitch_l = [p[1] for p in lh_points]
        ax_roll.scatter(t_l, pitch_l, marker="o", s=18, alpha=0.7, label="LH notes")

        for t, p, f in lh_points:
            if f is not None:
                ax_roll.text(t, p - 0.5, f, fontsize=7, ha="center", va="top", color="blue")

    ax_roll.set_ylabel("MIDI Pitch")
    ax_roll.grid(True, linestyle="--", alpha=0.3)
    ax_roll.legend(loc="upper right", fontsize=8)

    # Give a bit of padding around pitches
    ax_roll.set_ylim(pitch_min - 2, pitch_max + 2)

    # --- 2) Wrist X over time ---

    if rh_t:
        ax_x.plot(rh_t, rh_x, ".-", alpha=0.8, label="RH wrist X")
    if lh_t:
        ax_x.plot(lh_t, lh_x, ".-", alpha=0.8, label="LH wrist X")

    ax_x.set_ylabel("Wrist X (m)")
    ax_x.set_title("Wrist Lateral Trajectory (X along keyboard)")
    ax_x.grid(True, linestyle="--", alpha=0.3)
    if rh_t or lh_t:
        ax_x.legend(loc="upper right", fontsize=8)

    # --- 3) Wrist Y over time ---

    if rh_t:
        ax_y.plot(rh_t, rh_y, ".-", alpha=0.8, label="RH wrist Y")
    if lh_t:
        ax_y.plot(lh_t, lh_y, ".-", alpha=0.8, label="LH wrist Y")

    ax_y.set_ylabel("Wrist Y (m)")
    ax_y.set_xlabel("Time (s)")
    ax_y.set_title("Wrist Depth Trajectory (Y in/out)")
    ax_y.grid(True, linestyle="--", alpha=0.3)
    if rh_t or lh_t:
        ax_y.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[export_pdf] Saved plan visualization to: {pdf_path}")


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a CHORD plan JSON to a PDF visualization."
    )
    parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="Path to plan JSON (output of run_pipeline.py or inject_fingerings).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PDF path. If omitted, replaces .json with .pdf next to the plan.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title for the figure.",
    )

    args = parser.parse_args()

    plan_path = args.plan
    if args.out is None:
        base, _ext = os.path.splitext(plan_path)
        pdf_path = base + "_plan.pdf"
    else:
        pdf_path = args.out

    plot_plan_to_pdf(plan_path, pdf_path, title=args.title)


if __name__ == "__main__":
    main()
