from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple, Optional


class ContactMode(Enum):
    """
    Discrete contact modes for a single (pitch, finger) note event.

    HOVER    : fingertip is comfortably above the key (no intent to press yet).
    PRETOUCH : fingertip is just above the key surface, ready to strike.
    STRIKE   : actively pushing the key down.
    HOLD     : key is down, fingertip maintaining key depression.
    RELEASE  : lifting back up to hover.
    """
    HOVER = auto()
    PRETOUCH = auto()
    STRIKE = auto()
    HOLD = auto()
    RELEASE = auto()


@dataclass
class KeyStateMachine:
    """
    State machine for a **single note event** on one finger.

    Fields expected by the rest of the system:
        - pitch: MIDI pitch (21–108).
        - onset: desired onset time in seconds.
        - finger_idx: 1–5 (thumb..little finger).
        - mode: current ContactMode.
        - done: True once this event has fully completed (back in hover).

    Public methods:
        - update(t, key_depth)
        - get_target(key_surface_z) -> (z_target, z_vmax, stiffness)
    """

    pitch: int
    onset: float
    finger_idx: int

    mode: ContactMode = field(default=ContactMode.HOVER, init=False)
    done: bool = field(default=False, init=False)

    # --- Timing parameters (can be tuned globally) ---
    # How long before onset we start moving from HOVER → PRETOUCH
    approach_lead: float = 0.25  # seconds

    # Nominal durations (used as timeouts / fallbacks)
    strike_timeout: float = 0.15  # max time we allow STRIKE before forcing HOLD
    hold_duration: float = 0.25   # how long to keep key down (HOLD)
    release_duration: float = 0.20  # nominal lift-back time

    # --- Key depth parameters ---
    # Keys are slide joints in [0, key_range]; we treat "pressed" as depth ≥ key_down_threshold.
    key_range: float = 0.012
    key_down_fraction: float = 0.75  # fraction of full travel we count as a "press"

    # --- Vertical geometry offsets (relative to key_surface_z) ---
    hover_height: float = 0.040     # comfortable hover above key
    pretouch_clearance: float = 0.005  # mm clearance above top of key in PRETOUCH
    strike_overtravel: float = 0.003   # push slightly past surface (helps actuate)

    # internal bookkeeping
    _strike_start_time: Optional[float] = field(default=None, init=False)
    _hold_start_time: Optional[float] = field(default=None, init=False)
    _release_start_time: Optional[float] = field(default=None, init=False)

    def _key_down_threshold(self) -> float:
        """Absolute key depth at which we consider the key 'down'."""
        return self.key_range * self.key_down_fraction

    # ---------------------- PUBLIC API ---------------------- #

    def update(self, t: float, key_depth: float) -> None:
        """
        Advance the state machine given current simulation time and key depth.

        Args:
            t: current simulation time (seconds).
            key_depth: slide joint position of this key (meters).
        """
        if self.done:
            return

        key_down = key_depth >= self._key_down_threshold()

        if self.mode == ContactMode.HOVER:
            self._update_hover(t)
        elif self.mode == ContactMode.PRETOUCH:
            self._update_pretouch(t, key_down)
        elif self.mode == ContactMode.STRIKE:
            self._update_strike(t, key_down)
        elif self.mode == ContactMode.HOLD:
            self._update_hold(t, key_down)
        elif self.mode == ContactMode.RELEASE:
            self._update_release(t, key_down)

    def get_target(self, key_surface_z: float) -> Tuple[float, Optional[float], float]:
        """
        For the current mode, return a vertical target and vertical velocity bound.

        Args:
            key_surface_z: z-height of the key's *top surface* (meters), as estimated
                           from geometry (we use your PianoGeometry center_z for now).

        Returns:
            (z_target, z_vmax, stiffness)

            z_target : desired fingertip z coordinate.
            z_vmax   : symmetric bound on vertical velocity, i.e.
                       ee_vel_des.z is clamped to [-z_vmax, +z_vmax].
                       (None means "no special clamp".)
            stiffness: scalar weighting term you can feed into Layer A if needed
                       (for now we just return a mode-dependent constant).
        """
        # HOVER: far above the key, gentle velocities
        if self.mode == ContactMode.HOVER:
            z_target = key_surface_z + self.hover_height
            z_vmax = 0.30
            k = 1.0
        # PRETOUCH: just above the key surface, preparing to strike
        elif self.mode == ContactMode.PRETOUCH:
            z_target = key_surface_z + self.pretouch_clearance
            z_vmax = 0.40
            k = 2.0
        # STRIKE: push slightly into the key
        elif self.mode == ContactMode.STRIKE:
            z_target = key_surface_z - self.strike_overtravel
            # Allow faster motion for a crisp attack
            z_vmax = 0.90
            k = 3.0
        # HOLD: keep key down with minimal motion
        elif self.mode == ContactMode.HOLD:
            z_target = key_surface_z - self.strike_overtravel
            z_vmax = 0.20
            k = 2.5
        # RELEASE: lift back up to hover
        elif self.mode == ContactMode.RELEASE:
            z_target = key_surface_z + self.hover_height
            z_vmax = 0.50
            k = 1.5
        else:
            # Fallback (should not happen)
            z_target = key_surface_z + self.hover_height
            z_vmax = 0.30
            k = 1.0

        return float(z_target), float(z_vmax), float(k)

    # ---------------------- MODE UPDATES ---------------------- #

    def _update_hover(self, t: float) -> None:
        """
        HOVER → PRETOUCH when we are within `approach_lead` seconds of onset.
        """
        if t >= self.onset - self.approach_lead:
            self.mode = ContactMode.PRETOUCH

    def _update_pretouch(self, t: float, key_down: bool) -> None:
        """
        PRETOUCH → STRIKE at onset, or earlier if key already moving down.

        We don't strictly require key_down here; STRIKE is the active push phase.
        """
        # If we somehow already see the key down (e.g. other finger), jump to HOLD.
        if key_down:
            self.mode = ContactMode.HOLD
            self._hold_start_time = t
            return

        # Normal path: start STRIKE exactly at onset (or slightly later if we're behind).
        if t >= self.onset:
            self.mode = ContactMode.STRIKE
            self._strike_start_time = t

    def _update_strike(self, t: float, key_down: bool) -> None:
        """
        STRIKE → HOLD once the key is down OR once we hit the timeout.
        """
        if key_down:
            self.mode = ContactMode.HOLD
            self._hold_start_time = t
            return

        # Timeout safety: if we haven't achieved key_down in strike_timeout, proceed anyway.
        if self._strike_start_time is None:
            # Shouldn't happen, but guard against None
            self._strike_start_time = t

        if t - self._strike_start_time >= self.strike_timeout:
            self.mode = ContactMode.HOLD
            self._hold_start_time = t

    def _update_hold(self, t: float, key_down: bool) -> None:
        """
        HOLD → RELEASE once we have held for hold_duration OR if the key is already
        starting to come up (e.g. springs).
        """
        if not key_down:
            # Key is already coming back up: transition to RELEASE.
            self.mode = ContactMode.RELEASE
            self._release_start_time = t
            return

        if self._hold_start_time is None:
            self._hold_start_time = t

        if t - self._hold_start_time >= self.hold_duration:
            self.mode = ContactMode.RELEASE
            self._release_start_time = t

    def _update_release(self, t: float, key_down: bool) -> None:
        """
        RELEASE → HOVER (and done=True) once key is up OR timeout passes.
        """
        if not key_down:
            # Key is up: event completed.
            self.mode = ContactMode.HOVER
            self.done = True
            return

        if self._release_start_time is None:
            self._release_start_time = t

        if t - self._release_start_time >= self.release_duration:
            # Force completion even if the key depth is slightly noisy.
            self.mode = ContactMode.HOVER
            self.done = True
