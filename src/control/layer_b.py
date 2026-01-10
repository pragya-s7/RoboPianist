# src/control/layer_b.py
"""
Layer B — Timing & Target MPC

Frequency: 50–200 Hz (implemented at 100 Hz)
Role: Make Layer A succeed by:
  1. Adapting deadline margins based on observed residuals
  2. Providing linearized contact models (φ_j) to Layer A
  3. Adjusting nominal velocities to help Layer A succeed
  4. Tracking and logging prediction residuals

Per CHORD v5 Spec:
  - Decision variables: per-note deadline margins δm_j, velocity scalars
  - Outputs: updated aim times, linearized key rows φ_j, task-space velocities, weight schedules
  - Robustification: m_j ← m_j + c·|r̄| + d·σ_r
  - Constraint: Layer B NEVER outputs joint commands (only nominal velocities for Layer A)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from .state_machine import KeyStateMachine, ContactMode


@dataclass
class NoteTrackingState:
    """
    Per-note tracking state for Layer B.

    Tracks history and adaptation parameters for a single note event.
    """
    pitch: int
    finger_idx: int
    onset_original: float  # Original onset time (deadline)

    # Margin adaptation
    margin: float = 0.05  # Current margin (seconds before deadline)
    margin_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Residual tracking: r = Δk_true - Δk_pred
    residual_history: deque = field(default_factory=lambda: deque(maxlen=20))

    # Contact model state
    last_k_true: float = 0.0  # Last observed key position
    last_k_pred: float = 0.0  # Last predicted key position
    last_update_time: float = 0.0

    # Success tracking
    strike_attempted: bool = False
    strike_succeeded: bool = False
    max_key_depth_reached: float = 0.0


@dataclass
class ContactSurrogateModel:
    """
    Local key-contact surrogate model.

    Per spec:
        Δk ≈ f(q, finger_id, key_id, contact_state, q̇)

    At runtime, we linearize:
        φ_j = ∂f / ∂q̇ |_(current)

    For now, we use a simple linear approximation:
        φ_j ≈ -J_z (key travel increases when fingertip moves down)

    Future: Replace with learned MLP or data-driven model.
    """

    # Model parameters (can be learned)
    contact_stiffness: float = 0.9  # How much fingertip motion translates to key motion
    compliance_factor: float = 0.1  # Compliance/damping in contact

    def predict_key_delta(
        self,
        z_jac_row: np.ndarray,
        q_dot: np.ndarray,
        dt: float,
        contact_mode: ContactMode,
    ) -> Tuple[float, np.ndarray]:
        """
        Predict key displacement over time dt.

        Args:
            z_jac_row: Jacobian row for fingertip z-coordinate (ndof,)
            q_dot: Joint velocities (ndof,)
            dt: Time step
            contact_mode: Current contact mode

        Returns:
            Δk_pred: Predicted key displacement
            φ_j: Linearized gradient ∂Δk/∂q̇ (ndof,)
        """
        # Fingertip velocity in z
        z_dot = z_jac_row @ q_dot

        # Contact-mode-dependent scaling
        if contact_mode == ContactMode.STRIKE:
            scale = self.contact_stiffness
        elif contact_mode == ContactMode.HOLD:
            scale = self.contact_stiffness * 0.5  # Less aggressive
        elif contact_mode == ContactMode.PRETOUCH:
            scale = self.contact_stiffness * 0.1  # Very light
        else:
            scale = 0.0  # No contact

        # Key displacement (down is positive, z down is negative)
        delta_k_pred = -scale * z_dot * dt

        # Linearized gradient: φ_j = ∂(Δk)/∂q̇
        phi_j = -scale * dt * z_jac_row

        return delta_k_pred, phi_j

    def compute_residual(self, delta_k_true: float, delta_k_pred: float) -> float:
        """
        Compute prediction residual.

        r = Δk_true - Δk_pred

        Positive r means we under-predicted key motion (good).
        Negative r means we over-predicted key motion (bad - might miss deadline).
        """
        return delta_k_true - delta_k_pred

    def update_parameters(self, residual: float, learning_rate: float = 0.01):
        """
        Online parameter adaptation (optional, for future use).

        Simple gradient descent on contact_stiffness based on residual.
        """
        # If we consistently under-predict (r > 0), increase stiffness
        # If we over-predict (r < 0), decrease stiffness
        self.contact_stiffness += learning_rate * residual
        self.contact_stiffness = np.clip(self.contact_stiffness, 0.1, 1.5)


class LayerB:
    """
    Layer B: Timing & Target MPC

    Runs at 100 Hz (10ms period), processes state from Layer A,
    and outputs adapted targets/constraints for the next Layer A cycle.

    Key responsibilities:
    1. Track per-note state and residuals
    2. Adapt deadline margins based on prediction errors
    3. Provide linearized contact models (φ_j) to Layer A
    4. Adjust nominal velocities to help Layer A succeed
    5. Log all residuals and adaptations for offline analysis
    """

    def __init__(
        self,
        update_frequency: float = 100.0,  # Hz
        initial_margin: float = 0.05,  # seconds
        margin_adapt_c: float = 2.0,  # Coefficient for mean residual
        margin_adapt_d: float = 1.5,  # Coefficient for residual std
        min_margin: float = 0.01,  # Minimum margin (10ms)
        max_margin: float = 0.2,  # Maximum margin (200ms)
    ):
        """
        Initialize Layer B.

        Args:
            update_frequency: Layer B update rate (Hz), should be 50-200 Hz
            initial_margin: Default margin for new notes (seconds)
            margin_adapt_c: Weight for mean residual in margin adaptation
            margin_adapt_d: Weight for residual std in margin adaptation
            min_margin: Minimum allowed margin (prevents over-aggressive adaptation)
            max_margin: Maximum allowed margin (prevents excessive caution)
        """
        self.dt = 1.0 / update_frequency
        self.update_frequency = update_frequency

        # Margin adaptation parameters
        self.initial_margin = initial_margin
        self.margin_adapt_c = margin_adapt_c
        self.margin_adapt_d = margin_adapt_d
        self.min_margin = min_margin
        self.max_margin = max_margin

        # Contact surrogate model
        self.contact_model = ContactSurrogateModel()

        # Per-note tracking (key: (pitch, finger_idx))
        self.active_notes: Dict[Tuple[int, int], NoteTrackingState] = {}

        # Global residual statistics (for logging and analysis)
        self.residual_log: List[Dict] = []

        # Timing
        self.last_update_time: float = 0.0
        self.update_counter: int = 0

    def should_update(self, current_time: float) -> bool:
        """
        Check if Layer B should run this cycle.

        Layer B runs at ~100 Hz, so we skip updates if called more frequently.
        """
        if current_time - self.last_update_time >= self.dt:
            self.last_update_time = current_time
            return True
        return False

    def register_note(
        self,
        pitch: int,
        finger_idx: int,
        onset_time: float,
        initial_margin: Optional[float] = None,
    ):
        """
        Register a new note for tracking.

        Called when a KeyStateMachine enters PRETOUCH mode.
        """
        key = (pitch, finger_idx)
        if key not in self.active_notes:
            margin = initial_margin if initial_margin is not None else self.initial_margin
            self.active_notes[key] = NoteTrackingState(
                pitch=pitch,
                finger_idx=finger_idx,
                onset_original=onset_time,
                margin=margin,
            )

    def unregister_note(self, pitch: int, finger_idx: int):
        """
        Unregister a note (called when state machine completes).
        """
        key = (pitch, finger_idx)
        if key in self.active_notes:
            # Log final statistics
            state = self.active_notes[key]
            self._log_note_completion(state)
            del self.active_notes[key]

    def update(
        self,
        current_time: float,
        state_machines: List[KeyStateMachine],
        key_depths: Dict[int, float],
        finger_jacobians: Dict[int, Tuple[np.ndarray, np.ndarray]],  # finger_idx -> (pos, J_pos)
        qvel_current: np.ndarray,
    ) -> Dict:
        """
        Main Layer B update.

        Args:
            current_time: Current simulation time (seconds)
            state_machines: List of active KeyStateMachine objects
            key_depths: Dict of pitch -> current key depth (meters)
            finger_jacobians: Dict of finger_idx -> (position, J_pos)
            qvel_current: Current joint velocities (ndof,)

        Returns:
            Dict containing:
                - 'adapted_margins': Dict of (pitch, finger_idx) -> adapted margin
                - 'phi_models': Dict of (pitch, finger_idx) -> φ_j gradient
                - 'velocity_scales': Dict of (pitch, finger_idx) -> velocity scale factor
                - 'updated': bool, True if Layer B actually updated this cycle
        """
        # Check if we should update this cycle
        if not self.should_update(current_time):
            return {'updated': False}

        self.update_counter += 1

        # Register new notes and update existing
        adapted_margins = {}
        phi_models = {}
        velocity_scales = {}

        for sm in state_machines:
            key = (sm.pitch, sm.finger_idx)

            # Register if new
            if key not in self.active_notes:
                self.register_note(sm.pitch, sm.finger_idx, sm.onset)

            state = self.active_notes[key]

            # Get current kinematics for this finger
            if sm.finger_idx in finger_jacobians:
                pos, J_pos = finger_jacobians[sm.finger_idx]
                z_jac_row = J_pos[2, :]  # Z-row of Jacobian
            else:
                z_jac_row = np.zeros(len(qvel_current))

            # Update tracking state
            k_true = key_depths.get(sm.pitch, 0.0)
            dt_elapsed = current_time - state.last_update_time if state.last_update_time > 0 else self.dt

            # Predict key motion using contact model
            delta_k_pred, phi_j = self.contact_model.predict_key_delta(
                z_jac_row=z_jac_row,
                q_dot=qvel_current,
                dt=dt_elapsed,
                contact_mode=sm.mode,
            )

            # Compute residual
            delta_k_true = k_true - state.last_k_true
            residual = self.contact_model.compute_residual(delta_k_true, delta_k_pred)

            # Store residual (only during STRIKE/HOLD when contact is active)
            if sm.mode in [ContactMode.STRIKE, ContactMode.HOLD] and abs(delta_k_true) > 1e-6:
                state.residual_history.append(residual)
                self._log_residual(current_time, sm.pitch, sm.finger_idx, residual, delta_k_true, delta_k_pred)

            # Update state
            state.last_k_true = k_true
            state.last_k_pred = state.last_k_true - delta_k_true + delta_k_pred
            state.last_update_time = current_time
            state.max_key_depth_reached = max(state.max_key_depth_reached, k_true)

            # Adapt margin using robustification rule
            if len(state.residual_history) >= 3:
                residuals = np.array(state.residual_history)
                r_mean = np.mean(residuals)
                r_std = np.std(residuals)

                # Robustification rule: m_j ← m_j + c·|r̄| + d·σ_r
                # If we're under-predicting (positive residual), increase margin
                # If variance is high, increase margin for safety
                margin_adjustment = self.margin_adapt_c * abs(r_mean) + self.margin_adapt_d * r_std
                new_margin = state.margin + margin_adjustment * 0.1  # Gradual adaptation
                new_margin = np.clip(new_margin, self.min_margin, self.max_margin)

                state.margin = new_margin
                state.margin_history.append(new_margin)

            # Output adapted parameters
            adapted_margins[key] = state.margin
            phi_models[key] = phi_j

            # Velocity scaling (encourage more aggressive motion if falling behind)
            time_to_deadline = sm.onset - current_time
            if time_to_deadline < state.margin and sm.mode == ContactMode.STRIKE:
                # We're cutting it close, scale up velocity
                velocity_scales[key] = 1.2
            else:
                velocity_scales[key] = 1.0

        # Clean up completed notes
        active_keys = {(sm.pitch, sm.finger_idx) for sm in state_machines}
        for key in list(self.active_notes.keys()):
            if key not in active_keys:
                self.unregister_note(key[0], key[1])

        return {
            'updated': True,
            'adapted_margins': adapted_margins,
            'phi_models': phi_models,
            'velocity_scales': velocity_scales,
        }

    def get_adapted_deadline(self, pitch: int, finger_idx: int, original_onset: float) -> float:
        """
        Get the adapted deadline for a note (original onset - margin).

        This is used by Layer A to construct key-success constraints.
        """
        key = (pitch, finger_idx)
        if key in self.active_notes:
            return original_onset - self.active_notes[key].margin
        else:
            return original_onset - self.initial_margin

    def _log_residual(
        self,
        time: float,
        pitch: int,
        finger_idx: int,
        residual: float,
        delta_k_true: float,
        delta_k_pred: float,
    ):
        """
        Log a residual measurement for offline analysis.
        """
        self.residual_log.append({
            'time': time,
            'pitch': pitch,
            'finger': finger_idx,
            'residual': residual,
            'delta_k_true': delta_k_true,
            'delta_k_pred': delta_k_pred,
        })

    def _log_note_completion(self, state: NoteTrackingState):
        """
        Log statistics when a note completes.
        """
        if len(state.residual_history) > 0:
            residuals = np.array(state.residual_history)
            self.residual_log.append({
                'type': 'completion',
                'pitch': state.pitch,
                'finger': state.finger_idx,
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(np.std(residuals)),
                'final_margin': state.margin,
                'max_key_depth': state.max_key_depth_reached,
            })

    def export_residual_log(self) -> List[Dict]:
        """
        Export residual log for analysis.

        Per spec: "Residual logged: r = Δk_true − Δk_pred"
        """
        return self.residual_log

    def get_statistics(self) -> Dict:
        """
        Get current Layer B statistics for monitoring.
        """
        all_residuals = []
        for state in self.active_notes.values():
            all_residuals.extend(state.residual_history)

        if all_residuals:
            return {
                'n_active_notes': len(self.active_notes),
                'mean_residual': float(np.mean(all_residuals)),
                'std_residual': float(np.std(all_residuals)),
                'mean_margin': float(np.mean([s.margin for s in self.active_notes.values()])),
                'contact_stiffness': self.contact_model.contact_stiffness,
                'update_counter': self.update_counter,
            }
        else:
            return {
                'n_active_notes': len(self.active_notes),
                'update_counter': self.update_counter,
            }
