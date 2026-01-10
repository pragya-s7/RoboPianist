# src/control/constraint_factory.py
import numpy as np
from scipy import sparse
from typing import List, Optional 

from .servo import ConstraintBlock
from .key_progress import compute_key_success_row, normalize_key_travel


class ConstraintFactory:
    """
    Factory for constructing linear constraint blocks for the CHORD-Ω QP.

    This class builds constraints only in terms of q̇. The ChordServo
    attaches nonnegative slack variables and assembles the full OSQP
    problem.

    All methods return ConstraintBlock with:

        A : (m, ndof)  (CSC sparse)
        l : (m,)       lower bounds
        u : (m,)       upper bounds
    """

    def __init__(self, ndof: int):
        self.ndof = int(ndof)

    # ------------------------------------------------------------------
    # Geometry halfspaces with two-point collocation for zone constraints
    # ------------------------------------------------------------------
    def create_zone_constraint(
        self,
        a: np.ndarray,
        b: float,
        y: np.ndarray,
        J_pos: np.ndarray,
        tau_list: list[float],
    ) -> ConstraintBlock:
        """
        Build a *zone* constraint for a single geometric halfspace:

            aᵀ y ≤ b

        with two-point collocation over time horizon τ ∈ tau_list:

            aᵀ ( y + τ J_pos q̇ ) ≤ b + s_zone

        We hand back only the q̇ terms:

            A_zone q̇ - s_zone ≤ u_zone

        and let the servo attach s_zone >= 0.

        Parameters
        ----------
        a : np.ndarray
            Normal vector in ℝ³ defining the halfspace aᵀ y ≤ b.
        b : float
            Halfspace offset.
        y : np.ndarray
            Current fingertip position in ℝ³.
        J_pos : np.ndarray
            Position Jacobian, shape (3, ndof).
        tau_list : list[float]
            Collocation times (e.g., [dt/2, dt]).

        Returns
        -------
        ConstraintBlock
            A : (m, ndof) with one row per τ in tau_list
            l : (-∞, ..., -∞)
            u : (b - aᵀ y, ..., b - aᵀ y), row-normalized with A.
        """
        a = np.asarray(a, dtype=float).reshape(3)
        y = np.asarray(y, dtype=float).reshape(3)
        J_pos = np.asarray(J_pos, dtype=float)
        if J_pos.shape[0] != 3:
            raise ValueError(f"J_pos must be (3, ndof); got {J_pos.shape}")

        if J_pos.shape[1] != self.ndof:
            raise ValueError(
                f"J_pos has width {J_pos.shape[1]}, expected ndof={self.ndof}"
            )

        base_rhs = float(b - a @ y)

        rows = []
        u_vals = []
        for tau in tau_list:
            tau = float(tau)
            row = tau * (a @ J_pos)  # shape (ndof,)
            rows.append(row)
            u_vals.append(base_rhs)

        if not rows:
            # Benign no-op constraint if tau_list is empty
            A_csc = sparse.csc_matrix((0, self.ndof))
            l = np.zeros((0,), dtype=float)
            u = np.zeros((0,), dtype=float)
            return ConstraintBlock(A=A_csc, l=l, u=u)

        A_dense = np.vstack(rows)              # (m, ndof)
        u_dense = np.array(u_vals, dtype=float).reshape(-1, 1)

        # ---- Row normalization for numerical stability ----
        # We want each row of A to have ||row||₂ ≈ 1 so OSQP sees
        # reasonably scaled coefficients.
        row_norms = np.linalg.norm(A_dense, axis=1, keepdims=True)
        # Avoid division by zero for degenerate rows
        row_norms[row_norms < 1e-8] = 1.0
        A_dense /= row_norms
        u_dense /= row_norms

        A_csc = sparse.csc_matrix(A_dense)
        m = A_dense.shape[0]
        l = np.full(m, -np.inf, dtype=float)
        u = u_dense.reshape(m)

        return ConstraintBlock(A=A_csc, l=l, u=u)

    # ------------------------------------------------------------------
    # Spec-correct key-success constraint
    # ------------------------------------------------------------------
    def create_key_success_constraint(
        self,
        z_jac_row: np.ndarray,
        dt_rem: float,
        k_current: float,
        k80: float = 0.8,
    ) -> ConstraintBlock:
        """
        Build a key-success constraint implementing:

            -Δt_rem * φ_j q̇ - u_j ≤ -(k80 - k_j),  u_j ≥ 0

        where φ_j is approximated from the fingertip Jacobian's z-row
        using dk/dt ≈ -dz/dt.

        We represent this to the servo as:

            A_qdot q̇ - u_slack ≤ u

        with l = -∞, so the servo adds a nonnegative slack u_slack.
        """
        A_qdot, u_vec, _, _ = compute_key_success_row(
            z_jac_row=z_jac_row,
            dt_rem=dt_rem,
            k_current=k_current,
            k80=k80,
        )

        if A_qdot.shape[1] != self.ndof:
            raise ValueError(
                f"Key-success constraint row has width {A_qdot.shape[1]}, "
                f"expected ndof={self.ndof}"
            )

        # Row-normalize for stability (consistent with zone constraints)
        row_norm = np.linalg.norm(A_qdot, axis=1, keepdims=True)
        row_norm[row_norm < 1e-8] = 1.0
        A_qdot_norm = A_qdot / row_norm
        u_vec_norm = u_vec / row_norm.reshape(-1)

        A = sparse.csc_matrix(A_qdot_norm)     # (1, ndof)
        l = np.array([-np.inf], dtype=float)   # one-sided inequality

        return ConstraintBlock(
            A=A,
            l=l,
            u=u_vec_norm,
        )

    # ------------------------------------------------------------------
    # Backwards-compatible wrapper around key-success constraint
    # ------------------------------------------------------------------
    def create_key_constraint(
        self,
        z_jac_row: np.ndarray,
        *,
        dt_rem: float | None = None,
        k_current: float | None = None,
        k80: float = 0.8,
        key_travel_m: float | None = None,
    ) -> ConstraintBlock:
        """
        Wrapper around `create_key_success_constraint`.

        You must provide:
          - dt_rem: remaining time to strike deadline (seconds)
          - either k_current (normalized [0,1]) or key_travel_m (meters)
        """
        if k_current is None and key_travel_m is None:
            raise ValueError(
                "create_key_constraint requires either k_current "
                "(normalized) or key_travel_m (meters)."
            )

        if dt_rem is None:
            raise ValueError(
                "create_key_constraint now requires dt_rem (time to deadline)."
            )

        if k_current is None:
            k_current = normalize_key_travel(float(key_travel_m))

        return self.create_key_success_constraint(
            z_jac_row=np.asarray(z_jac_row, dtype=float).reshape(-1),
            dt_rem=float(dt_rem),
            k_current=float(k_current),
            k80=float(k80),
        )

    # ------------------------------------------------------------------
    # Separation constraints between fingertips (Tier 1)
    # ------------------------------------------------------------------
    def create_separation_constraint(
        self,
        J_i: np.ndarray,
        J_j: np.ndarray,
        p_i: np.ndarray,
        p_j: np.ndarray,
        n: np.ndarray,
        d_min: float,
    ) -> ConstraintBlock:
        """
        Build a *single* separation constraint between two fingertips:

            nᵀ (J_i - J_j) q̇ + s_sep ≥ d_min - nᵀ (p_i - p_j)

        which ensures that, along direction n, fingertip i stays at
        least d_min ahead of fingertip j.

        We provide only the q̇ row:

            A_sep q̇ ≥ rhs   ==>  l_sep = rhs, u_sep = +∞

        The servo then converts it into a soft inequality with s_sep.
        """
        J_i = np.asarray(J_i, dtype=float)
        J_j = np.asarray(J_j, dtype=float)
        p_i = np.asarray(p_i, dtype=float).reshape(3)
        p_j = np.asarray(p_j, dtype=float).reshape(3)
        n = np.asarray(n, dtype=float).reshape(3)

        if J_i.shape != J_j.shape:
            raise ValueError(
                f"J_i and J_j must have same shape; got {J_i.shape} vs {J_j.shape}"
            )
        if J_i.shape[1] != self.ndof:
            raise ValueError(
                f"Jacobian width {J_i.shape[1]} does not match ndof={self.ndof}"
            )

        # Use only position part (first 3 rows)
        if J_i.shape[0] < 3:
            raise ValueError("Jacobian must have at least 3 rows for position part.")

        J_pos_i = J_i[:3, :]
        J_pos_j = J_j[:3, :]

        # Normalize n to keep scaling sane
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            raise ValueError("Direction n must be non-zero for separation constraint.")
        n_unit = n / n_norm

        A_sep = (n_unit @ (J_pos_i - J_pos_j)).reshape(1, -1)  # (1, ndof)
        rhs = float(d_min - n_unit @ (p_i - p_j))

        A = sparse.csc_matrix(A_sep)
        l = np.array([rhs], dtype=float)
        u = np.array([np.inf], dtype=float)

        return ConstraintBlock(A=A, l=l, u=u)

    # ------------------------------------------------------------------
    # NEW: Tier 2 - Gantry Workspace Constraints
    # ------------------------------------------------------------------
    def create_gantry_workspace_constraint(
        self,
        gantry_qpos_translation: np.ndarray,  # (3,) for tx, ty, tz current positions
        gantry_qvel_indices: List[int],       # Indices of tx, ty, tz in full ndof (e.g., [0, 1, 2])
        workspace_min: np.ndarray,            # (3,) min for tx, ty, tz
        workspace_max: np.ndarray,            # (3,) max for tx, ty, tz
        dt: float,                            # Simulation timestep
        max_gantry_speed: float = 0.5,        # m/s max allowed translational speed
    ) -> ConstraintBlock:
        """
        Creates hard constraints to keep the gantry's translational joints
        (tx, ty, tz) within a specified workspace box.

        The constraint is formulated as:
            q_min <= q_current + q_dot * dt <= q_max
        which simplifies to:
            (q_min - q_current) / dt <= q_dot <= (q_max - q_current) / dt

        These bounds are applied to the relevant q_dot components.

        Parameters
        ----------
        gantry_qpos_translation : np.ndarray
            Current positions of gantry's translational joints (tx, ty, tz), shape (3,).
        gantry_qvel_indices : List[int]
            List of indices corresponding to tx, ty, tz in the full q_dot vector.
        workspace_min : np.ndarray
            Minimum allowed position for tx, ty, tz, shape (3,).
        workspace_max : np.ndarray
            Maximum allowed position for tx, ty, tz, shape (3,).
        dt : float
            Simulation timestep.
        max_gantry_speed : float, optional
            Maximum absolute velocity (m/s) allowed for the gantry axes,
            used to cap computed q_dot bounds for numerical stability.

        Returns
        -------
        ConstraintBlock
            A : (3, ndof) - Identity for gantry translation DOFs, zero elsewhere.
            l : (3,)      - Lower bounds for q_dot (tx, ty, tz).
            u : (3,)      - Upper bounds for q_dot (tx, ty, tz).
        """
        num_trans_axes = len(gantry_qpos_translation)

        if dt <= 0:
            return ConstraintBlock(sparse.csc_matrix((0, self.ndof)), np.array([]), np.array([]))

        # Build sparse matrix efficiently using COO format then convert to CSC
        row_indices = []
        col_indices = []
        data_vals = []
        l_gantry_trans = np.zeros(num_trans_axes)
        u_gantry_trans = np.zeros(num_trans_axes)

        for i in range(num_trans_axes):
            q_curr = gantry_qpos_translation[i]
            q_min = workspace_min[i]
            q_max = workspace_max[i]

            # This row constrains q_dot_i
            row_indices.append(i)
            col_indices.append(gantry_qvel_indices[i])
            data_vals.append(1.0)

            # Calculate bounds on q_dot_i to reach q_min/q_max in one timestep
            lower_bound_q_dot_i = (q_min - q_curr) / dt
            upper_bound_q_dot_i = (q_max - q_curr) / dt

            # Numerical stability & ensuring movement away from violation
            # If already below min, must move positively or stay still.
            if q_curr < q_min:
                lower_bound_q_dot_i = max(lower_bound_q_dot_i, 0.0)
            # If already above max, must move negatively or stay still.
            if q_curr > q_max:
                upper_bound_q_dot_i = min(upper_bound_q_dot_i, 0.0)

            # Cap bounds by overall max gantry speed to prevent huge velocities
            l_gantry_trans[i] = np.clip(lower_bound_q_dot_i, -max_gantry_speed, max_gantry_speed)
            u_gantry_trans[i] = np.clip(upper_bound_q_dot_i, -max_gantry_speed, max_gantry_speed)

        A_gantry_trans = sparse.csc_matrix(
            (data_vals, (row_indices, col_indices)),
            shape=(num_trans_axes, self.ndof)
        )

        return ConstraintBlock(A=A_gantry_trans, l=l_gantry_trans, u=u_gantry_trans)