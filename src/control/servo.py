# src/control/servo.py
import numpy as np
import osqp
from scipy import sparse
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ConstraintBlock:
    """
    Represents a set of linear constraints l <= A q̇ <= u
    in the *q̇ subspace only*.

    Layer A (ChordServo) will attach nonnegative slack variables to
    these rows as needed (zone, separation, key-success).
    """
    A: sparse.csc_matrix  # shape (m, ndof)
    l: np.ndarray         # shape (m,)
    u: np.ndarray         # shape (m,)


@dataclass
class RobotState:
    q: np.ndarray        # joint positions [rad or meters]
    q_dot: np.ndarray    # joint velocities [rad/s or m/s]
    J: np.ndarray        # Jacobian (6 x ndof) for active DOFs
    ee_pos: np.ndarray   # end-effector position
    ee_rot: np.ndarray   # end-effector rotation matrix


class ChordServo:
    """
    Layer A: formulates and solves the QP at 1 kHz.

    Decision variable:

        x = [ q_dot, s_zone, s_guard, s_sep, sigma_key ]

    where:
        - q_dot     : joint velocities
        - s_zone    : non-negative slack for geometric zone constraints
        - s_guard   : reserved for future hard safety constraints
        - s_sep     : non-negative slack for finger separation constraints
        - sigma_key : non-negative slack for key-strike constraints
    """

    def __init__(
        self,
        n_dof: int = 7,
        acc_limit: float = 10.0,
        v_max: float | np.ndarray = 2.0,
        dt: float = 0.001,
    ):
        self.n_dof = n_dof
        self.solver = osqp.OSQP()

        # OSQP problem state
        self.is_setup: bool = False
        self._last_pattern: Optional[dict] = None  # to detect sparsity changes

        # Weights
        self.W_nom = 1.0      # tracking nominal velocity
        self.W_eff = 0.01     # minimize effort (acceleration proxy)
        self.Rho_zone = 1e4   # cost of violating geometry zones
        self.Rho_guard = 1e6  # cost of violating hard safety guards (unused)
        self.Rho_sep = 1e5    # cost of violating separation constraints
        self.W_key = 1e3      # cost of missing key strike deadlines (sigma_key)

        self.acc_limit = float(acc_limit)
        self.v_max = v_max
        self.dt = float(dt)

        self.prev_q_dot = np.zeros(n_dof)

    # ------------------------------------------------------------------ #
    # Internal helpers for warm-start / sparsity tracking
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pattern_signature(P: sparse.csc_matrix,
                           A: sparse.csc_matrix,
                           block_sizes: Tuple[int, int, int, int]) -> dict:
        """
        Build a lightweight signature of the sparsity pattern + sizes.

        We use:
          - shapes of P and A
          - nnz counts of P and A
          - block_sizes: (n_zone, n_guard, n_sep, n_key)

        This is conservative: if any of these change, we re-setup OSQP.
        """
        return {
            "P_shape": P.shape,
            "A_shape": A.shape,
            "P_nnz":   P.nnz,
            "A_nnz":   A.nnz,
            "blocks":  block_sizes,
        }

    def _same_pattern(self, sig: dict) -> bool:
        if self._last_pattern is None:
            return False
        # Simple structural check
        keys = ["P_shape", "A_shape", "P_nnz", "A_nnz", "blocks"]
        return all(self._last_pattern[k] == sig[k] for k in keys)

    # ------------------------------------------------------------------ #
    # Main QP step
    # ------------------------------------------------------------------ #
    def step(
        self,
        state: RobotState,
        q_dot_nom: np.ndarray,
        key_constraints: Optional[ConstraintBlock] = None,
        zone_constraints: Optional[ConstraintBlock] = None,
        guard_constraints: Optional[ConstraintBlock] = None,
        sep_constraints: Optional[ConstraintBlock] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve QP for current timestep.

        Args:
            state: RobotState with current q, q_dot, J, ee_pos, ee_rot.
            q_dot_nom: nominal joint velocities (e.g., from resolved-rate IK).
            key_constraints: l_k <= A_k q_dot <= u_k, to be softened via sigma_key.
            zone_constraints: l_z <= A_z q_dot <= u_z, softened via s_zone.
            guard_constraints: reserved (currently unused).
            sep_constraints: l_sep <= A_sep q_dot <= +inf, softened via s_sep.

        Returns:
            q_dot_cmd: command joint velocities.
            diagnostics: dict with status and slack magnitudes.
        """

        # Determine sizes of slack variable blocks
        n_zone = zone_constraints.A.shape[0] if zone_constraints is not None else 0
        n_guard = guard_constraints.A.shape[0] if guard_constraints is not None else 0
        n_sep = sep_constraints.A.shape[0] if sep_constraints is not None else 0
        n_key = key_constraints.A.shape[0] if key_constraints is not None else 0

        # Total decision vars: [q_dot, s_zone, s_guard, s_sep, sigma_key]
        n_vars = self.n_dof + n_zone + n_guard + n_sep + n_key

        # ---------------- COST FUNCTION ----------------
        # minimize: ||q_dot - q_dot_nom||_W + rho * ||slacks||^2
        P_diag = np.concatenate(
            [
                np.full(self.n_dof, self.W_nom + self.W_eff),  # q_dot costs
                np.full(n_zone, self.Rho_zone),                # s_zone costs
                np.full(n_guard, self.Rho_guard),              # s_guard costs
                np.full(n_sep, self.Rho_sep),                  # s_sep costs
                np.full(n_key, self.W_key),                    # sigma_key costs
            ]
        )
        P = sparse.diags(P_diag, format="csc")

        q_vec = np.zeros(n_vars)
        q_vec[: self.n_dof] = -self.W_nom * q_dot_nom

        # ---------------- CONSTRAINTS ----------------
        constraints: List[Tuple[sparse.csc_matrix, np.ndarray, np.ndarray]] = []

        # Helper to build zero blocks with correct width
        def zrows(m: int, n: int) -> sparse.csc_matrix:
            return sparse.csc_matrix((m, n))

        # 1) Dynamic limits (joint velocity & acceleration bounds merged)
        #    q_dot_min <= I * q_dot <= q_dot_max
        dt = self.dt
        v_max = np.asarray(self.v_max, dtype=float)
        if v_max.ndim == 0:
            v_max = np.full(self.n_dof, float(v_max))
        elif v_max.shape[0] != self.n_dof:
            raise ValueError(f"v_max must have shape ({self.n_dof},) or be scalar.")

        min_v = np.maximum(-v_max, self.prev_q_dot - self.acc_limit * dt)
        max_v = np.minimum(v_max, self.prev_q_dot + self.acc_limit * dt)
        # Guard against infeasible bounds when prev_q_dot drifts outside limits.
        viol = min_v > max_v
        if np.any(viol):
            q_clip = np.clip(self.prev_q_dot, -v_max, v_max)
            min_v[viol] = q_clip[viol]
            max_v[viol] = q_clip[viol]
        # Final numeric guard
        min_v = np.nan_to_num(min_v, nan=0.0, posinf=0.0, neginf=0.0)
        max_v = np.nan_to_num(max_v, nan=0.0, posinf=0.0, neginf=0.0)

        dyn_A = sparse.hstack(
            [
                sparse.eye(self.n_dof, format="csc"),
                zrows(self.n_dof, n_zone + n_guard + n_sep + n_key),
            ]
        )
        constraints.append((dyn_A, min_v, max_v))

        # 2) Zone constraints (soft via s_zone)
        #
        # Base constraint from zone_constraints:
        #       l_z <= A_z * q_dot <= u_z
        #
        # We implement:
        #       A_z * q_dot - s_zone <= u_z
        #       s_zone >= 0
        if n_zone > 0:
            # A_z qdot - s_zone <= u_z
            z_A = sparse.hstack(
                [
                    zone_constraints.A,
                    -sparse.eye(n_zone, format="csc"),      # s_zone
                    zrows(n_zone, n_guard),                # s_guard
                    zrows(n_zone, n_sep),                  # s_sep
                    zrows(n_zone, n_key),                  # sigma_key
                ]
            )
            l_z = np.full(n_zone, -np.inf, dtype=float)
            constraints.append((z_A, l_z, zone_constraints.u))

            # s_zone >= 0
            s_z_A = sparse.hstack(
                [
                    zrows(n_zone, self.n_dof),
                    sparse.eye(n_zone, format="csc"),
                    zrows(n_zone, n_guard + n_sep + n_key),
                ]
            )
            constraints.append(
                (s_z_A, np.zeros(n_zone), np.full(n_zone, np.inf))
            )

        # 3) Separation constraints (soft via s_sep)
        #
        # Base constraint from sep_constraints:
        #       l_sep <= A_sep * q_dot <= +inf
        #
        # We convert l_sep into an upper bound with slack:
        #       A_sep q_dot >= l_sep   ⇔  -A_sep q_dot <= -l_sep
        # Then soften:
        #       -A_sep q_dot + s_sep <= -l_sep
        if n_sep > 0:
            # -A_sep qdot + s_sep <= -l_sep
            sep_A = sparse.hstack(
                [
                    -sep_constraints.A,                      # -A_sep on q_dot
                    zrows(n_sep, n_zone + n_guard),         # s_zone, s_guard
                    sparse.eye(n_sep, format="csc"),        # s_sep
                    zrows(n_sep, n_key),                    # sigma_key
                ]
            )
            l_sep = np.full(n_sep, -np.inf, dtype=float)
            u_sep = -sep_constraints.l.astype(float)
            constraints.append((sep_A, l_sep, u_sep))

            # s_sep >= 0
            s_sep_A = sparse.hstack(
                [
                    zrows(n_sep, self.n_dof + n_zone + n_guard),
                    sparse.eye(n_sep, format="csc"),
                    zrows(n_sep, n_key),
                ]
            )
            constraints.append(
                (s_sep_A, np.zeros(n_sep), np.full(n_sep, np.inf))
            )

        # 4) Key success constraints (soft via sigma_key)
        #
        # Base block from key_constraints:
        #       l_k <= A_k * q_dot <= u_k
        #
        # We introduce sigma_key >= 0:
        #       A_k * q_dot - sigma_key <= u_k
        if n_key > 0:
            k_A = sparse.hstack(
                [
                    key_constraints.A,
                    zrows(n_key, n_zone + n_guard + n_sep),
                    -sparse.eye(n_key, format="csc"),  # sigma_key
                ]
            )
            constraints.append((k_A, key_constraints.l, key_constraints.u))

            # Non-negative sigma_key: sigma >= 0
            sigma_A = sparse.hstack(
                [
                    zrows(n_key, self.n_dof + n_zone + n_guard + n_sep),
                    sparse.eye(n_key, format="csc"),
                ]
            )
            constraints.append(
                (sigma_A, np.zeros(n_key), np.full(n_key, np.inf))
            )

        # (Optional) guard constraints would be inserted here if present.

        # Stack all constraints
        A_stack = sparse.vstack([c[0] for c in constraints], format="csc")
        l_stack = np.concatenate([c[1] for c in constraints])
        u_stack = np.concatenate([c[2] for c in constraints])

        # ---------------- OSQP SETUP / UPDATE (warm-start) ----------------
        block_sizes = (n_zone, n_guard, n_sep, n_key)
        sig = self._pattern_signature(P, A_stack, block_sizes)

        if (not self.is_setup) or (not self._same_pattern(sig)):
            # First time or sparsity changed: full setup.
            self.solver.setup(
                P=P,
                q=q_vec,
                A=A_stack,
                l=l_stack,
                u=u_stack,
                verbose=False,
                time_limit=0.0008,
            )
            self.is_setup = True
            self._last_pattern = sig
        else:
            # Same sparsity pattern: use warm-start via update.
            #
            # We update:
            #   - P diagonal values via Px
            #   - A nonzero values via Ax
            #   - linear term and bounds via q, l, u
            #
            # NOTE: P and A must have same sparsity structure as in the
            # last setup; the pattern signature check above enforces that.
            self.solver.update(
                Px=P.data,
                q=q_vec,
                Ax=A_stack.data,
                l=l_stack,
                u=u_stack,
            )

        # ---------------- SOLVE QP ----------------
        res = self.solver.solve()

        if res.info.status != "solved":
            # Deterministic fallback: damp current velocity toward zero
            return self.prev_q_dot * 0.9, {"status": "fail"}

        q_dot_cmd = res.x[: self.n_dof]

        # Diagnostics (failure attribution)
        slacks = res.x[self.n_dof :] if n_vars > self.n_dof else np.array([])

        # Layout of slacks: [s_zone (n_zone), s_guard (n_guard), s_sep (n_sep), sigma_key (n_key)]
        offset = 0
        s_zone_val = slacks[offset : offset + n_zone] if n_zone else np.array([])
        offset += n_zone
        s_guard_val = slacks[offset : offset + n_guard] if n_guard else np.array([])
        offset += n_guard
        s_sep_val = slacks[offset : offset + n_sep] if n_sep else np.array([])
        offset += n_sep
        sigma_val = slacks[offset : offset + n_key] if n_key else np.array([])

        diag = {
            "status": "opt",
            "max_zone_slack": float(np.max(s_zone_val)) if s_zone_val.size else 0.0,
            "max_sep_slack": float(np.max(s_sep_val)) if s_sep_val.size else 0.0,
            "max_key_slack": float(np.max(sigma_val)) if sigma_val.size else 0.0,
        }

        self.prev_q_dot = q_dot_cmd
        return q_dot_cmd, diag
