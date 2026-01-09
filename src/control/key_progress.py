import numpy as np

def normalize_key_travel(
    k_raw: float,
    k_down_m: float = 0.012,
) -> float:
    """
    Normalize physical key travel (meters) into [0, 1].

    Parameters
    ----------
    k_raw : float
        Current key displacement from rest in meters (>= 0).
    k_down_m : float
        Approximate full key travel from rest to fully down, in meters.

    Returns
    -------
    float
        Normalized key position k in [0, 1].
    """
    if k_down_m <= 0.0:
        raise ValueError("k_down_m must be positive")
    k_norm = k_raw / float(k_down_m)
    # Numeric guard + clipping
    return float(np.clip(k_norm, 0.0, 1.0))


def compute_key_success_row(
    z_jac_row: np.ndarray,
    dt_rem: float,
    k_current: float,
    k80: float = 0.8,
    min_dt: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build the linear row implementing the spec:

        -Δt_rem * φ_j q̇ - u_j ≤ -(k80 - k_j),  u_j ≥ 0

    with φ_j approximated from the fingertip Jacobian.

    We assume:
      * key travel k increases when the fingertip moves *downwards*.
      * In the MuJoCo piano, "down" is negative z, so dk/dt ∝ -dz/dt.

    Parameters
    ----------
    z_jac_row : np.ndarray
        1D array (ndof,) giving the fingertip Jacobian's z-row J_z.
    dt_rem : float
        Remaining time to the strike deadline (seconds). If <= 0, it is
        clamped in magnitude to [min_dt, +∞) so the constraint stays
        well-conditioned and doesn't explode for tiny dt.
    k_current : float
        Current normalized key position in [0, 1].
    k80 : float, optional
        Actuation threshold in normalized units; default 0.8.
    min_dt : float, optional
        Minimum effective |dt| used to avoid degeneracy, default 1e-3.

    Returns
    -------
    A_qdot : np.ndarray
        Shape (1, ndof). Row to be multiplied by q̇ in the QP.
    u : np.ndarray
        Shape (1,). Upper bound for the row in the form:

            A_qdot q̇ - u_slack ≤ u

        where the servo attaches a nonnegative slack u_slack.
    delta_k : float
        The positive amount of key travel remaining (k80 - k_current)+.
    dt_eff : float
        The clamped dt actually used in the computation (signed).
    """
    z_row = np.asarray(z_jac_row, dtype=float).reshape(-1)
    if z_row.ndim != 1:
        raise ValueError("z_jac_row must be 1D")

    # Preserve the sign of dt_rem (early vs late), but clamp magnitude
    # so we don't blow up the row when |dt_rem| is tiny.
    if dt_rem >= 0.0:
        dt_eff = max(float(dt_rem), float(min_dt))
    else:
        dt_eff = -max(abs(float(dt_rem)), float(min_dt))

    # Remaining key travel to threshold; never negative.
    delta_k = max(0.0, float(k80) - float(k_current))

    # If we are already beyond k80, delta_k = 0 → benign constraint:
    #   -dt * φ q̇ - u ≤ 0  (slack can absorb any leftover error)
    # Approximate φ_j = dk/dq̇ via dk/dt ≈ -dz/dt and dz/dt = J_z q̇.
    phi_row = -z_row  # shape (ndof,)

    # From the spec:
    #   -Δt_rem * φ_j q̇ - u_j ≤ -(k80 - k_j)
    # We represent this as:
    #   A_qdot q̇ - u_slack ≤ u
    A_qdot = (-dt_eff * phi_row)[np.newaxis, :]        # (1, ndof)
    u = np.array([-delta_k], dtype=float)              # (1,)

    return A_qdot, u, delta_k, dt_eff
