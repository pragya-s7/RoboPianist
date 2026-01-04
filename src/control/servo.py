import numpy as np
import osqp
from scipy import sparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class RobotState:
    q: np.ndarray # joint positions [rad]
    q_dot: np.ndarray # joint velocities [rad/s]
    J: np.ndarray # Jacobian (6 x n_dof)
    ee_pos: np.ndarray # end-effector position 
    ee_rot: np.ndarray # end-effector rotation matrix

@dataclass
class ConstraintBlock:
    ''' represents set of linear constraints l <= A <= u'''
    A: sparse.csc_matrix
    l: np.ndarray
    u: np.ndarray


class ChordServo:
    '''
    Layer A: formulates and solves the QP at 1 kHz
    decision var x = [q_dot(N), s_zone(M), s_guard(K), s_sep(P), u_key(L)]
    '''
    def __init__(self, n_dof=7):
        self.n_dof = n_dof
        self.solver = osqp.OSQP()
        self.is_setup = False

        # weights
        self.W_nom = 1.0 # tracking nominal velocity
        self.W_eff = 0.01 # minimize effort (accel)
        self.Rho_zone = 1e4 # cost of violating geometry zones
        self.Rho_guard = 1e6 # cost of violating hard safety guards
        self.Rho_sep = 1e5 # cost of finger collision
        self.W_key = 1e3 # cost of missing a key strike deadline

        self.prev_q_dot = np.zeros(n_dof)

    def step(self, state: RobotState, q_dot_nom: np.ndarray,
             key_constraints: Optional[ConstraintBlock] = None, 
             zone_constraints: Optional[ConstraintBlock] = None,
             guard_constraints: Optional[ConstraintBlock] = None) -> Tuple[np.ndarray, dict]:
        ''' solves QP for curr timestep, returns: (q_dot_cmd, diagnostics)'''

        # determine dimension -- need to know sizes of slacks based on input constraints
        n_zone = zone_constraints.l.shape[0] if zone_constraints else 0
        n_guard = guard_constraints.l.shape[0] if guard_constraints else 0
        n_key = key_constraints.l.shape[0] if key_constraints else 0

        # total decision vars [q_dot, s_zone, s_guard, u_key] (skipping s_sep for MVP)
        n_vars = self.n_dof + n_zone + n_guard + n_key

        # build cost matrix P and vector q
        # minimize: || q_dot - q_dot_nom||_W + rho * ||s||^2
        P_diag = np.concatenate([
            np.full(self.n_dof, self.W_nom + self.W_eff), # q_dot costs
            np.full(n_zone, self.Rho_zone), # s_zone costs
            np.full(n_guard, self.Rho_guard), # s_guard costs
            np.full(n_key, self.W_key) # u_key costs
        ])
        P = sparse.diags(P_diag, format='csc')

        q_vec = np.zeros(n_vars)
        q_vec[:self.n_dof] = -self.W_nom * q_dot_nom

        # add regulariation/damping to q_vec if needed based on prev state

        # build constraints
        # A_total * x <= upper and >= lower
        constraints = []

        # dynamics limits (joint velocity & accel bounds merged)
        # q_dot_min <= I * q_dot <= q_dot_max
        acc_limit = 10.0 # rad/s^2
        dt = 0.001 
        v_max = 2.0 # rad/s

        # compute dynamic window
        min_v = np.maximum(-v_max, self.prev_q_dot - acc_limit * dt)
        max_v = np.minimum(v_max, self.prev_q_dot + acc_limit * dt)

        # block [ I_dof, 0, 0, 0]
        dyn_A = sparse.hstack([
            sparse.eye(self.n_dof),
            sparse.csc_matrix((self.n_dof, n_zone + n_guard + n_key))
        ])
        constraints.append((dyn_A, min_v, max_v))

        # zone constraints (soft)
        # A_zone * q_dot <= b + s_zone --> A_zone * q_dot - I * s_zone <= b
        if n_zone > 0:
            z_A = sparse.hstack([
                zone_constraints.A,
                -sparse.eye(n_zone),
                sparse.csc_matrix((n_zone, n_guard + n_key))
            ])
            constraints.append((z_A, np.full(n_zone, -np.inf), zone_constraints.u))

            # non-negative slack constraint: s_zone >= 0
            # block: [0, I, 0, 0]
            s_z_A = sparse.hstack([
                sparse.csc_matrix((n_zone, self.n_dof)),
                sparse.eye(n_zone),
                sparse.csc_matrix((n_zone, n_guard + n_key))
            ])
            constraints.append((s_z_A, np.zeros(n_zone), np.full(n_zone, np.inf)))

        # key success constraints (soft/relaxable via u)
        # J_key * q_dot >= v_req - u_key --> J_key * q_dot + I * u_key >= v_req
        if n_key > 0:
            # block: [A_key, 0, 0, I]
            k_A = sparse.hstack([
                key_constraints.A,
                sparse.csc_matrix((n_key, n_zone + n_guard)),
                sparse.eye(n_key)
            ])
            constraints.append((k_A, key_constraints.l, np.full(n_key, np.inf)))

            # non-negative u: u >= 0
            u_A = sparse.hstack([
                sparse.csc_matrix((n_key, self.n_dof + n_zone + n_guard)),
                sparse.eye(n_key)
            ])
            constraints.append((u_A, np.zeros(n_key), np.full(n_key, np.inf)))

        # stack all matrices
        A_stack = sparse.vstack([c[0] for c in constraints], format='csc')
        l_stack = np.concatenate([c[1] for c in constraints])
        u_stack = np.concatenate([c[2] for c in constraints])

        # solve
        if not self.is_setup:
            self.solver.setup(P=P, q=q_vec, A=A_stack, l=l_stack, u=u_stack, verbose=False, time_limit = 0.0008)
            self.is_setup = True
        else:
            # note: OSQP update requires same sparsity structure. if active constraints change size must resetup
            # for now assume strictly fixed sizes or re-setup
            self.solver = osqp.OSQP()
            self.solver.setup(P=P, q=q_vec, A=A_stack, l=l_stack, u=u_stack, verbose=False)

        res = self.solver.solve()

        if res.info.status != 'solved':
            # deterministic fallback: damp curr velocity to zero
            return self.prev_q_dot * 0.9, {'status': 'fail'}

        q_dot_cmd = res.x[:self.n_dof]

        # diagnostics (failure attrib)
        slacks = res.x[self.n_dof:]
        s_zone_val = slacks[:n_zone] if n_zone else []
        u_val = slacks[n_zone+n_guard:] if n_key else []

        diag = {
            'status': 'opt',
            'max_zone_slack': np.max(s_zone_val) if len(s_zone_val) > 0 else 0.0,
            'max_key_slack': np.max(u_val) if len(u_val) > 0 else 0.0
        }

        self.prev_q_dot = q_dot_cmd
        return q_dot_cmd, diag