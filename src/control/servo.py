import numpy as np
import osqp
from scipy import sparse
from dataclasses import dataclass

@dataclass
class RobotState:
    q: np.ndarray # joint positions [rad]
    q_dot: np.ndarray # joint velocities [rad/s]
    ee_pose: np.ndarray # end-effector [x, y, z] (simplified)

class ChordServo:
    def __init__(self, n_dof=7):
        self.n_dof = n_dof
        self.solver = osqp.OSQP()
        self.is_setup = False

        # hyperparameters
        self.W_vel = 1.0 # cost of deviating from normal velocity
        self.W_slack = 1000.0 # cost of violating a constraint (safety)
        self.dt = 0.001 # 1 kHz control loop

    def _build_qp(self, J, x_err, q_dot_nom):
        '''
        constructs the QP matrices for:
        min || q_dot - q_dot_nom || ^2 + rho * || slack || ^2 
        s.t. J * q_dot = v_des + slack
        q_min <= q_dot <= q_max
        '''
        # decision variables: X = [q_dot (n_dof_), slack (3)]
        n_vars = self.n_dof + 3

        # quadratic cost matrix (P)
        # minimize q_dot ^2 (smoothness) and slack^2 (tracking)
        P = np.eye(n_vars)
        P[:self.n_dof, :self.n_dof] *= self.W_vel
        P[self.n_dof:, self.n_dof:] *= self.W_slack
        P = sparse.csc_matrix(P)

        # linear cost vector (q)
        # want q_dot to be close to q_dot_nom
        q_vec = np.zeros(n_vars)
        q_vec[:self.n_dof] = -self.W_vel * q_dot_nom

        # constraints (A, l, u)
        # equality: J * q_dot - slack = Kp * error
        # A matrix structure: [J, -I]
        neg_eye = -np.eye(3)
        A_eq = np.hstack([J, neg_eye])

        # velocity gain (P-controller in velocity space)
        Kp = 10.0
        v_des = Kp * x_err

        l = v_des # lower bound
        u = v_des # upper bound (equality)

        # convert to sparse
        A = sparse.csc_matrix(A_eq)

        return P, q_vec, A, l, u
    
    def step(self, robot_state: RobotState, target_pos: np.ndarray, J: np.ndarray) -> np.ndarray:
        '''
        calculates the optimal joint velocities to reach target_pos

        args:
        robot_state: current q, q_dot
        target_pos: desired [x, y, z] of wrist
        J: Jacobian matrix (3 x n_dof) at current q

        returns: q_dot_cmd: safe velocity command for robot drivers
        '''
        # error calculation: simple cartesian difference
        x_err = target_pos - robot_state.ee_pose

        # nominal velocity (damping / nullspace bias)
        q_dot_nom = np.zeros(self.n_dof)

        # setup or update solver
        P, q, A, l, u = self._build_qp(J, x_err, q_dot_nom)

        if not self.is_setup:
            self.solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
            self.is_setup = True
        else:
            self.solver.update(q=q, l=l, u=u, Ax=A.data)

        # solve
        res = self.solver.solve()

        if res.info.status != 'solved':
            print(f"QP FAIL: {res.info.status}")
            return np.zeros(self.n_dof)

        # extract q_dot from solution vector
        solution = res.x
        q_dot_cmd = solution[:self.n_dof]
        slack = solution[self.n_dof:]

        # diagnostic: if slack is high, we are failing to track
        if np.linalg.norm(slack) > 0.1:
            pass # really should log as tracking warning

        return q_dot_cmd