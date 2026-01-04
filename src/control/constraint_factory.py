import numpy as np
from scipy import sparse
from src.control.servo import ConstraintBlock

class ConstraintFactory:
    '''generates linear constraints for the CHORD Servo'''

    @staticmethod
    def create_zone_constraint(J: np.ndarray, current_pos: np.ndarray, target_pos: np.ndarray, radius: float = 0.01) -> ConstraintBlock:
        '''
        creates a 'Zone' constraint: keep end-effector within radius of target
        linearized: J*qdot*Dt <= radius + target - current
        '''
        # J comes in as 6xN, we only care about linear position
        J_pos = J[:3, :]
        dt = 0.001
        max_approach_vel = 1.0 # m/s -- cap the approach speed

        # vector from current to target
        raw_err = target_pos - current_pos

        # clamp the error 
        max_step = max_approach_vel * dt
        err = np.clip(raw_err, -0.5, 0.5)

        dist = np.linalg.norm(raw_err)
        effective_radius = max(radius, dist - max_step)

        # approx a sphere with a box (6 constraints: +/- X, _/- Y, +/- Z)

        # upper bounds (x, y, z); J*dt * qdot <= radius + target - current
        A_upper = J_pos * dt
        u_upper = np.full(3, radius) + err
        l_upper = np.full(3, -np.inf) # only enforcing upper bound here

        # lower bound (x, y, z); -J*dt * qdot <= radius - target + current
        A_lower = -J_pos * dt
        u_lower = np.full(3, radius) - err

        # combine into one block of 6 rows
        A = sparse.vstack([sparse.csc_matrix(A_upper), sparse.csc_matrix(A_lower)])
        u = np.concatenate([u_upper, u_lower])
        l = np.full(6, -np.inf)

        return ConstraintBlock(A=A, l=l, u=u)
    
    @staticmethod
    def create_key_constraint(J: np.ndarray, key_velocity: float = 0.5) -> ConstraintBlock:
        '''
        ensures downward velocity > key_veloicty
        assumes Z-down or Z-up depending on frame
        if Z is up, want v_z < -key_vel
        '''
        # row for Z velocity (index 2)
        row_z = J[2, :]

        # A * qdot <= u
        # v_z <= -key_vel

        A = sparse.csc_matrix(row_z.reshape(1, -1))
        u = np.array([-key_velocity])
        l = np.array([-np.inf])
        
        return ConstraintBlock(A=A, l=l, u=u)