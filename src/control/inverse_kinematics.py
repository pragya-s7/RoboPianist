import numpy as np
import mujoco

class DifferentialIK:
    def __init__(self, model, data, ee_site_name="ee_site"):
        self.model = model
        self.data = data
        
        # FIND THE SITE
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_id < 0:
             print("IK WARNING: Site 'ee_site' not found. Check XML.")
             self.ee_id = 0

        # IDENTIFY JOINTS
        # 0,1,2 are slide_x, slide_y, slide_z
        # 3,4,5 are hinge_yaw, hinge_pitch, hinge_roll
        self.dof_ids = np.array([0, 1, 2, 3, 4, 5])
        self.nv = model.nv

    def solve(self, target_pos):
        """
        Calculates joint velocities (qdot).
        """
        curr_pos = self.data.site_xpos[self.ee_id]
        curr_mat = self.data.site_xmat[self.ee_id].reshape(3, 3)
        
        # 1. Position Error
        err_pos = target_pos - curr_pos
        
        # 2. Orientation Error
        # We want the hand FLAT relative to the world.
        # Desired: X=Right(1,0,0), Y=Forward(0,1,0), Z=Up(0,0,1)
        # Note: In the XML, the palm box is naturally aligned this way when Euler=0
        target_mat = np.eye(3) 

        err_rot = np.zeros(3)
        for i in range(3):
            err_rot += np.cross(curr_mat[:, i], target_mat[:, i])
        err_rot *= 0.5
        
        error = np.concatenate([err_pos * 10.0, err_rot * 5.0])
        
        # 3. Jacobian
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
        
        J = np.vstack([jacp, jacr])
        J_active = J[:, self.dof_ids]

        # 4. Solve
        damping = 0.01
        hess = J_active.T @ J_active + np.eye(len(self.dof_ids)) * damping**2
        grad = J_active.T @ error
        
        q_dot = np.linalg.solve(hess, grad)
        
        return q_dot