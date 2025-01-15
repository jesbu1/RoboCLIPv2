import mujoco
import numpy as np


class SimulatedRobot:
    def __init__(self, m, d) -> None:
        """
        :param m: mujoco model
        :param d: mujoco data
        """
        self.m = m
        self.d = d

    def _pos2pwm(self, pos: np.ndarray) -> np.ndarray:
        """
        :param pos: numpy array of joint positions in range [-pi, pi]
        :return: numpy array of pwm values in range [0, 4096]
        """
        return (pos / 3.14 + 1.0) * 4096

    def _pwm2pos(self, pwm: np.ndarray) -> np.ndarray:
        """
        :param pwm: numpy array of pwm values in range [0, 4096]
        :return: numpy array of joint positions in range [-pi, pi]
        """
        return (pwm / 2048 - 1) * 3.14

    def _pwm2norm(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of pwm values in range [0, 4096]
        :return: numpy array of values in range [0, 1]
        """
        return x / 4096

    def _norm2pwm(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of values in range [0, 1]
        :return: numpy array of pwm values in range [0, 4096]
        """
        return x * 4096

    def read_position(self) -> np.ndarray:
        """
        :return: numpy array of current joint positions in range [0, 4096]
        """
        return self.d.qpos[:6]

    def read_velocity(self):
        """
        Reads the joint velocities of the robot.
        :return: list of joint velocities,
        """
        return self.d.qvel

    def rot_mat_to_quat(self, rot_mat):
        """
        :param rot_mat: 3x3 rotation matrix
        :return: quaternion
        """
        assert rot_mat.shape == (3, 3), "Invalid shape rotation matrix {}".format(
            rot_mat
        )

        w = np.sqrt(1 + rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]) / 2
        x = (rot_mat[2, 1] - rot_mat[1, 2]) / (4 * w)
        y = (rot_mat[0, 2] - rot_mat[2, 0]) / (4 * w)
        z = (rot_mat[1, 0] - rot_mat[0, 1]) / (4 * w)
        return np.array([x, y, z, w])

    def quat_to_rpy(self, quat):
        """
        :param quat: quaternion
        :return: roll, pitch, yaw angles (radians)
        """
        x, y, z, w = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = np.arctan2(t3, t4)
        return X, Y, Z

    def rpy_to_quat(self, rpy):
        """
        :param rpy: roll, pitch, yaw angles (radians)
        :return: quaternion
        """
        r, p, y = rpy
        sr, cr = np.sin(r), np.cos(r)
        sp, cp = np.sin(p), np.cos(p)
        sy, cy = np.sin(y), np.cos(y)
        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr
        return np.array([qx, qy, qz, qw])

    def read_ee_pos(self, joint_name="link_6"):
        """
        :param joint_name: name of the end effector joint
        :return: numpy array of end effector position
        """
        joint_id = self.m.body(joint_name).id
        ee_position = self.d.geom_xpos[joint_id]

        # Rotation matrix of the end effector (flattened 3x3 matrix in row-major order)
        ee_rotation_matrix_flat = self.d.geom_xmat[joint_id]

        # Reshape into a 3x3 matrix
        ee_rotation_matrix = np.array(ee_rotation_matrix_flat).reshape(3, 3)

        quat = self.rot_mat_to_quat(ee_rotation_matrix)

        rpy = self.quat_to_rpy(quat)

        return ee_position, rpy

    def inverse_kinematics(self, ee_target_pos, joint_name="link_6"):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id
        # get the current end effector position
        ee_pos = self.d.geom_xpos[joint_id]
        # compute the jacobian
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jac, None, joint_id)
        # compute target joint velocities
        qpos = self.read_position()
        qdot = np.dot(np.linalg.pinv(jac[:, :6]), ee_target_pos - ee_pos)
        # apply the joint velocities
        q_target_pos = qpos + qdot * 0.2
        return q_target_pos

    def inverse_kinematics_rot(self, ee_target_pos, ee_target_rpy, joint_name="link_6"):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param ee_target_rot: numpy array or quaternion of target end effector rotation (RPY or quaternion)
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id

        # Get the current end effector position and rotation (in quaternion)
        ee_pos = self.d.geom_xpos[joint_id]
        ee_rot = self.d.geom_xmat[joint_id].reshape(
            3, 3
        )  # 3x3 rotation matrix for the end effector

        # Target rotation is in RPY
        target_rot_quat = self.rpy_to_quat(ee_target_rpy)

        # Current end effector rotation as quaternion
        current_rot_quat = self.rot_mat_to_quat(ee_rot)

        # Compute the change in position (translation)
        delta_pos = ee_target_pos - ee_pos

        breakpoint()

        # Compute the change in orientation (rotation)
        delta_rot_quat = self.quaternion_multiply(
            self.quaternion_conjugate(current_rot_quat), target_rot_quat
        )

        # Compute the Jacobian for both position and rotation (6xN)
        jac_pos = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jac_pos, None, joint_id)

        # Compute the rotation Jacobian (3xN)
        jac_rot = np.zeros((3, self.m.nv))  # Placeholder for rotational Jacobian

        # Compute the rotational Jacobian using mj_jacBody
        mujoco.mj_jacBody(self.m, self.d, jac_rot, None, joint_id)

        # Now combine both the position and rotation Jacobians (6xN)
        jac = np.vstack([jac_pos, jac_rot])

        # Compute the target joint velocities for both position and orientation
        qpos = self.read_position()

        # For position: Solve for joint velocities to reach the target position
        qdot_pos = np.dot(np.linalg.pinv(jac_pos), delta_pos)

        # For orientation: Solve for joint velocities to reach the target orientation
        qdot_rot = np.dot(
            np.linalg.pinv(jac_rot), self.quaternion_to_axis_angle(delta_rot_quat)
        )

        # Combine both velocity contributions (position and rotation)
        qdot = np.concatenate([qdot_pos, qdot_rot])

        # Apply the joint velocities
        q_target_pos = qpos + qdot * 0.2

        return q_target_pos

    def quaternion_multiply(self, q1, q2):
        # Function to multiply two quaternions
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    def quaternion_conjugate(self, q):
        # Function to compute the conjugate of a quaternion
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quaternion_to_axis_angle(self, q):
        # Converts a quaternion to an axis-angle representation (3x1)
        angle = 2 * np.arccos(q[0])
        s = np.sqrt(1 - q[0] ** 2)
        if s < 0.001:
            axis = np.array([1, 0, 0])
        else:
            axis = np.array([q[1] / s, q[2] / s, q[3] / s])
        return axis * angle

    def set_target_pos(self, target_pos):
        # self.d.ctrl = target_pos
        self.d.qpos[:6] = target_pos
