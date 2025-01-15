# import time
# import mujoco
# import mujoco.viewer

# from interface import SimulatedRobot

# m = mujoco.MjModel.from_xml_path("envs/simulation/low_cost_robot_6dof/lift_cube.xml")
# d = mujoco.MjData(m)

# r = SimulatedRobot(m, d)

# with mujoco.viewer.launch_passive(m, d) as viewer:
#     start = time.time()
#     while viewer.is_running():
#         step_start = time.time()
#         mujoco.mj_step(m, d)
#         viewer.sync()
#         # Rudimentary time keeping, will drift relative to wall clock.
#         time_until_next_step = m.opt.timestep - (time.time() - step_start)
#         if time_until_next_step > 0:
#             time.sleep(time_until_next_step)

import pybullet as p
import pybullet_data
import numpy as np

# Connect to PyBullet and set up the environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the URDF of your robot
robot_id = p.loadURDF(
    "envs/simulation/low_cost_robot_6dof/follower_meshes/follower.urdf",
    useFixedBase=False,
)

# Define the target end effector pose (position and orientation)
target_position = [0.5, 0.5, 0.5]
target_orientation = p.getQuaternionFromEuler(
    [0, 0, np.pi / 2]
)  # Example: 90 degrees rotation about Z axis

# Compute the IK solution for the end effector pose
end_effector_link_index = 6  # Change this to the actual end effector link index
ik_solution = p.calculateInverseKinematics(
    robot_id, end_effector_link_index, target_position, target_orientation
)

# Print the joint angles from the IK solution
print("IK Solution (joint angles):", ik_solution)

# To compute the end effector pose from given joint positions (FK):
joint_positions = ik_solution  # Using IK solution as joint positions for FK

# Get the position and orientation of the end effector based on these joint positions
link_state = p.getLinkState(
    robot_id, end_effector_link_index, computeForwardKinematics=True
)

# Extract position and orientation
ee_position = link_state[0]
ee_orientation = link_state[1]

# Print the end effector pose
print("End Effector Position:", ee_position)
print("End Effector Orientation (Quaternion):", ee_orientation)

# Disconnect from PyBullet
p.disconnect()
