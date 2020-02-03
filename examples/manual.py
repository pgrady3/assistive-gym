import gym, assistive_gym
import pybullet as p
import numpy as np
from p_utils import *

class Waypoint:
    def __init__(self, x, theta): 
        self.x = np.array(x)
        self.theta = p.getQuaternionFromEuler(np.array(theta))

env = gym.make('DrinkingJaco-v0')
env.render()
observation = env.reset()
env.world_creation.print_joint_info(env.robot)
keys_actions = {p.B3G_LEFT_ARROW: np.array([-0.01, 0, 0]), p.B3G_RIGHT_ARROW: np.array([0.01, 0, 0]), p.B3G_UP_ARROW: np.array([0, 0, 0.01]), p.B3G_DOWN_ARROW: np.array([0, 0, -0.01])}


waypoints = []
waypoints.append(Waypoint([-0.2, -0.5, 1], [0, np.pi/2, 0]))
waypoints.append(Waypoint([0, 0, 1.3], [0, np.pi/2, 0]))
waypoints.append(Waypoint([-0.2, -0.5, 0.8], [np.pi/2, np.pi/2, 0]))

for wpt in waypoints:
    createPoseMarker(wpt.x, wpt.theta, lifeTime=0)

wpt_idx = 0
while True:
    env.render()

    # Get the position and orientation of the end effector
    x, theta = p.getLinkState(env.robot, 8, computeForwardKinematics=True)[:2]
    wpt_x = waypoints[wpt_idx].x
    wpt_theta = waypoints[wpt_idx].theta
    #print(position)
    #print(p.getEulerFromQuaternion(orientation))

    dist_to_goal = np.linalg.norm(wpt_x - x)
    #print(dist_to_goal)
    if dist_to_goal < 0.05:
        wpt_idx = min(wpt_idx + 1, len(waypoints)-1)

    # IK to get new joint positions (angles) for the robot
    target_joint_positions = p.calculateInverseKinematics(env.robot, 8, wpt_x, wpt_theta)
    target_joint_positions = target_joint_positions[:7]

    # Get the joint positions (angles) of the robot arm
    joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
    joint_positions = np.array(joint_positions)[:7]

    # Set joint action to be the error between current and target joint positions
    joint_action = (target_joint_positions - joint_positions) * 10
    observation, reward, done, info = env.step(joint_action)

