import gym, assistive_gym
import pybullet as p
import numpy as np
from p_utils import *
from pyquaternion import Quaternion

class Waypoint:
    def __init__(self, x, theta): 
        self.x = np.array(x)
        self.theta = p.getQuaternionFromEuler(np.array(theta))

def ParseArgs():
    parser = argparse.ArgumentParser(description='Manual waypoint follower')
    parser.add_argument('--file', help='Waypoint file')
    parser.add_argument('--env', default='DrinkingJaco-v0', help='Environment, eg DrinkingJaco-v0')
    parser.add_argument('--nogui', action='store_true', help='Environment, eg DrinkingJaco-v0')
    args = parser.parse_args()

    waypoints = []

    with open(args.file) as file_in: # Load waypoints file in
        for line in file_in:
            floats = [float(x) for x in line.split(',')]
            waypoints.append(Waypoint([floats[0],floats[1],floats[2]], [floats[3],floats[4],floats[5]]))
            print(floats)
            
    env = gym.make(args.env)
    return env, waypoints, args

env, waypoints, args = ParseArgs()
if not args.nogui:
    env.render()

for rollout in range(10):

    observation = env.reset()
    x, theta = p.getLinkState(env.robot, 8, computeForwardKinematics=True)[:2] # Get the position and orientation of the end effector
    old_wpt_theta = theta
    
    wpt_idx = 0

    observation, reward, done, info = env.step(np.zeros(7))
    total_reward = 0

    while not done:
        if not args.nogui:
            env.render()
        
        x, theta = p.getLinkState(env.robot, 8, computeForwardKinematics=True)[:2] # Get the position and orientation of the end effector
        delta_tool_goal = observation[7:10]
        goal_pos = x - delta_tool_goal
        wpt_x = waypoints[wpt_idx].x
        wpt_theta = waypoints[wpt_idx].theta

        target_pos = goal_pos + wpt_x        
        dist_to_goal = np.linalg.norm(target_pos - x)

        # IK to get new joint positions (angles) for the robot
        interp_x = x + (target_pos - x) / 10
        q0 = Quaternion(theta)
        q1 = Quaternion(wpt_theta)
        q  = Quaternion.slerp(q0, q1, 0.15) # Rotate 120 degrees (2 * pi / 3)

        if dist_to_goal < 0.04 and wpt_idx < len(waypoints)-1:
            wpt_idx = wpt_idx + 1
            old_wpt_theta = waypoints[wpt_idx-1].theta

        target_joint_positions = p.calculateInverseKinematics(env.robot, 8, interp_x, q.elements) # Go to interpolated positions
        target_joint_positions = target_joint_positions[:7]

        # Get the joint positions (angles) of the robot arm
        joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
        joint_positions = np.array(joint_positions)[:7]

        # Set joint action to be the error between current and target joint positions
        joint_action = (target_joint_positions - joint_positions) * 100
        if np.linalg.norm(joint_action) > 3:
            joint_action = joint_action / np.linalg.norm(joint_action) * 3

        #print(np.linalg.norm(joint_action))

        observation, reward, done, info = env.step(joint_action)
        total_reward += reward

    print('Total episode reward:', total_reward)
