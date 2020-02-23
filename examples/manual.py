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
    parser.add_argument('--nogui', action='store_true', help='Disable GUI')
    parser.add_argument('--train', action='store_true', help='Go into training mode')
    args = parser.parse_args()

    waypoints = []
    if not args.train:
        with open(args.file) as file_in: # Load waypoints file in
            for line in file_in:
                floats = [float(x) for x in line.split(',')]
                waypoints.append(Waypoint([floats[0],floats[1],floats[2]], [floats[3],floats[4],floats[5]]))
                print(floats)

    env = gym.make(args.env)
    env.seed(np.random.randint(0, 1000))
    return env, waypoints, args


env, waypoints, args = ParseArgs()

if not args.nogui:
    env.render()

joint_idxes = {'Jaco':[0,1,2,3,4,5,6], 'Sawyer':[0, 2, 3, 4, 5, 6, 7], 'PR2':range(15, 15+7), 'Baxter':range(1, 8)}
tool_idxes = {'Jaco':8, 'Sawyer':19, 'PR2':54, 'Baxter':26}
initial_eulers = {'Jaco':[1.57,1.57,1.57], 'Sawyer':[1.57,-1.57,1.57], 'PR2':[0,0,0], 'Baxter':[1.57,-1.57,1.57]}

for key in joint_idxes:
    if key in args.env:
        joint_idx = joint_idxes[key]
        tool_idx = tool_idxes[key]
        initial_euler = initial_eulers[key]

cum_reward = 0
iters = 100

for rollout in range(iters):
    observation = env.reset()
    x, theta = p.getLinkState(env.robot, tool_idx, computeForwardKinematics=True)[:2] # Get the position and orientation of the end effector
    #env.world_creation.print_joint_info(env.robot, show_fixed=False)

    wpt_idx = 0
    observation, reward, done, info = env.step(np.zeros(7))
    total_reward = 0


    triad_id = createTriad()
    if args.train and rollout==0:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
        slider_x = p.addUserDebugParameter("X",-0.2,0.2,0)
        slider_y = p.addUserDebugParameter("Y",-0.5,0.1,-0.2)
        slider_z = p.addUserDebugParameter("Z",-0.2,0.1,0)
        slider_roll = p.addUserDebugParameter("Roll",-4,4,initial_euler[0])
        slider_pitch = p.addUserDebugParameter("Pitch",-4,4,initial_euler[1])
        slider_yaw = p.addUserDebugParameter("Yaw",-4,4,initial_euler[2])

    while not done or args.train:
        if not args.nogui:
            env.render()
        
        x, theta = p.getLinkState(env.robot, tool_idx, computeForwardKinematics=True)[:2] # Get the position and orientation of the end effector
        delta_tool_goal = observation[7:10]
        goal_pos = x - delta_tool_goal

        if args.train:
            wpt_x = np.array([p.readUserDebugParameter(slider_x), p.readUserDebugParameter(slider_y), p.readUserDebugParameter(slider_z)])
            target_euler = np.array([p.readUserDebugParameter(slider_roll), p.readUserDebugParameter(slider_pitch), p.readUserDebugParameter(slider_yaw)])
            wpt_theta = p.getQuaternionFromEuler(target_euler)
        else:
            wpt_x = waypoints[wpt_idx].x
            wpt_theta = waypoints[wpt_idx].theta

        target_pos = goal_pos + wpt_x        
        dist_to_goal = np.linalg.norm(target_pos - x)
        moveTriad(triad_id, target_pos, wpt_theta)

        # IK to get new joint positions (angles) for the robot
        interp_x = x + (target_pos - x) / 10
        q0 = Quaternion(theta)
        q1 = Quaternion(wpt_theta)
        q  = Quaternion.slerp(q0, q1, 0.15) # Interpolate quaternion

        if dist_to_goal < 0.04 and wpt_idx < len(waypoints)-1:
            wpt_idx = wpt_idx + 1

        target_joint_positions = p.calculateInverseKinematics(env.robot, tool_idx, interp_x, q.elements) # Go to interpolated positions
        target_joint_positions = np.array(target_joint_positions)[joint_idx]

        # Get the joint positions (angles) of the robot arm
        joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
        joint_positions = np.array(joint_positions)[joint_idx]

        # Set joint action to be the error between current and target joint positions
        joint_action = (target_joint_positions - joint_positions) * 100
        if np.linalg.norm(joint_action) > 3:
            joint_action = joint_action / np.linalg.norm(joint_action) * 3

        #print(np.linalg.norm(joint_action))

        observation, reward, done, info = env.step(joint_action)
        total_reward += reward

        keys = p.getKeyboardEvents()
        if ord('z') in keys and keys[ord('z')]&p.KEY_WAS_TRIGGERED and args.train:
            print("{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(wpt_x[0], wpt_x[1], wpt_x[2], target_euler[0], target_euler[1], target_euler[2]))


    print('Episode', rollout+1, 'reward:', total_reward)
    cum_reward += total_reward

avg_reward = cum_reward/iters
print('Average reward per', iters, 'trials:', avg_reward)