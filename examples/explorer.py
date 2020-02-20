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
p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
slider_x = p.addUserDebugParameter("X",-0.2,0.2,0)
slider_y = p.addUserDebugParameter("Y",-0.5,0.1,-0.2)
slider_z = p.addUserDebugParameter("Z",-0.2,0.1,0)
slider_roll = p.addUserDebugParameter("Roll",-4,4,np.pi/2)
slider_pitch = p.addUserDebugParameter("Pitch",-4,4,np.pi/2)
slider_yaw = p.addUserDebugParameter("Yaw",-4,4,np.pi/2)

#env.world_creation.print_joint_info(env.robot)
keys_actions = {p.B3G_LEFT_ARROW: np.array([-0.01, 0, 0]), p.B3G_RIGHT_ARROW: np.array([0.01, 0, 0]), p.B3G_UP_ARROW: np.array([0, 0, 0.01]), p.B3G_DOWN_ARROW: np.array([0, 0, -0.01])}

observation, reward, done, info = env.step(np.zeros(7))

sphere_collision = -1
sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 1])
cup_point = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=[0,0,0], useMaximalCoordinates=False)

triad_id = createTriad()

while True:
    env.render()
    
    x, theta = p.getLinkState(env.robot, 8, computeForwardKinematics=True)[:2] # Get the position and orientation of the end effector
    delta_tool_goal = observation[7:10]
    goal_pos = x - delta_tool_goal
    p.resetBasePositionAndOrientation(cup_point, x + np.array([0, 0.035, 0.09]), [0, 0, 0, 1])

    target_x = p.readUserDebugParameter(slider_x)
    target_y = p.readUserDebugParameter(slider_y)
    target_z = p.readUserDebugParameter(slider_z)
    target_roll = p.readUserDebugParameter(slider_roll)
    target_pitch = p.readUserDebugParameter(slider_pitch)
    target_yaw = p.readUserDebugParameter(slider_yaw)

    target_pos = np.array([target_x, target_y, target_z]) + goal_pos
    target_euler = np.array([target_roll, target_pitch, target_yaw])
    target_quaternion = p.getQuaternionFromEuler(target_euler)
    dist_to_goal = np.linalg.norm(target_pos - x)


    keys = p.getKeyboardEvents()
    if ord('z') in keys and keys[ord('z')]&p.KEY_WAS_TRIGGERED:
        print("[{:.3f},{:.3f},{:.3f}], [{:.3f},{:.3f},{:.3f}]".format(target_x, target_y, target_z, target_roll, target_pitch, target_yaw))

    #createPoseMarker(target_pos, target_quaternion, lifeTime=0.1)
    moveTriad(triad_id, target_pos, target_quaternion)
    
    # IK to get new joint positions (angles) for the robot
    target_joint_positions = p.calculateInverseKinematics(env.robot, 8, target_pos, target_quaternion)
    target_joint_positions = target_joint_positions[:7]

    # Get the joint positions (angles) of the robot arm
    joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
    joint_positions = np.array(joint_positions)[:7]

    # Set joint action to be the error between current and target joint positions
    joint_action = (target_joint_positions - joint_positions) * dist_to_goal * 100
    if np.linalg.norm(joint_action) > 2:
        joint_action = joint_action / np.linalg.norm(joint_action) * 2

    observation, reward, done, info = env.step(joint_action)
