import gym, sys, argparse
import numpy as np
import assistive_gym
import pybullet as p
import time

def createPoseMarker(position=np.array([0,0,0]),
                     orientation=np.array([0,0,0,1]),
                     text="",
                     xColor=np.array([1,0,0]),
                     yColor=np.array([0,1,0]),
                     zColor=np.array([0,0,1]),
                     textColor=np.array([0,0,0]),
                     lineLength=0.06,
                     lineWidth=5,
                     textSize=1,
                     textPosition=np.array([0,0,0.1]),
                     textOrientation=None,
                     lifeTime=0.3,
                     parentObjectUniqueId=-1,
                     parentLinkIndex=-1,
                     physicsClientId=0):
    '''Create a pose marker that identifies a position and orientation in space with 3 colored lines.
    '''
    pts = np.array([[0,0,0],[lineLength,0,0],[0,lineLength,0],[0,0,lineLength]])
    rotIdentity = np.array([0,0,0,1])
    po, _ = p.multiplyTransforms(position, orientation, pts[0,:], rotIdentity)
    px, _ = p.multiplyTransforms(position, orientation, pts[1,:], rotIdentity)
    py, _ = p.multiplyTransforms(position, orientation, pts[2,:], rotIdentity)
    pz, _ = p.multiplyTransforms(position, orientation, pts[3,:], rotIdentity)
    p.addUserDebugLine(po, px, xColor, lineWidth, lifeTime)#, parentObjectUniqueId, parentLinkIndex, physicsClientId)
    p.addUserDebugLine(po, py, yColor, lineWidth, lifeTime)#, parentObjectUniqueId, parentLinkIndex, physicsClientId)
    p.addUserDebugLine(po, pz, zColor, lineWidth, lifeTime)#, parentObjectUniqueId, parentLinkIndex, physicsClientId)
    if len(text) == 0:
        return

    if textOrientation is None:
        textOrientation = orientation
    p.addUserDebugText(text, [0,0,0.1],textColorRGB=textColor,textSize=textSize,
                       parentObjectUniqueId=parentObjectUniqueId,
                       parentLinkIndex=parentLinkIndex,
                       physicsClientId=physicsClientId)

def accurateIK(bodyId, endEffectorId, targetPosition, targetOrientation, lowerLimits, upperLimits, jointRanges, restPoses, 
               useNullSpace=False, maxIter=10, threshold=1e-4):
    """
    Parameters
    ----------
    bodyId : int
    endEffectorId : int
    targetPosition : [float, float, float]
    lowerLimits : [float] 
    upperLimits : [float] 
    jointRanges : [float] 
    restPoses : [float]
    useNullSpace : bool
    maxIter : int
    threshold : float

    Returns
    -------
    jointPoses : [float] * numDofs
    """
    closeEnough = False
    iter = 0
    dist2 = 1e30

    numJoints = p.getNumJoints(bodyId)

    while (not closeEnough and iter<maxIter):
        if useNullSpace:
            jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition,
                lowerLimits=lowerLimits, upperLimits=upperLimits, jointRanges=jointRanges, 
                restPoses=restPoses)
        else:
            jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition, targetOrientation)
    
        # for i in range(numJoints): 
        #     jointInfo = p.getJointInfo(bodyId, i)
        #     qIndex = jointInfo[3]
        #     if qIndex > -1:
        #         p.resetJointState(bodyId,i,jointPoses[qIndex-7]) #Somehow this actually SETS the robot position


        ls = p.getLinkState(bodyId,endEffectorId)    
        newPos = ls[4]
        diff = [targetPosition[0]-newPos[0],targetPosition[1]-newPos[1],targetPosition[2]-newPos[2]]
        dist2 = np.sqrt((diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]))
        #print("dist2=",dist2)
        closeEnough = (dist2 < threshold)
        iter=iter+1
    #print("iter=",iter)
    np.set_printoptions(precision=3)
    jointPoses = jointPoses[0:7]
    #print('joint poses', np.array(jointPoses), len(jointPoses))
    return np.array(jointPoses)