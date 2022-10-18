from scipy.spatial.transform import Rotation as npRotation
from scipy.special import comb
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import time
import yaml

from Pybullet_Simulation_base import Simulation_base


class Simulation(Simulation_base):
    """A Bullet simulation involving Nextage robot"""

    def __init__(self, pybulletConfigs, robotConfigs, refVect=None):
        """Constructor
        Creates a simulation instance with Nextage robot.
        For the keyword arguments, please see in the Pybullet_Simulation_base.py
        """
        super().__init__(pybulletConfigs, robotConfigs)
        self.p = self.pybulletConfigs["simulation"]
        if refVect:
            self.refVector = np.array(refVect)
        else:
            self.refVector = np.array([1, 0, 0])

    ########## Task 1: Kinematics ##########
    # Task 1.1 Forward Kinematics
    jointRotationAxis = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint
        # Rotation matrices based on page 11-14 of the lab guide
        'CHEST_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT0': np.array([0, 0, 1]),
        'LARM_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT2': np.array([0, 1, 0]),
        'LARM_JOINT3': np.array([1, 0, 0]),
        'LARM_JOINT4': np.array([0, 1, 0]),
        'LARM_JOINT5': np.array([0, 0, 1]),
        'RARM_JOINT0': np.array([0, 0, 1]),
        'RARM_JOINT1': np.array([0, 1, 0]),
        'RARM_JOINT2': np.array([0, 1, 0]),
        'RARM_JOINT3': np.array([1, 0, 0]),
        'RARM_JOINT4': np.array([0, 1, 0]),
        'RARM_JOINT5': np.array([0, 0, 1]),
        'RHAND': np.array([0, 0, 0]),
        'LHAND': np.array([0, 0, 0])
    }

    frameTranslationFromParent = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint
        # Translational matrices based on page 11-14 of the lab guide
        'CHEST_JOINT0': np.array([0, 0, 0.267]),
        'HEAD_JOINT0': np.array([0, 0, 0.302]),
        'HEAD_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT0': np.array([0.04, 0.135, 0.1015]),
        'LARM_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT2': np.array([0, 0.095, -0.25]),
        'LARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'LARM_JOINT4': np.array([0.1495, 0, 0]),
        'LARM_JOINT5': np.array([0, 0, -0.1335]),
        'RARM_JOINT0': np.array([0.04, -0.135, 0.1015]),
        'RARM_JOINT1': np.array([0, 0, 0.066]),
        'RARM_JOINT2': np.array([0, -0.095, -0.25]),
        'RARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'RARM_JOINT4': np.array([0.1495, 0, 0]),
        'RARM_JOINT5': np.array([0, 0, -0.1335]),
        'RHAND': np.array([0, 0, 0]),  # optional
        'LHAND': np.array([0, 0, 0])  # optional
    }

    jointList = (
        'CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3',
        'LARM_JOINT4', 'LARM_JOINT5', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4',
        'RARM_JOINT5')

    """
    Returns the 3x3 rotation matrix for a joint from the axis-angle representation,
    where the axis is given by the revolution axis of the joint and the angle is theta.
    @ param jointName codename for a joint e.g. CHEST_JOINT0
    @ param theta angle of rotation of the joint
    @ return a 3x3 rotational matrix as a numpy array
    """

    def getJointRotationalMatrix(self, jointName=None, theta=None):
        if jointName is None:
            raise Exception("[getJointRotationalMatrix] \
                Must provide a joint in order to compute the rotational matrix!")
        else:
            try:
                axis = self.jointRotationAxis[jointName]
            except Exception as e:
                print("ERROR:", e, "|", jointName)

            # NOTE: ALL MATRICES ARE WRITTEN IN TRANSPOSE FROM WHAT IS GIVEN IN THE SLIDE 25 of Coordinate_transforms.pdf

            # print("got axis:",axis)

            # z-axis
            if np.array_equal(axis, np.array([0, 0, 1])):
                return np.matrix([
                    [math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1]
                ])

            # y-axis
            elif np.array_equal(axis, np.array([0, 1, 0])):
                return np.matrix([
                    [math.cos(theta), 0, math.sin(theta)],
                    [0, 1, 0],
                    [-math.sin(theta), 0, math.cos(theta)]

                ])

            # x-axis
            elif np.array_equal(axis, np.array([1, 0, 0])):
                return np.matrix([
                    [1, 0, 0],
                    [0, math.cos(theta), -math.sin(theta)],
                    [0, math.sin(theta), math.cos(theta)]
                ])

            else:
                # print("returning no rot")
                return np.matrix([
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ])

    def append_to_array(self, np_arr, element):
        """
            Adds element to np.array (1D), returns combined np.array
        """

        # print("np_arr:",np_arr.tolist())
        np_arr = np_arr.tolist()[0] + [element]
        # print("element:",element,"returning:",np_arr)
        return np.array(np_arr)

    def getTransformationMatrices(self):  # add q for configuration
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
        """
        transformationMatrices = {}
        # TODO modify from here
        # Hint: the output should be a dictionary with joint names as keys and
        # their corresponding homogeneous transformation matrices as values.

        keys = self.jointRotationAxis.keys()

        for i in keys:
            if (i == 'LHAND') or (i == 'RHAND'):
                continue

            theta = self.getJointPos(i)

            rmat = self.getJointRotationalMatrix(i, theta)

            transformationMatrices[i] = np.array([
                self.append_to_array(rmat[0], self.frameTranslationFromParent[i][0]),
                self.append_to_array(rmat[1], self.frameTranslationFromParent[i][1]),
                self.append_to_array(rmat[2], self.frameTranslationFromParent[i][2]),
                np.array([0, 0, 0, 1])
            ])

        return transformationMatrices

    """
        Returns the position and rotation matrix of a given joint using Forward Kinematics
        according to the topology of the Nextage robot.
        @ param jointName name of the joint whose position to return
        @ return pos, orient two numpy arrays, a 3x1 array for the position vector,
         and a 3x3 array for the rotation matrix
    """

    def getJointLocationAndOrientation(self, jointName):  # add q argument for configuration

        # multiply part of the segment by its predecessor only (predecessor will contain other rotation mats)
        # alg :
        #
        # if joint 0, return the trans matrix calc
        # else mult with all prev matrices and return#

        # possible optimization, store rotation values, but need to be careful if it needs to be constantly updated
        # such as when in a trajectory

        name = jointName.split("_")
        joint_class = name[0]
        if joint_class == 'base':
            joint_nr = name[-1]
        elif (joint_class == 'RHAND') or (joint_class == 'LHAND'):
            joint_nr = name[0]
        else:
            joint_nr = int(name[-1][-1])  # only works for 1-digit codes
        tmats = self.getTransformationMatrices()

        if joint_nr == 'base':
            # TODO
            pass
        elif joint_nr == 'RHAND':
            # TODO
            pass
        elif joint_nr == 'LHAND':
            # TODO
            pass
        elif (joint_class == 'CHEST') and (type(joint_nr) == int) and (joint_nr == 0):
            return np.array([tmats[jointName][0, 3],
                             tmats[jointName][1, 3],
                             tmats[jointName][2, 3]]), \
                   np.array([tmats[jointName][0, 0:3],
                             tmats[jointName][1, 0:3],
                             tmats[jointName][2, 0:3]])
        elif ((joint_class == 'LARM') or (joint_class == 'RARM') or (joint_class == 'HEAD')) and (
                type(joint_nr) == int) and (joint_nr == 0):
            trans_mat = np.matmul(tmats['CHEST_JOINT0'], tmats[jointName])
            return np.array([trans_mat[0, 3], trans_mat[1, 3], trans_mat[2, 3]]), \
                   np.array([trans_mat[0, :3],
                             trans_mat[1, :3],
                             trans_mat[2, :3]])
        elif (type(joint_nr) == int) and (joint_nr > 0):
            prev_name = ""
            trans_mat = np.matmul(tmats['CHEST_JOINT0'], tmats[joint_class + "_JOINT0"])
            for i in range(0, joint_nr, 1):
                name = joint_class + "_JOINT" + str(i)
                next_name = joint_class + "_JOINT" + str(i + 1)

                trans_mat = np.matmul(trans_mat, tmats[next_name])

                # #for debugging
                # name = name + "*" + prev_name
                # print("iter:",i,": ","*",prev_name)

            return np.array([trans_mat[0, 3], trans_mat[1, 3], trans_mat[2, 3]]), \
                   np.array([trans_mat[0, :3],
                             trans_mat[1, :3],
                             trans_mat[2, :3]])
        else:
            print("ERROR: didn't recognise joint number")
            return None, None

    def getJointPosition(self, jointName):
        """Get the position of a joint in the world frame, leave this unchanged please."""
        return self.getJointLocationAndOrientation(jointName)[0]

    def getJointOrientation(self, jointName, ref=None):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        if ref is None:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.refVector).squeeze()
        else:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ ref).squeeze()

    def getJointAxis(self, jointName):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.jointRotationAxis[jointName]).squeeze()

    # compute the geometric Jacobian
    # def geomJacobian(jnt2pos, jnt3pos, endEffPos,ai):

    #     endEffPos3d = np.pad(endEffPos,(0, 1), 'constant') #append a 0 on z
    #     col0 = endEffPos3d
    #     col1 = endEffPos3d - np.array(jnt2pos + [0])
    #     col2 = endEffPos3d - np.array(jnt3pos + [0])
    #     J = np.array([np.cross(ai,col0), np.cross(ai,col1), np.cross(ai,col2)]).T 
    #     return J

    """
    Calculate the Jacobian Matrix for the Nextage Robot.
    @param endEffector string id of the endEffector e.g. LARM_JOINT5
    @return 3x15 Jacobian matrix
    """

    def jacobianMatrix(self, endEffector):

        pos = self.getJointPosition('CHEST_JOINT0')
        orient = 0

        joint_class = ""

        end_pos, orient = self.getJointLocationAndOrientation(endEffector)
        col = np.array([np.cross(self.jointRotationAxis['CHEST_JOINT0'], end_pos - pos)])
        tmats = self.getTransformationMatrices()
        for i in tmats.keys():

            name = i.split("_")
            joint_class = name[0]
            if endEffector.find(joint_class) != -1:
                # print("considering:", i)
                pos, orient = self.getJointLocationAndOrientation(i)

                col = np.append(col, np.array([np.cross(self.jointRotationAxis[i], end_pos - pos)]), axis=0)

            elif (joint_class == 'base') or (joint_class == 'CHEST') or (joint_class == 'RHAND') or (
                    joint_class == 'LHAND'):

                # print("skipping:", i)
                continue
            else:
                # print("appending:", i)
                col = np.append(col, np.zeros((1, 3)), axis=0)

        return np.array(col.T)

    def efForwardKinematics(self, endEffector, q):

        keys = self.jointRotationAxis.keys()
        transformationMatrices = {}
        for index, i in enumerate(self.jointList):
            name = i.split("_")
            joint_class = name[0]

            theta = q[index]
            # print('theta=', theta)
            rmat = self.getJointRotationalMatrix(i, theta)
            # print('rmat=', rmat)
            transformationMatrices[i] = np.array([
                self.append_to_array(rmat[0], self.frameTranslationFromParent[i][0]),
                self.append_to_array(rmat[1], self.frameTranslationFromParent[i][1]),
                self.append_to_array(rmat[2], self.frameTranslationFromParent[i][2]),
                np.array([0, 0, 0, 1])
            ])

        name = endEffector.split("_")
        joint_class = name[0]
        if joint_class == 'base':
            joint_nr = name[-1]
        elif (joint_class == 'RHAND') or (joint_class == 'LHAND'):
            joint_nr = name[0]
        else:
            joint_nr = int(name[-1][-1])  # only works for 1-digit codes
        # tmats = self.getTransformationMatrices()
        tmats = transformationMatrices

        if joint_nr == 'base':
            # TODO
            pass
        elif joint_nr == 'RHAND':
            # TODO
            pass
        elif joint_nr == 'LHAND':
            # TODO
            pass
        elif (joint_class == 'CHEST') and (type(joint_nr) == int) and (joint_nr == 0):
            return np.array([tmats[endEffector][0, 3],
                             tmats[endEffector][1, 3],
                             tmats[endEffector][2, 3]]), \
                   np.array([tmats[endEffector][0, 0:3],
                             tmats[endEffector][1, 0:3],
                             tmats[endEffector][2, 0:3]])
        elif ((joint_class == 'LARM') or (joint_class == 'RARM') or (joint_class == 'HEAD')) and (
                type(joint_nr) == int) and (joint_nr == 0):
            trans_mat = np.matmul(tmats['CHEST_JOINT0'], tmats[endEffector])
            return np.array([trans_mat[0, 3], trans_mat[1, 3], trans_mat[2, 3]]), \
                   np.array([trans_mat[0, :3],
                             trans_mat[1, :3],
                             trans_mat[2, :3]])
        elif (type(joint_nr) == int) and (joint_nr > 0):
            prev_name = ""
            trans_mat = np.matmul(tmats['CHEST_JOINT0'], tmats[joint_class + "_JOINT0"])
            for i in range(0, joint_nr, 1):
                name = joint_class + "_JOINT" + str(i)
                next_name = joint_class + "_JOINT" + str(i + 1)

                trans_mat = np.matmul(trans_mat, tmats[next_name])

                # #for debugging
                # name = name + "*" + prev_name
                # print("iter:",i,": ","*",prev_name)

            return np.array([trans_mat[0, 3], trans_mat[1, 3], trans_mat[2, 3]]), \
                   np.array([trans_mat[0, :3],
                             trans_mat[1, :3],
                             trans_mat[2, :3]])
        else:
            print("ERROR: didn't recognise joint number")
            return None, None

    # Task 1.2 Inverse Kinematics

    def inverseKinematics(self, endEffector, targetPosition, orientation, interpolationSteps, maxIterPerStep,
                          threshold):
        """Your IK solver \\
        Arguments: \\
            endEffector: the jointName the end-effector \\
            targetPosition: final destination the the end-effector \\
            orientation: the desired orientation of the end-effector
                         together with its parent link \\
            interpolationSteps: number of interpolation steps
            maxIterPerStep: maximum iterations per step
            threshold: accuracy threshold
        Return: \\
            Vector of x_refs
        """
        # TODO add your code here
        # Hint: return a numpy array which includes the reference angular
        # positions for all joints after performing inverse kinematics.

        # inits

        starting_EFpos, initOrientation = self.getJointLocationAndOrientation(endEffector)

        intermediate_targets = np.linspace(starting_EFpos, targetPosition, interpolationSteps)

        q = np.array([])
        tmats = self.getTransformationMatrices()
        for i in tmats.keys():
            name = i.split("_")
            joint_class = name[0]
            if (joint_class == 'base') or (joint_class == 'RHAND') or (
                    joint_class == 'LHAND'):

                # print("skipping:", i)
                continue
            else:
                q = np.append(q, np.array([self.getJointPos(i)]), axis=0)

        trajectory = np.array([q])  # should contain the current configuration angles
        # need to get matrix of thetas for reaching the final position

        for i in range(interpolationSteps):

            # setting max limit
            # if (i >= maxIterPerStep):
            #    break

            curr_target = intermediate_targets[i, :]

            for iteration in range(maxIterPerStep):
                # dy = curr_target - self.efForwardKinematics(endEffector, q)[0]
                dy = curr_target - self.getJointPosition(endEffector)
                J = self.jacobianMatrix(endEffector)
                dq = np.matmul(np.linalg.pinv(J), dy)

                q = q + dq
                trajectory = np.append(trajectory, np.array([q]), axis=0)

                # TODO: move this part to move without pd
                for idj, j in enumerate(self.jointList):
                    self.p.resetJointState(
                        self.robot, self.jointIds[j], q[idj])

                EF_position = self.efForwardKinematics(endEffector, q)[0]
                if np.linalg.norm(EF_position - curr_target) < threshold:
                    print('target number=', i, 'iteration=', iteration, 'target=', curr_target, 'distance to target=',
                          np.linalg.norm(EF_position - curr_target), 'ef_position=', self.getJointPosition(endEffector),
                          'config=', q)
                    break
                else:
                    print('target number=', i, 'iteration=', iteration, 'target=', curr_target, 'distance to target=',
                          np.linalg.norm(EF_position - curr_target), 'ef_position=', self.getJointPosition(endEffector),
                          'config=', q)

        return trajectory

    def move_without_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
                        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        # TODO add your code here
        # iterate through joints and update joint states based on IK solver
        targetPosition[2] -= 0.85
        trajectory = self.inverseKinematics(endEffector, targetPosition, orientation, 10, maxIter, threshold)
        pltDistance = np.array([])
        # TODO: figure out what the x axis should be
        pltTime = np.arange(11)
        for i in trajectory:
            pltDistance = np.append(pltDistance, np.array(
                [np.linalg.norm(self.efForwardKinematics(endEffector, i)[0] - targetPosition)]), axis=0)

        return pltTime, pltDistance
        pass

    def tick_without_PD(self):
        """Ticks one step of simulation without PD control. """
        # TODO modify from here
        # Iterate through all joints and update joint states.
        # For each joint, you can use the shared variable self.jointTargetPos.

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

    ########## Task 2: Dynamics ##########
    # Task 2.1 PD Controller
    def calculateTorque(self, x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd):
        """ This method implements the closed-loop control \\
        Arguments: \\
            x_ref - the target position \\
            x_real - current position \\
            dx_ref - target velocity \\
            dx_real - current velocity \\
            integral - integral term (set to 0 for PD control) \\
            kp - proportional gain \\
            kd - derivetive gain \\
            ki - integral gain \\
        Returns: \\
            u(t) - the manipulation signal
        """
        # TODO: Add your code here
        pass

    # Task 2.2 Joint Manipulation
    def moveJoint(self, joint, targetPosition, targetVelocity, verbose=False):
        """ This method moves a joint with your PD controller. \\
        Arguments: \\
            joint - the name of the joint \\
            targetPos - target joint position \\
            targetVel - target joint velocity
        """

        def toy_tick(x_ref, x_real, dx_ref, dx_real, integral):
            # loads your PID gains
            jointController = self.jointControllers[joint]
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Start your code here: ###
            # Calculate the torque with the above method you've made
            torque = 0.0
            ### To here ###

            pltTorque.append(torque)

            # send the manipulation signal to the joint
            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )
            # calculate the physics and update the world
            self.p.stepSimulation()
            time.sleep(self.dt)

        targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)

        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [], [], [], [], [], []

        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    def move_with_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
                     threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        # TODO add your code here
        # Iterate through joints and use states from IK solver as reference states in PD controller.
        # Perform iterations to track reference states using PD controller until reaching
        # max iterations or position threshold.

        # Hint: here you can add extra steps if you want to allow your PD
        # controller to converge to the final target position after performing
        # all IK iterations (optional).

        # return pltTime, pltDistance
        pass

    def tick(self):
        """Ticks one step of simulation using PD control."""
        # Iterate through all joints and update joint states using PD control.
        for joint in self.joints:
            # skip dummy joints (world to base joint)
            jointController = self.jointControllers[joint]
            if jointController == 'SKIP_THIS_JOINT':
                continue

            # disable joint velocity controller before apply a torque
            self.disableVelocityController(joint)

            # loads your PID gains
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Implement your code from here ... ###
            # TODO: obtain torque from PD controller
            torque = 0.0  # TODO: fix me
            ### ... to here ###

            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )

            # Gravity compensation
            # A naive gravitiy compensation is provided for you
            # If you have embeded a better compensation, feel free to modify
            compensation = self.jointGravCompensation[joint]
            self.p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.jointIds[joint],
                forceObj=[0, 0, -compensation],
                posObj=self.getLinkCoM(joint),
                flags=self.p.WORLD_FRAME
            )
            # Gravity compensation ends here

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

    ########## Task 3: Robot Manipulation ##########
    def cubic_interpolation(self, points, nTimes=100):
        """
        Given a set of control points, return the
        cubic spline defined by the control points,
        sampled nTimes along the curve.
        """
        # TODO add your code here
        # Return 'nTimes' points per dimension in 'points' (typically a 2xN array),
        # sampled from a cubic spline defined by 'points' and a boundary condition.
        # You may use methods found in scipy.interpolate

        # return xpoints, ypoints
        pass

    # Task 3.1 Pushing
    def dockingToPosition(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005,
                          threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

    # Task 3.2 Grasping & Docking
    def clamp(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005, threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

### END
