import scipy.spatial
import numpy as np
import math
import time

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
        for idi, i in enumerate(self.jointList):
            self.target_pos[i] = 0
            self.target_vel[i] = 0

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
                return np.matrix([
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ])

    def append_to_array(self, np_arr, element):
        """
            Adds element to np.array (1D), returns combined np.array
        """
        np_arr = np_arr.tolist()[0] + [element]
        return np.array(np_arr)

    def getTransformationMatrices(self, q=None):  # add q for configuration
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
        """
        transformationMatrices = {}

        keys = self.jointRotationAxis.keys()

        for i in self.jointList:
            if (i == 'LHAND') or (i == 'RHAND'):
                continue

            if q is None:

                theta = self.getJointPos(i)
            else:
                theta = q[self.jointList.index(i)]
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

    def getJointLocationAndOrientation(self, jointName, q=None):

        name = jointName.split("_")
        joint_class = name[0]
        if joint_class == 'base':
            joint_nr = name[-1]
        elif (joint_class == 'RHAND') or (joint_class == 'LHAND'):
            joint_nr = name[0]
        else:
            joint_nr = int(name[-1][-1])  # only works for 1-digit codes
        tmats = self.getTransformationMatrices(q)

        if joint_nr == 'base':
            pass
        elif joint_nr == 'RHAND':
            pass
        elif joint_nr == 'LHAND':
            pass
        # from transformation matrix, get rotational matrix
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
            # consider chest movement if needed
            trans_mat = np.matmul(tmats['CHEST_JOINT0'], tmats[joint_class + "_JOINT0"])
            for i in range(0, joint_nr, 1):
                name = joint_class + "_JOINT" + str(i)
                next_name = joint_class + "_JOINT" + str(i + 1)

                # compute transformation matrix for the current joint if it is not
                # at base(i.e position and orientation is dependent on another joint)
                trans_mat = np.matmul(trans_mat, tmats[next_name])

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

    """
    Calculate the Jacobian Matrix for the Nextage Robot.
    @param endEffector string id of the endEffector e.g. LARM_JOINT5
    @return 3x15 Jacobian matrix
    """

    def jacobianMatrix(self, endEffector, q=None):

        pos, orient = self.getJointLocationAndOrientation('CHEST_JOINT0', q)
        orient = 0

        joint_class = ""
        aeff = self.getJointLocationAndOrientation(endEffector)[1] @ [1, 0, 0]

        end_pos, orient = self.getJointLocationAndOrientation(endEffector, q)
        col = np.array([np.cross(self.jointRotationAxis['CHEST_JOINT0'], end_pos - pos)])
        orientCol = np.array([np.cross(self.jointRotationAxis['CHEST_JOINT0'], aeff)])
        tmats = self.getTransformationMatrices()

        for i in tmats.keys():
            # identify if the joint needs to be computed for
            name = i.split("_")
            joint_class = name[0]
            if endEffector.find(joint_class) != -1:
                pos, orient = self.getJointLocationAndOrientation(i, q)

                # calculate jacobian for postion and orientation
                col = np.append(col, np.array([np.cross(self.jointRotationAxis[i], end_pos - pos)]), \
                                axis=0)
                orientCol = np.append(orientCol, np.array(
                    [np.cross(self.jointRotationAxis[i], aeff)]), axis=0)


            elif (joint_class == 'base') or (joint_class == 'CHEST') or (joint_class == 'RHAND') or (
                    joint_class == 'LHAND'):
                # skip for parts that are not moved in any case
                continue
            else:
                col = np.append(col, np.zeros((1, 3)), axis=0)
                orientCol = np.append(orientCol, np.zeros((1, 3)), axis=0)
        return np.vstack((col.T, orientCol.T))



    # Task 1.2 Inverse Kinematics

    def inverseKinematics(self, Effector, targetPosition, orientation, interpolationSteps, maxIterPerStep,
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

        # inits
        if Effector == 'LHAND':
            endEffector = 'LARM_JOINT5'
        elif Effector == 'RHAND':
            endEffector = 'RARM_JOINT5'
        else:
            endEffector = Effector

        starting_EFpos, initOrientation = self.getJointLocationAndOrientation(endEffector)

        # get trajectory for position and orientation
        intermediate_targets = np.linspace(starting_EFpos, targetPosition, interpolationSteps)
        intermediate_orientations = scipy.spatial.geometric_slerp(initOrientation @ [1, 0, 0], orientation,
                                                                  np.linspace(0, 1, interpolationSteps))

        q = np.array([])
        tmats = self.getTransformationMatrices()

        # build robot configuration vector
        for i in tmats.keys():
            name = i.split("_")
            joint_class = name[0]
            if (joint_class == 'base') or (joint_class == 'RHAND') or (
                    joint_class == 'LHAND'):
                continue
            else:
                q = np.append(q, np.array([self.getJointPos(i)]), axis=0)

        # contains the current configuration angles as starting point
        trajectory = np.array([q])

        # generate trajectory
        for i in range(interpolationSteps):

            curr_target = intermediate_targets[i, :]
            curr_target_orientation = intermediate_orientations[i]
            for iteration in range(maxIterPerStep):
                dy = curr_target - self.getJointLocationAndOrientation(endEffector, q)[0]
                dtheta = (curr_target_orientation - (
                        self.getJointLocationAndOrientation(endEffector, q)[1] @ [1, 0, 0]))
                dy = np.hstack((dy, dtheta))

                J = self.jacobianMatrix(endEffector, q)  # get Jacobian
                # if end-effector is set to L or RHAND then exclude the chest from the kinematic chain
                if (Effector == 'RHAND') or (
                        Effector == 'LHAND'):
                    J[:, 0] = np.zeros(6)

                # get new configuration
                dq = np.matmul(np.linalg.pinv(J), dy)
                q = q + dq

                # append to trajectory
                trajectory = np.append(trajectory, np.array([q]), axis=0)

                # estimate new EF position via forward kinematics
                EF_position = self.getJointLocationAndOrientation(endEffector, q)[0]
                if np.linalg.norm(EF_position - curr_target) < threshold:
                    break
                else:
                    pass

        return trajectory

    def move_without_PD(self, Effector, targetPosition, speed=0.01, orientation=None,
                        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """

        # iterate through joints and update joint states based on IK solver
        if Effector == 'LHAND':
            endEffector = 'LARM_JOINT5'
        elif Effector == 'RHAND':
            endEffector = 'RARM_JOINT5'
        else:
            endEffector = Effector

        # account for base height
        targetPosition[2] -= 0.85
        # generate trajectory to reach goal
        trajectory = self.inverseKinematics(Effector, targetPosition, orientation, 10, maxIter, threshold)
        pltDistance = []
        pltTime = []
        initTime = time.time()
        for i in trajectory:
            # for every subtarget, get delta q for all joints
            for idj, j in enumerate(self.jointList):
                self.p.resetJointState(
                    self.robot, self.jointIds[j], i[idj])

            # record the distance from the target and time
            pltDistance.append(np.linalg.norm(self.getJointLocationAndOrientation(endEffector, i)[0] - targetPosition))
            pltTime.append(time.time() - initTime)

        pltDistance = np.array(pltDistance)
        pltTime = np.array(pltTime)

        return pltTime, pltDistance
        pass

    def tick_without_PD(self):
        """Ticks one step of simulation without PD control. """
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

        u = kp * (x_ref - x_real) + kd * (dx_ref - dx_real)

        return u

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

            # Calculate the torque
            torque = self.calculateTorque(x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd)
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

        joint_pos = self.getJointPos(joint)
        prev_joint_pos = self.getJointPos(joint)

        dist = targetPosition - joint_pos
        dist_remaining = dist

        joint_vel = (joint_pos - prev_joint_pos) / self.dt
        threshold = 0.035

        while abs(dist_remaining) > abs(threshold) or abs(joint_vel - targetVelocity) > abs(threshold):
            toy_tick(targetPosition, joint_pos, targetVelocity, joint_vel, 0)

            prev_joint_pos = joint_pos
            joint_pos = self.getJointPos(joint)
            joint_vel = (joint_pos - prev_joint_pos) / self.dt

            dist_remaining = targetPosition - joint_pos

            pltTime.append(time.time())
            pltPosition.append(joint_pos)
            pltVelocity.append(joint_vel)
            pltTarget.append(targetPosition)
            pltTorqueTime.append(time.time())

        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    # variables to contain persistent values between iterations of tick
    prev_joint_pos = {}
    target_pos = {}
    target_vel = {}

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

            x_ref = self.target_pos[joint]  # target pos
            dx_ref = self.target_vel[joint]  # target vel
            x_real = self.getJointPos(joint)  # current pos

            if not (joint in self.prev_joint_pos.keys()):
                # if joint not encountered previously, make an entry for it
                self.prev_joint_pos[joint] = x_real

            dx_real = (x_real - self.prev_joint_pos[joint]) / self.dt  # current speed
            self.prev_joint_pos[joint] = x_real

            torque = self.calculateTorque(x_ref, x_real, dx_ref, dx_real, 0, kp, ki, kd)

            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )

            # Gravity compensation
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
    # implemented in template solution() methods

### END
