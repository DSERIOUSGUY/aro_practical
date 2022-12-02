import subprocess
import math
import time
import sys
import os
import numpy as np
import pybullet as bullet_simulation
import pybullet_data

# setup paths and load the core
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path
core_path = root_path + '/core'
sys.path.append(core_path)
from Pybullet_Simulation import Simulation

pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": True,  # Ture | False
    "panels": False,  # Ture | False
    "realTime": False,  # Ture | False
    "controlFrequency": 1000,  # Recommand 1000 Hz
    "updateFrequency": 250,  # Recommand 250 Hz
    "gravity": -9.81,  # Gravity constant
    "gravityCompensation": 1.,  # Float, 0.0 to 1.0 inclusive
    "floor": True,  # Ture | False
    "cameraSettings": 'cameraPreset1'  # cameraPreset{1..3},
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains.yaml",
    "robotStartPos": [0, 0, 0.85],  # (x, y, z)
    "robotStartOrientation": [0, 0, 0, 1],  # (x, y, z, w)
    "fixedBase": True,  # Ture | False
    "colored": True  # Ture | False
}

# sim = Simulation_template(pybulletConfigs, robotConfigs)
sim = Simulation(pybulletConfigs, robotConfigs)
print(sim.joints)


def getMotorJointStates(p, robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
    joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques


mpos, mvel, mtorq = getMotorJointStates(bullet_simulation, sim.robot)
res = bullet_simulation.getLinkState(sim.robot,
                                     sim.jointIds['LARM_JOINT5'],
                                     computeLinkVelocity=1,
                                     computeForwardKinematics=1)
link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = res
j_geo, j_rot = bullet_simulation.calculateJacobian(
    sim.robot,
    sim.jointIds['LARM_JOINT5'],
    [0, 0, 0],
    mpos,
    [0.0] * len(mpos),
    [0.0] * len(mpos),
)
print()
for col in j_geo:
    print(col)
    pass
print()
for col in j_rot:
    print(col)
    pass
print()

# print(sim.getTransformationMatrices())

targetPosition = np.array([0.37, 0.23, 1.06385])
#sim.move_without_PD('LARM_JOINT5', targetPosition)

print("jaco= ", sim.jacobianMatrix("LARM_JOINT5"))

print("orient=", sim.getJointOrientation('LARM_JOINT5'))

try:
    time.sleep(float(sys.argv[1]))
except:
    time.sleep(10)
