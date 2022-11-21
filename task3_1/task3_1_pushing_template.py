import subprocess, math, time, sys, os, numpy as np
import matplotlib.pyplot as plt
import pybullet as bullet_simulation
import pybullet_data

# setup paths and load the core
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path + '/..'
core_path = root_path + '/core'
sys.path.append(core_path)
from Pybullet_Simulation import Simulation

# specific settings for this task

taskId = 3.1

try:
    if sys.argv[1] == 'nogui':
        gui = False
    else:
        gui = True
except:
    gui = True

pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": gui,
    "panels": False,
    "realTime": False,
    "controlFrequency": 1000,
    "updateFrequency": 250,
    "gravity": -9.81,
    "gravityCompensation": 1.,
    "floor": True,
    "cameraSettings": (1.07, 90.0, -52.8, (0.07, 0.01, 0.76))
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains.yaml",
    "robotStartPos": [0, 0, 0.85],
    "robotStartOrientation": [0, 0, 0, 1],
    "fixedBase": False,
    "colored": False
}

sim = Simulation(pybulletConfigs, robotConfigs)


##### Please leave this function unchanged, feel free to modify others #####
def getReadyForTask():
    global finalTargetPos
    # compile urdfs
    finalTargetPos = np.array([0.7, 0.00, 0.91])
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path + "/lib/task_urdfs/task3_1_target_compiled.urdf",
                     abs_path + "/lib/task_urdfs/task3_1_target.urdf"])

    sim.p.resetJointState(bodyUniqueId=1, jointIndex=12, targetValue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)
    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName=abs_path + "/lib/task_urdfs/table/table_taller.urdf",
        basePosition=[0.8, 0, 0],
        baseOrientation=sim.p.getQuaternionFromEuler([0, 0, math.pi / 2]),
        useFixedBase=True,
        globalScaling=1.4
    )
    cubeId = sim.p.loadURDF(
        fileName=abs_path + "/lib/task_urdfs/cubes/cube_small.urdf",
        basePosition=[0.33, 0, 1.0],
        baseOrientation=sim.p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=False,
        globalScaling=1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1, 1, 0, 1])

    targetId = sim.p.loadURDF(
        fileName=abs_path + "/lib/task_urdfs/task3_1_target_compiled.urdf",
        basePosition=finalTargetPos,
        baseOrientation=sim.p.getQuaternionFromEuler([0, 0, math.pi]),
        useFixedBase=True,
        globalScaling=1
    )
    for _ in range(200):
        sim.tick()
        time.sleep(1. / 1000)

    return tableId, cubeId, targetId


def solution():
    goals = [[0.4, -0.1, 0.15], [0.2, -0.10, 0.15], [0.2, 0.02, 0.15], [0.2, 0.0, 0.15], [0.4, 0.0, 0.09], [0.60, 0.0, 0.09]]
    threshold = 0.1
    goalOrient = [1, 0, 0]
    for k in goals:
        target = sim.inverseKinematics('RARM_JOINT5', k, goalOrient
                                       , 10, 100, 1e-3)
        print("target= ", k)
        for j in target:
            print("subtarget= ", sim.efForwardKinematics('RARM_JOINT5', j)[0])
            for idi, i in enumerate(sim.jointList):
                sim.target_pos[i] = j[idi]
                sim.target_vel[i] = 0.0001
            q = np.array([])
            for joint in sim.jointList:

                q = np.append(q, np.array([sim.getJointPos(joint)]), axis=0)
                # print(q)
            counter = 0
            while np.amax(np.absolute(q - j)) > threshold:
                # while np.linalg.norm(
                #        sim.efForwardKinematics('RARM_JOINT5', q)[0] - sim.efForwardKinematics('RARM_JOINT5', j)[
                #            0]) > threshold:
                sim.tick()
                time.sleep(1. / 1000)
                q = np.array([])
                for joint in sim.jointList:
                    q = np.append(q, np.array([sim.getJointPos(joint)]), axis=0)
                    # print(q)
                counter += 1
                if counter >= 2000:
                    print("____NOT REACHED_____", "position=", sim.getJointPosition('RARM_JOINT5'))
                    print(list(sim.target_pos.values())-q)

                    break
            print("subtarget reached=", sim.getJointPosition('RARM_JOINT5'), "orientation=",
                  sim.getJointOrientation('RARM_JOINT5'))
        print("target reached=", k, "position=", sim.getJointPosition('RARM_JOINT5'), "orientation=",
              sim.getJointOrientation('RARM_JOINT5'))


tableId, cubeId, targetId = getReadyForTask()
solution()
time.sleep(5)
