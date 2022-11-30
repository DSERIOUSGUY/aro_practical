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

taskId = 3.2

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
    "cameraSettings": (1.2, 90, -22.8, (-0.12, -0.01, 0.99))
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
    global taleId, cubeId, targetId, obstacle
    finalTargetPos = np.array([0.35, 0.38, 1.0])
    # compile target urdf
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path + "/lib/task_urdfs/task3_2_target_compiled.urdf",
                     abs_path + "/lib/task_urdfs/task3_2_target.urdf"])

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
        fileName=abs_path + "/lib/task_urdfs/cubes/task3_2_dumb_bell.urdf",
        basePosition=[0.5, 0, 1.1],
        baseOrientation=sim.p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=False,
        globalScaling=1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1, 1, 0, 1])

    targetId = sim.p.loadURDF(
        fileName=abs_path + "/lib/task_urdfs/task3_2_target_compiled.urdf",
        basePosition=finalTargetPos,
        baseOrientation=sim.p.getQuaternionFromEuler([0, 0, math.pi / 4]),
        useFixedBase=True,
        globalScaling=1
    )
    obstacle = sim.p.loadURDF(
        fileName=abs_path + "/lib/task_urdfs/cubes/task3_2_obstacle.urdf",
        basePosition=[0.43, 0.275, 0.9],
        baseOrientation=sim.p.getQuaternionFromEuler([0, 0, math.pi / 4]),
        useFixedBase=True,
        globalScaling=1
    )

    for _ in range(300):
        sim.tick()
        time.sleep(1. / 1000)

    return tableId, cubeId, targetId


def move_with_q(target,threshold,vel):
    for j in target:
        for idi, i in enumerate(sim.jointList):
            sim.target_pos[i] = j[idi]
            sim.target_vel[i] = vel
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
            time.sleep(1 / 1000)
            q = np.array([])
            for joint in sim.jointList:
                q = np.append(q, np.array([sim.getJointPos(joint)]), axis=0)
                # print(q)
            counter += 1
            if counter >= 2000:
                print("____NOT REACHED_____", "position=", sim.getJointPosition('RARM_JOINT5'))
                print(list(sim.target_pos.values()) - q)

                break
        print("subtarget reached=", sim.getJointPosition('RARM_JOINT5'), "orientation=",
                sim.getJointOrientation('RARM_JOINT5'))


def move_arms_identical(k,goalOrient,separation=0.2):
    targetL = sim.inverseKinematics('LHAND', k, goalOrient
                                    , 10, 1, 1e-3)
    k[1] -= separation
    targetR = targetL

    for l in range(len(targetL)):
        for m in range(3, 9):
            if (sim.jointRotationAxis[sim.jointList[m]] @ np.array([0, 0, 1]) == 1) or sim.jointRotationAxis[
                sim.jointList[m]] @ np.array([1, 0, 0]) == 1:
                targetR[l][6 + m] = -1 * targetL[l][m]
            else:
                targetR[l][6 + m] = targetL[l][m]
    target = np.array([targetL[0]])
    for i in range(len(targetL)):
        target = np.append(target, np.array([np.hstack((targetL[i][0:9], targetR[i][9:15]))]), axis=0)

    return target


def solution():
    # TODO: Add your code here


    #move arms up
    goals = [[0.40, 0.30, 0.25], [0.50, 0.07, 0.22], [0.45, 0.07, 0.40]]
    threshold = [0.2,0.1,0.1,0.1]
    goalOrient = [[0, 0, 1],[1, 0, 0],[1, 0, 0],[0, 1, 0]]
    for k in range(len(goals)):
        target = move_arms_identical(goals[k],goalOrient[k])
        print("_________target= ", k)
        move_with_q(target,threshold[k],0)
        print("target reached=", k, "position=", sim.getJointPosition('RARM_JOINT5'), "orientation=",
        sim.getJointOrientation('RARM_JOINT5'))
        

    #rotate chest
    q = np.array([])
    for joint in sim.jointList:
        q = np.append(q, np.array([sim.getJointPos(joint)]), axis=0)
    targetChest = np.insert(q[1:15], 0, np.deg2rad(55))

    move_with_q([targetChest],0.1,0.05)

    #move arms down
    # curr_arm_pos = sim.getJointPosition('LARM_JOINT5')
    # curr_arm_pos[2] = 0.18

    goals = [[0.30482338, 0.35085965, 0.23],[0.1, 0.35, 0.23]]
    goalOrient = [[0.54114996, 0.79068419, 0.28631317],[0, 1, 0]]
    separation = [0,0.5]

    #only height decreases
    
 
    for k in range(2):
        target = move_arms_identical(goals[k],goalOrient[k],separation[k])
        print("_________target= ", k)
        move_with_q(target,threshold[k],0)
        print("target reached=", k, "position=", sim.getJointPosition('RARM_JOINT5'), "orientation=",
        sim.getJointOrientation('RARM_JOINT5'))


    print("***FINISHED***")
    for i in range(10):
        sim.tick()

tableId, cubeId, targetId = getReadyForTask()
solution()
time.sleep(5)
