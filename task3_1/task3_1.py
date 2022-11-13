import subprocess, math, time, sys, os, numpy as np
import matplotlib.pyplot as plt
import pybullet as bullet_simulation
import pybullet_data
import pandas as pd

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
                     "-o", abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
                     abs_path+"/lib/task_urdfs/task3_1_target.urdf"])

    sim.p.resetJointState(bodyUniqueId=1, jointIndex=12, targetValue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)
    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/table/table_taller.urdf",
        basePosition        = [0.8, 0, 0],
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, math.pi/2]),
        useFixedBase        = True,
        globalScaling       = 1.4
    )
    cubeId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/cubes/cube_small.urdf",
        basePosition        = [0.33, 0, 1.0],
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase        = False,
        globalScaling       = 1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1, 1, 0, 1])

    targetId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
        basePosition        = finalTargetPos,
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, math.pi]),
        useFixedBase        = True,
        globalScaling       = 1
    )
    for _ in range(200):
        print("sending final target:",finalTargetPos)
        # sim.tick(finalTargetPos)
        time.sleep(1./1000)

    return tableId, cubeId, targetId


# finalTargetPos = np.array([0.7, 0.00, 0.91])
def solution():
    # TODO: Add your code here
    global finalTargetPos

    goal = finalTargetPos
    
    #steps:

    #move z
        #interpolate
        #move
    #move y
        #interpolate
        #move
    #move x
        #interpolate
        #move

    # def check_completion_condition():
    #     for i in goal:
    #         if(i != 0):
    #             return False
    #     return True


    # while check_completion_condition() :
    #     for i in range(len(goal)):
    #         print("new goal to reach:",goal[i],"on axis:",i)
    #         #interpolate
    #         Simulation.cubic_interpolation()


tableId, cubeId, targetId = getReadyForTask()

print("IN SIM TICK")
traj = np.linspace([0,0,0],[-1,0.0,-1],100)
delta_pos_plot = []
delta_vel_plot = []
for i in traj:
    del_pos, del_vel = sim.tick(i)
    delta_pos_plot.append(del_pos)
    delta_vel_plot.append(del_vel)

    time.sleep(1/1000)


# print(delta_pos_plot)

delta_pos_df = pd.DataFrame()
delta_vel_df = pd.DataFrame()

for i in delta_pos_plot:
    delta_pos_df = delta_pos_df.append(i,ignore_index=True)

for j in delta_vel_plot:
    delta_vel_df = delta_vel_df.append(j,ignore_index=True)

fig, axes = plt.subplots(nrows=2, ncols=1)

delta_pos_df.plot(ax=axes[0])
delta_vel_df.plot(ax=axes[1])

axes[0].get_legend().remove()
axes[1].legend(loc="lower left")

plt.show()
# solution()