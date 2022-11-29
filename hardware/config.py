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
from log import joint_order


# specific settings for this task
taskId = 1

try:
    if sys.argv[1] == 'nogui':
        gui = False
    else:
        gui = True
except:
    gui = True


### You may want to change the code since here
pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": True,
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
    "fixedBase": True,
    "colored": False
}

verbose = False
debugLine = True


# TODO: Add your code here to start simulation
def transparent():
    for j in sim.joints:
        bullet_simulation.changeVisualShape(sim.robot, sim.jointIds[j], rgbaColor=[1, 1, 1, .5])


ref = [0, 0, 1]
sim = Simulation(pybulletConfigs, robotConfigs, refVect=ref)
transparent()



__joint_limits=None
#init joints_limits

def get_joint_limits(sim):
	global __joint_limits
	if __joint_limits is None:
		joint_infos = [sim.p.getJointInfo(sim.robot,sim.jointIds[name]) for name in joint_order ]
		__joint_limits = [(jInfo[8],jInfo[9]) for jInfo in joint_infos]
	return __joint_limits

