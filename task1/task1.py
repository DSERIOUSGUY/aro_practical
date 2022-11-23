# Please don't submit this file.
# This file is only for helping you while development

import subprocess, math, time, sys, os, numpy as np
from matplotlib import markers
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

ref = [0, 0, 1]
sim = Simulation(pybulletConfigs, robotConfigs, refVect=ref)

# This is an example target position for the left end effector. This target
# position assumes your world frame is located at the base. If your world
# frame is located at the waist, you will need to transform this vector using
# the base_to_waist translation.
endEffector = "RHAND"
targetPosition = np.array([0.4, -0.1, 0.3])  # x,y,z coordinates in world frame
#endEffector = "LHAND"
#targetPosition = np.array(sim.getJointPosition('LARM_JOINT5'))  # x,y,z coordinates in world frame

# #test code
# tmats = sim.getTransformationMatrices()
# print(tmats["LARM_JOINT5"])
# print(sim.getJointLocationAndOrientation("LARM_JOINT5"))
# exit()

# Example code. Feel free to modify
pltTime, pltEFPosition = sim.move_without_PD(endEffector, targetPosition, speed=0.01, orientation=[0, 1, 0],
                                             threshold=1e-2, maxIter=3000, debug=False, verbose=False)

# Now plot some graphs
task1_figure_name = "task1_kinematics.png"
task1_savefig = True
# ...

fig = plt.figure(figsize=(6, 4))

plt.plot(pltTime, pltEFPosition, color='blue', marker="x")
plt.xlabel("Time s")
plt.ylabel("Distance to target position")

plt.suptitle("task1 IK without PD", size=16)
plt.tight_layout()
plt.subplots_adjust(left=0.15)

if task1_savefig:
    fig.savefig(task1_figure_name)
plt.show()
