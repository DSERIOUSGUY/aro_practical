import numpy as np
from numpy import array
from math import pi
from pickle import dump, load

NUM_DOFS = 15

configs = None

joint_order = [ 'CHEST_JOINT0',
				'HEAD_JOINT0',
				'HEAD_JOINT1',
				'LARM_JOINT0',
				'LARM_JOINT1',
				'LARM_JOINT2',
				'LARM_JOINT3',
				'LARM_JOINT4',
				'LARM_JOINT5',
				'RARM_JOINT0',
				'RARM_JOINT1',
				'RARM_JOINT2',
				'RARM_JOINT3',
				'RARM_JOINT4',
				'RARM_JOINT5']

def mod_pi(qi):
	return abs(qi) % pi * np.sign(qi) 

def clamp(q):
	return q
	# ~ return [mod_pi(qi) for qi in q]

def get_config(sim):
	#adding eps to first config for amending num. error
	q = clamp([sim.getJointPos(j) for j in joint_order]);
	return q
	
def set_config(s, q):
	[s.p.resetJointState(s.robot, s.jointIds[joint], q[i]) for (i,joint) in enumerate(joint_order)]

def log(s):
	global configs
	if configs is None:
		configs = []
	configs += [get_config(s)]

def save_trajectory(traj=None):
	if traj is None:
		traj = trajectory()
	global configs
	traj2 = traj[:]
	traj2.reverse()
	traj+=traj2
	file = open('traj', 'wb')
	dump(traj, file)
	file.close()
	configs = None

	
def trajectory():
	global configs
	if configs is None:
		file = open('traj', 'rb')
		configs = load(file)
		file.close()
	return configs
