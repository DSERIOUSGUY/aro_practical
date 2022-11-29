import numpy as np
from log import trajectory, set_config, joint_order
from config import get_joint_limits


#TODO load urdf with collisions


	
def respectsLimits(sim,q):	
	ok = [ q_i <= ul_i and q_i >= ll_i for (q_i, (ll_i, ul_i)) in zip(q,get_joint_limits(sim))]
	if all(ok):
		return True
	print ("joint limit violations")
	for (name, val, qi, lim) in zip(joint_order, ok, q, get_joint_limits(sim)):
		if not val:
			print("\t " + name + " joint limit violation: " + str(qi) + "not in range [ " + str(lim[0]) + ", " +  str(lim[1]) + "]"  )
	return False
	
	
def firstAndLatIsZero(sim, trajectory):
	norm = np.linalg.norm(trajectory[0])
	if (norm > 0.0001):
		print ('init position is not at 0')
		return False
	norm = np.linalg.norm(trajectory[-1])
	if (norm > 0.0001):
		print ('Final position is not at 0')
		return False
	return True
	

def isColliding(sim, q):
	contact_a = sim.p.getContactPoints()
	contact_ids = set(item[2] for item in contact_a if item[2] in [sim.robot])
	return len(contact_ids) == 1



from time import sleep
	
def isFeasible(sim, trajectory):
	ret = firstAndLatIsZero(sim, trajectory)
	if len(trajectory) < 200:
		ret = False 
		print ("Consider a smaller interpolation step, aim for more than 100 configurations in the path")
	for i, q in enumerate(trajectory):
		set_config(sim,q)
		sim.p.stepSimulation()
		sleep(0.01)
		if not respectsLimits(sim,q):
			print ("at configuration ", i)
			ret = False
		if isColliding(sim, q):
			print ("collision at configuration ", i)
			
	return ret
