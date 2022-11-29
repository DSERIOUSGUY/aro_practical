from config import *
import log
from methods import isFeasible


#TODO load simulation with self collision




def testGenData():
	targets = np.array([[0.5, -0.2, 1.1], [0.2, -0.5, 1.1]])
	ori = np.array([[0, np.sqrt(2)/2, np.sqrt(2)/2], [0, -np.sqrt(2)/2, np.sqrt(2)/2]])
	sim.move_without_PD("LARM_JOINT5", targets[0], speed=0.9, orientation=np.array([0, 0, 1]),
                                          threshold=10e-3, maxIter=1000, debug=False, verbose=False)
	log.save_trajectory()


def loadTrajAndValidate():
	traj = log.trajectory()
	traj_ok = isFeasible(sim,traj)
	if traj_ok:
		print ("Trajectory is valid")
	else:
		print ("Trajectory is not valid")
	return traj_ok
	
	
if __name__ == "__main__":
	# ~ testGenData() #comment for own data
	loadTrajAndValidate()

	traj = log.trajectory()
	log.save_trajectory(traj)
	
	# ~ q = [0.0,
 # ~ 0.0,
 # ~ -0.17626101355903293,
 # ~ 0.0,
 # ~ 0.0,
 # ~ -10,
 # ~ -10,
 # ~ 10,
 # ~ 3e-18,
 # ~ 0.2395955755943071,
 # ~ 0.0,
 # ~ 0.0,
 # ~ 0.0,
 # ~ 0.0,
 # ~ 0.0,
 # ~ 0.0,
 # ~ 0.0]

	
	# ~ print("following trajectory should not be valid")
	# ~ isFeasible(sim,traj+[q]) 
	#should fail
	
