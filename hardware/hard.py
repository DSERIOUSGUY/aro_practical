#! /usr/bin/env python

import rospy 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import random
from pickle import load
import sys
import time

print("Moving to out")
global file_name, joint_pos

def trajectory():

	file = open(file_name, 'rb')
	configs = load(file)
	file.close()
	return configs

def my_publisher():
    global joint_pos
    # control part

    rospy.init_node('rrr_control_python_node')
    control_publisher = rospy.Publisher('/nextagea/trajectory_controller/command', JointTrajectory, queue_size=10)
    first = True
    
    j_names = [ 'CHEST_JOINT0',
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
	
    while not rospy.is_shutdown():
    #if first:
#    	if first:
        msg = JointTrajectory()

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ''
        msg.joint_names = j_names
        if first:    
            for i, pos in enumerate(joint_pos):

                point = JointTrajectoryPoint()
                point.positions = pos
                point.velocities = []
                point.accelerations = []
                point.effort = []

                msg.points.append( point )                
                point.time_from_start = rospy.Duration(1. + float(i)*0.1)
                control_publisher.publish( msg )
                time.sleep(0.1)

        #control_publisher.publish( msg )
        rospy.loginfo( msg ) 
        first = False


if __name__ == '__main__':
    global file_name, joint_pos
    file_name = sys.argv[1]
    joint_pos = trajectory()
    print(joint_pos[-1])
	
    my_publisher()


