#!/usr/bin/env python2.7

import rospy
import sys
import actionlib
import tf2_ros
import math
import geometry_msgs.msg
import time

from tf.transformations import quaternion_from_euler
from std_msgs.msg import Int32, String, Float64
from actionlib_msgs.msg import GoalStatusArray
import move_base_msgs.msg as move
from geometry_msgs.msg import Quaternion


def status_callback(status):
	global fail
	if len(status.status_list) > 0 and status.status_list[0].status == 4:
		fail = True

def compute_goal(x,y,yaw):
	goal = move.MoveBaseGoal()
	goal.target_pose.header.frame_id = "map"
	goal.target_pose.pose.position.x = x
	goal.target_pose.pose.position.y = y
	goal.target_pose.pose.orientation = Quaternion(*quaternion_from_euler(0,0,yaw))

	return goal

def check_arrived(listener,goal,goal_tol):
  try:
    trans = listener.lookup_transform('map', 'base_link', rospy.Time(0))
  except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    # print ("Could not get TF")
    return False
  x = goal.target_pose.pose.position.x
  y = goal.target_pose.pose.position.y
  if math.sqrt(pow(x-trans.transform.translation.x,2)+pow(y-trans.transform.translation.y,2)) < goal_tol:
    # print("Arrived!!!")
    return True
  else:
  	# print("Not there yet")
  	return False

if __name__ == '__main__':
	# Init node, needed for roslaunch functionality (not sure why)
	rospy.init_node('navigation_manager', anonymous=True)


	if len(sys.argv) != 4:
		print('Not enough input arguments')

	# rospack = RosPack()
	rospy.sleep(6.0)

	# Arguments
	xg = float(sys.argv[1])
	yg = float(sys.argv[2])
	yawg = float(sys.argv[3])
	goal_tol = 1					# in meters
	goal = compute_goal(xg,yg,yawg)

	# transformations
	tfBuffer = tf2_ros.Buffer()
	listener = tf2_ros.TransformListener(tfBuffer)

	# Run SimpleActionClient, needed for publishing goals
	client = actionlib.SimpleActionClient('move_base', move.MoveBaseAction)
	client.wait_for_server()

	# Check if move_base = ready

	# Start
	starttime = rospy.get_time()
	client.send_goal(goal)

	# Loop untill robot is at the goal
	arrive = False
	fail = False
	while not arrive and not fail:
		arrive = check_arrived(tfBuffer,goal,goal_tol)
		# print(rospy.get_time()-starttime)
		if rospy.get_time()-starttime > 60:
			fail = True
		rospy.sleep(0.2)