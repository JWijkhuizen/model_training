#!/usr/bin/env python2.7

import rospy
import sys
import actionlib
import tf2_ros
import math
import geometry_msgs.msg
import time
import roslaunch
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Int32, String, Float64
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from robot_localization.srv import SetPose
import rosbag

from launch_class import *
from functions_experiment import *

# Environment parameters
w = [4.0, 3.0]
n_shelf = 2
l = 7*n_shelf

obs_dense = [0.20]
runs = 15
run0 = 0
tries_max = 2
d_min = [1.4, 1.4]
sx = 100
# n_max = int(max(w)*l*max(obs_dense))
# Initial pose
x0 = 0
x0_r = x0 - 2
y0 = 0
yaw0 = 0
# Goal pose
xg = l * len(w) * len(obs_dense) -1
xg_r = xg + 3
yg = 0
yawg = 0
# goal = compute_goal(x,y,yaw)
goal_tol = 1					# in meters
goal = compute_goal(xg_r,yg,yawg)
print(goal)


# Experiment name
exp = 'Experiment7'

# Launch class
package = 'simulation_tests'
files =[]
# Robot name
robot = 'boxer'
# files.append(['spawn_boxer.launch', 'x:=%s'%x0_r, 'y:=%s'%y0, 'yaw:=%s'%yaw0])
# Configurations
configs = ['dwa1','dwa2','teb1']
# configs = ['dwa1','teb1']
files.append(['navigation_dwa1.launch'])
files.append(['navigation_dwa2.launch'])
files.append(['navigation_teb1.launch'])
# files.append(['navigation_teb2.launch'])
# Observers
observers = 'observers'
# files.append(['observers.launch'])
# Puth everything in the launch class
launch_robot = launch_class(package, [robot], [['spawn_boxer.launch', 'x:=%s'%x0_r, 'y:=%s'%y0, 'yaw:=%s'%yaw0]])
launch_movebase = launch_class(package, configs, files)
launch_observers = launch_class(package, [observers], [['observers.launch']])

pub = rospy.Publisher('/metrics/log', String, queue_size=10)
pub2 = rospy.Publisher('/metrics/time', Float64, queue_size=10)

fail = False
arrive = False

delete_robot_client_ = rospy.ServiceProxy("/gazebo/delete_model",DeleteModel)


def status_callback(status):
	global fail
	if len(status.status_list) > 0 and status.status_list[0].status == 4:
		fail = True

if __name__ == '__main__':
	# Init node, needed for roslaunch functionality (not sure why)
	rospy.init_node('experiment_manager', anonymous=True)

	# Listen to /move_base/status
	rospy.Subscriber("/move_base/status", GoalStatusArray, status_callback)
	pub.publish('Experiment start')

	# Delete robot is present
	delete_robot_client_("/")

	# Observers launch
	observer_launch = launch_observers.run(observers)
	observer_launch.start()

	# Spawn init shelves and obstacles
	nstart = 0
	nstart_c = 10
	xstart = x0
	spawn_corridor(w[0],1,x0-7,0)
	for wi in w:
		for obs_dense_i in obs_dense:
			n_max = int(wi*l*obs_dense_i)
			spawn_obs_init(n_max,xstart,nstart)
			spawn_corridor(wi,n_shelf,xstart,nstart_c)
			nstart += n_max+1
			nstart_c += 10
			xstart += l
	spawn_corridor(wi,1,xstart,nstart_c)

	# transformations
	tfBuffer = tf2_ros.Buffer()
	listener = tf2_ros.TransformListener(tfBuffer)

	# Simulation
	for run in range(run0,runs,1): 
		pub.publish('run=%s/%s, Seed=%s'%(run,(runs-1),sx))
		nstart = 0
		nstart_c = 10
		xstart = x0
		idx = 0
		for wi in w:
			for obs_dense_i in obs_dense:
				n_max = int(wi*l*obs_dense_i)
				# Place obstacles randomly
				move_obstacles(wi,obs_dense_i,d_min[idx],sx,xstart,nstart,idx)
				# Next start values
				nstart += n_max+1
				nstart_c += 10
				xstart += 14
				sx += 1
			idx+=1
		
		for config in configs:
			for tries in range(tries_max):
				arrive = False
				fail = False

				# Reset robot and goal
				robot_launch = launch_robot.run(robot)
				robot_launch.start()
				rospy.sleep(1)

				# Load and launch config
				config_launch = launch_movebase.run(config)
				config_launch.start()
				rospy.sleep(1)

				# Bag
				# rosbagnode = roslaunch.core.Node("rosbag", "record", name="record", args='-e "/metrics/.*" -O $(find model_training)/bags/%s_%s_%s.bag'%(exp,run,config))
				# launchbag = roslaunch.scriptapi.ROSLaunch()
				# try:
				# 	launchbag.start()
				# 	process = launchbag.launch(rosbagnode)
				# except:
				# 	pub.publish('error with bag')

				# Run SimpleActionClient, needed for publishing goals
				client = actionlib.SimpleActionClient('move_base', move.MoveBaseAction)
				client.wait_for_server()

				# Start
				starttime = rospy.get_time()
				client.send_goal(goal)

				# Loop untill robot is at the goal
				while not arrive and not fail:
					arrive = check_arrived(tfBuffer,goal,goal_tol)
					rospy.sleep(0.1)
									
				if fail:
					duration = float("nan")
				else:
					endtime = rospy.get_time()
					duration = endtime-starttime
				pub2.publish(duration)

				# Kill things
				# process.stop()
				config_launch.shutdown()
				robot_launch.shutdown()
				delete_robot_client_("/")

				# End of try
				pub.publish('Config=%s, Try=%s, Time=%s'%(config,tries,duration))

				if arrive or tries == tries_max-1:	# This was last try
					break

	# End
	observer_launch.shutdown()
	pub.publish('Experiment finished')