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
from experiment_functions import *

# Environment parameters
w = [4.0,3.0,2.0]
l = 14

obs_dense = [0.125,0.25]
runs = 1
tries_max = 3
d_min = [1.4, 1.4, 2]
sx = 12
# n_max = int(max(w)*l*max(obs_dense))
# Initial pose
x0 = -40
y0 = 0
yaw0 = 0
# Goal pose
x = 14 * len(w) * len(obs_dense) -1
# x = 5
y = 0
yaw = 0
# goal = compute_goal(x,y,yaw)
goal_tol = 1					# in meters

# Experiment name
exp = 'Experiment1'

# Launch class
package = 'simulation_tests'
files =[]
# Robot name
robot = 'boxer'
files.append(['spawn_boxer.launch', 'x:=%s'%x0, 'y:=%s'%y0, 'yaw:=%s'%yaw0])
# Configurations
configs = ['dwa2','teb2']
# files.append(['navigation_dwa1.launch'])
files.append(['navigation_dwa2.launch'])
# files.append(['navigation_teb1.launch'])
files.append(['navigation_teb2.launch'])
# Observers
observers = 'observers'
files.append(['observers.launch'])
# Puth everything in the launch class
launch = launch_class(package, [robot]+configs+[observers], files)

pub = rospy.Publisher('/metrics/log', String, queue_size=10)
pub2 = rospy.Publisher('/metrics/time', Float64, queue_size=10)

fail = False
arrive = False

startpose = PoseWithCovarianceStamped()
startpose.header.frame_id = "map"
startpose.pose.pose.position.x = x0
startpose.pose.pose.position.y = y0
startpose.pose.pose.position.z = 0
startpose.pose.pose.orientation = Quaternion(*quaternion_from_euler(0,0,yaw0))
startpose.pose.covariance = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
reset_pose = rospy.ServiceProxy("/set_pose",SetPose)

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


	# Robot launch
	robot_launch = launch.run(robot)
	robot_launch.start()

	rospy.sleep(2)

	# Observers launch
	observer_launch = launch.run(observers)
	observer_launch.start()

	# transformations
	try:
		print('tfbuffer thing')
		tfBuffer = tf2_ros.Buffer()
		listener = tf2_ros.TransformListener(tfBuffer)
	except:
		pub.publish('error_tf1')


	# Simulation
	for run in range(runs): 
		pub.publish('run=%s, Seed=%s'%(run,sx))
		nstart = 0
		nstart_c = 0
		xstart = -40
		idx = 0
		for wi in w:
			for obs_dense_i in obs_dense:
				n_max = int(wi*l*obs_dense_i)
				spawn_obs_init(n_max,xstart,nstart)
				move_obstacles(wi,obs_dense_i,d_min[idx],sx,xstart,nstart,idx)
				spawn_corridor(wi,xstart,nstart_c)

				print('n_max=%s, n_start=%s w=%s, C=%s, xstart=%s'%(n_max,nstart,wi,obs_dense_i,xstart))

				# Next start values
				nstart += n_max+1
				nstart_c += 10
				xstart += 14
				sx += 1
			idx+=1
		for config in configs:
		# for config in ['teb1']:
			for tries in [1,2,3]:
				# Load launch files
				config_launch = launch.run(config)
				
				arrive = False
				fail = False

				# Launch robot and navigation
				reset_robot(x0,y0,yaw0)
				reset_pose(startpose)

				try:
					print('config start')
					config_launch.start()
				except:
					pub.publish('error_config_launch')
				rospy.sleep(1)

				try:
					print('rosbag1')
					rosbagnode = roslaunch.core.Node("rosbag", "record", name="record", args='-e "/metrics/.*" -O $(find model_training)/results/exp1/%s_%s_%s.bag'%(exp,run,config))
					launchbag = roslaunch.scriptapi.ROSLaunch()
				except:
					pub.publish('error_bag1')

				if run > 0:
					goal = compute_goal(x-x0,y,yaw,tfBuffer)
				else:
					goal = compute_goal(x,y,yaw,tfBuffer)

				try:
					print('bag launch')
					launchbag.start()
					process = launchbag.launch(rosbagnode)
				except:
					pub.publish('error_bag2')

				# Run SimpleActionClient, needed for publishing goals
				try:
					print('client launch')
					client = actionlib.SimpleActionClient('move_base', move.MoveBaseAction)
					client.wait_for_server()
				except:
					pub.publish('error_client_start')

				# Start
				starttime = rospy.get_time()
				try:
					print('send goal')
					client.send_goal(goal)
				except:
					pub.publish('error_client_send_goal')

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
				# Kill robot and navigation config
				try:
					print('kill bag')
					process.stop()
				except:
					pub.publish('error kill bag')
				try:
					print('config shutdown')
					config_launch.shutdown()
				except:
					pub.publish('config3')
				# try:
				# 	print('robot shutdown')
				# 	robot_launch.shutdown()
				# except:
				# 	pub.publish('robot3')


				pub.publish('Config=%s, Try=%s, Time=%s'%(config,tries,duration))

				if arrive or tries == 3:
					break
	observer_launch.shutdown()
	pub.publish('Experiment finished')