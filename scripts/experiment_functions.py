#!/usr/bin/env python

import rospy
import sys
import os
from random import seed
from random import random
import math
import tf2_ros
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
from gazebo_msgs.srv import DeleteModel
import move_base_msgs.msg as move
import rospkg 
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState


def reset_obs_pos(n,xstart):
	obs_x = []
	obs_y = []
	for i in range(n):
		obs_x.append(i+xstart)
		obs_y.append(-10)
	return (obs_x,obs_y)


def spawn_obs_init(n,xstart,nstart):
	# Model path
	workspace = os.path.dirname(os.path.realpath(__file__)).replace('model_training/scripts','')
	path_models = workspace+'ahxl_gazebo/gazebo_models/'


	obs_x, obs_y = reset_obs_pos(n,xstart)
	obs_z = 0.5

	# Open sdf files
	f = open(path_models+'cylinder/model.sdf','r')
	cylinder = f.read()
	f = open(path_models+'box/model.sdf','r')
	box = f.read()

	# Set up proxy
	spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
	initial_pose = Pose()

	# Spawn Obs
	for i in range(n):
		initial_pose.position.x = obs_x[i]
		initial_pose.position.y = obs_y[i]
		initial_pose.position.z = obs_z
		rospy.wait_for_service('gazebo/spawn_sdf_model')
		if i <= (n/2):
			spawn_model_prox("cylinder%s"%(i+nstart), cylinder, "robotos_name_space", initial_pose, "world")
		else:
			spawn_model_prox("box%s"%(i+nstart), box, "robotos_name_space", initial_pose, "world")


def move_obstacles(w,obs_dense,d_min,sx,xstart,nstart,idy):
	# Random generator number
	seed(sx)
	max_tries = 100

	# Obstacle amount
	l = 14
	n = int(w*l*obs_dense)
	print("Number of obstacles: " + str(n))

	# Position limits
	w_lim = w - 1.2
	if idy == 0:
		x = [xstart+2, l-1+xstart]
	elif idy == 6:
		[xstart, l-2.5+xstart]
	else:
		x = [xstart, l-1+xstart]
	y = [-w_lim/2, w_lim/2]

	# Cylinder positions
	obs_x = []
	obs_y = []
	for idx in range(n):
		pos_check = True
		tries = 0
		while pos_check:
			xi = random()
			xi = x[0] + (xi * (x[1]-x[0]))
			yi = random()
			yi = y[0] + (yi * (y[1]-y[0]))
			d = []
			if idx == 0:
				break
			for j in range(len(obs_x)):
				d.append(math.sqrt(pow(obs_x[j]-xi,2)+pow(obs_y[j]-yi,2)))
			s = check_space(w,yi)
			if min(d) > d_min and s > 0.85:
				pos_check = False
			else:
				tries += 1
			if tries > max_tries:
				break
		if idx > 0:
			print('tries = %s'%tries)
			print('min(d) = %s'%(min(d)))
		if tries < 100:
			obs_x.append(xi)
			obs_y.append(yi)
	obs_z = 0.5

	# Set up proxy
	set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
	model_pose = ModelState()

	# Move Obstacles
	for i in range(len(obs_x)):
		if i <= n/2:
			model_pose.model_name = 'cylinder%s'%(i+nstart)
		else:
			model_pose.model_name = 'box%s'%(i+nstart)
		model_pose.pose.position.x = obs_x[i]
		model_pose.pose.position.y = obs_y[i]
		model_pose.pose.position.z = obs_z
		rospy.wait_for_service('/gazebo/set_model_state')
		resp = set_state( model_pose )

	print('%s obstacles are moved!'%(len(obs_x)))


def check_space(w,y):
	return 0.5*w-(-abs(y)+0.2)

def check_arrived(listener,goal,goal_tol):
  try:
    trans = listener.lookup_transform('map', 'base_link', rospy.Time(0))
  except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    print ("Could not get TF")
    return False
  x = goal.target_pose.pose.position.x
  y = goal.target_pose.pose.position.y
  if math.sqrt(pow(x-trans.transform.translation.x,2)+pow(y-trans.transform.translation.y,2)) < goal_tol:
    print ("Arrived!!!")
    return True
  else:
  	# print("Not there yet")
  	return False

def spawn_corridor(w,xstart,nstart):
	# Model path
	workspace = os.path.dirname(os.path.realpath(__file__)).replace('model_training/scripts','')
	path_models = workspace+'ahxl_gazebo/gazebo_models/'

	# Open sdf file
	f = open(path_models+'AH_shelf_7_filled/model.sdf','r')
	shelves = f.read()

	# Set up proxy
	spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
	initial_pose = Pose()

	# Spawn shelves
	idx = 1
	s = 1
	yaw = 0
	for i in [0, 7, 6, 13]:
		if idx > 2:
			s = -1
			yaw = math.pi
		initial_pose.position.x = i+xstart
		initial_pose.position.y = s*(w/2 + 0.5)
		initial_pose.position.z = 0
		initial_pose.orientation = Quaternion(*quaternion_from_euler(0,0,yaw))

		rospy.wait_for_service('gazebo/spawn_sdf_model')
		spawn_model_prox("shelves%s"%(idx+nstart), shelves, "robotos_name_space", initial_pose, "world")
		idx += 1
	print('Width of the corridor = %sm'%w)

def reset_robot(x,y,yaw):
	set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
	model_pose = ModelState()
	model_pose.model_name = "/"
	model_pose.pose.position.x = x
	model_pose.pose.position.y = y
	model_pose.pose.position.z = 0
	model_pose.pose.orientation = Quaternion(*quaternion_from_euler(0,0,yaw))
	rospy.wait_for_service('/gazebo/set_model_state')
	resp = set_state( model_pose )

def compute_goal(x,y,yaw):
	goal = move.MoveBaseGoal()
	goal.target_pose.header.frame_id = "map"
	goal.target_pose.pose.position.x = x
	goal.target_pose.pose.position.y = y
	goal.target_pose.pose.orientation = Quaternion(*quaternion_from_euler(0,0,yaw))

	return goal

# def compute_goal(x,y,yaw,listener):
# 	try:
# 		trans = listener.lookup_transform('map', 'base_link', rospy.Time(0))
# 	except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
# 		print ("Could not get TF")
# 		return False
# 	# goal.target_pose.pose.orientation += Quaternion(*quaternion_from_euler(0,0,yaw))
# 	euler = euler_from_quaternion([trans.transform.rotation.x,trans.transform.rotation.y,trans.transform.rotation.z,trans.transform.rotation.w])
# 	yaw0 = euler[2]

	print(trans.transform)
	goal = move.MoveBaseGoal()
	goal.target_pose.header.frame_id = "map"
	goal.target_pose.pose.position.x = trans.transform.translation.x + x*math.cos(yaw0)-y*math.sin(yaw0)
	goal.target_pose.pose.position.y = trans.transform.translation.y + x*math.sin(yaw0)+y*math.cos(yaw0)
	goal.target_pose.pose.orientation = Quaternion(*quaternion_from_euler(0,0,yaw+yaw0))
	print(goal)

	return goal
