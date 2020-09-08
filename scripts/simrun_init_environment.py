#!/usr/bin/env python

import sys
import os

import rospy
import rospkg

import math
from random import seed, random

from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose, Quaternion
from gazebo_msgs.srv import DeleteModel, SetModelState
from gazebo_msgs.msg import ModelState 
import move_base_msgs.msg as move
from tf.transformations import quaternion_from_euler, euler_from_quaternion



# Command line args:
# 	w_cor		corridor width
# 	l_cor		corridor length 
# 	C			clutterness
# 	sx 			seed number


def reset_obs_pos(n,xstart):
	obs_x = []
	obs_y = []
	for i in range(n):
		obs_x.append(i+xstart)
		obs_y.append(-10)
	return (obs_x,obs_y)


def spawn_obstacles(n,xstart,nstart):
	# Model path
	rospack = rospkg.RosPack()
	path = rospack.get_path('ahxl_gazebo')
	path_models = path + '/gazebo_models/'

	# Open sdf files
	f = open(path_models+'cylinder/model.sdf','r')
	cylinder = f.read()
	f = open(path_models+'box/model.sdf','r')
	box = f.read()

	# Set up proxy
	spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
	initial_pose = Pose()

	# Spawn Obs
	ido = nstart
	obs_x, obs_y = reset_obs_pos(n,xstart)
	for i in range(n):
		initial_pose.position.x = obs_x[i]
		initial_pose.position.y = obs_y[i]
		initial_pose.position.z = 0.5
		rospy.wait_for_service('gazebo/spawn_sdf_model')
		if (i % 2) == 0:
			spawn_model_prox("obstacle%s"%(ido), cylinder, "robotos_name_space", initial_pose, "world")
		else:
			spawn_model_prox("obstacle%s"%(ido), box, "robotos_name_space", initial_pose, "world")
		ido += 1

	return ido

def spawn_shelves(w,n,xstart,nstart):
	# Model path
	rospack = rospkg.RosPack()
	path = rospack.get_path('ahxl_gazebo')
	path_models = path + '/gazebo_models/'

	# Open sdf file
	f = open(path_models+'AH_shelf_7_filled/model.sdf','r')
	shelves = f.read()

	# Set up proxy
	spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
	initial_pose = Pose()

	# Spawn shelves
	x = [0, 6, 7, 13, 14, 20, 21, 27, 28, 34, 35, 41]
	ids = nstart
	for i in range(2*n):
		if (i % 2) == 0:
			s = 1
			yaw = 0
		else:
			s = -1
			yaw = math.pi
		initial_pose.position.x = x[i]+xstart
		initial_pose.position.y = s*(w/2 + 0.5)
		initial_pose.position.z = 0
		initial_pose.orientation = Quaternion(*quaternion_from_euler(0,0,yaw))

		rospy.wait_for_service('gazebo/spawn_sdf_model')
		spawn_model_prox("shelves%s"%(ids), shelves, "robotos_name_space", initial_pose, "world")
		ids += 1
	# print('Width of the corridor = %sm'%w)

	return ids


if __name__ == '__main__':
	if len(sys.argv) != 4:
		print('Not enough input arguments')
		# break

	# Initialize a ROS Node
	rospy.init_node('init_environment')
	# rospack = RosPack()
	rospy.sleep(6.0)

	# Arguments
	w_cor = float(sys.argv[1])
	l_cor = float(sys.argv[2])
	n_max = int(sys.argv[3])

	# Dingen
	n_shelves = int(l_cor / 7) + 2
	x0 = 0
	xstart = x0 + 4
	x0_shelves = xstart - 7
	n0 = 0
	d_min = 1.4

	# Spawn init
	# TODO check if last shelve already exists
	spawn_shelves(w_cor,n_shelves,x0_shelves,0)
	# TODO: check if last obs already exists
	spawn_obstacles(n_max,xstart,0)
