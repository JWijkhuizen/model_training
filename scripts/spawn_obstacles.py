#!/usr/bin/env python

import sys
import os

import rospy
import rospkg

import math
from random import seed, random

from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose, Quaternion
from gazebo_msgs.srv import DeleteModel, ModelState, SetModelState
import move_base_msgs.msg as move


# Command line args:
# 	w_cor		corridor width
# 	l_cor		corridor length 
# 	C			clutterness
# 	seed 		seed number


def reset_obs_pos(n,xstart):
	obs_x = []
	obs_y = []
	for i in range(n):
		obs_x.append(i+xstart)
		obs_y.append(-10)
	return (obs_x,obs_y)

def spawn_obs(n,xstart,nstart):
	# Model path
	workspace = os.path.dirname(os.path.realpath(__file__)).replace('model_training/scripts','')
	path_models = workspace+'/ahxl_gazebo/gazebo_models/'


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

def move_obstacles(w,l,obs_dense,d_min,sx,xstart,nstart):
	# Random generator number
	seed(sx)
	max_tries = 100

	# Obstacle amount
	n = int(w*l*obs_dense)
	# print("Number of obstacles: " + str(n))

	# Position limits
	w_lim = w - 1.2
	x = [xstart, l-1+xstart]
	y = [-w_lim/2, w_lim/2]

	# Obstacle positions
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
		# if idx > 0:
			# print('tries = %s'%tries)
			# print('min(d) = %s'%(min(d)))
		if tries < 100:
			obs_x.append(xi)
			obs_y.append(yi)
	obs_z = 0.5

	# Set up proxy
	set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
	model_pose = ModelState()

	# Move Obstacles
	for i in range(len(obs_x)):
		model_pose.model_name = 'obstacle%s'%(i+nstart)
		model_pose.pose.position.x = obs_x[i]
		model_pose.pose.position.y = obs_y[i]
		model_pose.pose.position.z = obs_z
		rospy.wait_for_service('/gazebo/set_model_state')
		resp = set_state( model_pose )

def check_space(w,y):
	return 0.5*w-(-abs(y)+0.2)

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
	print(sys.argv)
	if len(sys.argv) != 5:
		print('Not enough input arguments')
		# break

	# Initialize a ROS Node
	rospy.init_node('place_obstacles')
	rospack = RosPack()
	rospy.sleep(6.0)

	# Arguments
	w_cor = sys.argv[1]
	l_cor = sys.argv[2]
	C = sys.argv[3]
	seed = sys.argv[4]

	# Dingen
	obs_n = int(w_cor*l_cor*C)
	n_shelves = int(l_cor / 7)
	x0 = 0
	n0 = 0

	# Spawn init
	# TODO check if last shelve already exists
	spawn_shelves(w_cor,n_shelves,x0,0)
	# TODO: check if last obs already exists
	spawn_obstacles(obs_n,x0,0)
