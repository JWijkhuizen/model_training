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


def move_obstacles(w,l,obs_dense,d_min,sx,xstart,nstart,nmax):
	# Random generator number
	seed(sx)
	max_tries = 100

	# Obstacle amount
	n = int(w*l/obs_dense)
	# print("Number of obstacles: " + str(n))

	# Position limits
	w_lim = w - 1.2
	x = [xstart, l-1+xstart]
	y = [-w_lim/2, w_lim/2]

	# Obstacle positions
	obs_x, obs_y = reset_obs_pos(nmax,xstart)
	# print(obs_x)
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
		if tries < 100:
			obs_x[idx] = xi
			obs_y[idx] = yi
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


def move_shelves(w,n,xstart,nstart):
	# Set up proxy
	set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
	model_pose = ModelState()
	
	# Spawn shelves
	x = [0, 6, 7, 13, 14, 20, 21, 27, 28, 34, 35, 41]
	ids = nstart
	for i in range(n*2):
		# sign of y coordinate and yaw.
		if (i % 2) == 0:
			s = 1
			yaw = 0
		else:
			s = -1
			yaw = math.pi
		model_pose.model_name = "shelves%s"%ids
		model_pose.pose.position.x = x[i]+xstart
		model_pose.pose.position.y = s*(w/2 + 0.5)
		model_pose.pose.position.z = 0
		model_pose.pose.orientation = Quaternion(*quaternion_from_euler(0,0,yaw))

		rospy.wait_for_service('/gazebo/set_model_state')
		resp = set_state( model_pose )
		ids += 1


if __name__ == '__main__':
	if len(sys.argv) != 6:
		print('Not enough input arguments')
		# break

	# Initialize a ROS Node
	rospy.init_node('Move_environment')
	# rospack = RosPack()
	rospy.sleep(6.0)

	# Arguments
	w_cor = float(sys.argv[1])
	l_cor = float(sys.argv[2])
	C = float(sys.argv[3])
	sx = int(sys.argv[4])
	n_max = int(sys.argv[5])

	# Dingen
	obs_n = int(w_cor*l_cor*C)
	n_shelves = int(l_cor / 7) + 2
	x0 = 0
	xstart = x0 + 4
	x0_shelves = xstart - 7
	n0 = 0
	d_min = 1.4

	# Move shelves if needed
	move_shelves(w_cor,n_shelves,x0_shelves,n0)
	# Move to random location
	move_obstacles(w_cor,l_cor,C,d_min,sx,xstart,n0,n_max)

	sys.exit(1)