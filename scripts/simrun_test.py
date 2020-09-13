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
# from tf.transformations import quaternion_from_euler, euler_from_quaternion





if __name__ == '__main__':
	code = int(sys.argv[1])
	sys.exit(code)
