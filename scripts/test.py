#!/usr/bin/env python3.6

# import rosbag
import rospy
import rospkg
import os
import sys
import glob
import rosbag_pandas
import matplotlib.pyplot as plt
# import statistics as stat
from openpyxl import Workbook
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
import pickle

from functions_postprocess import *


# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
dir_bags = path + '/bags'
dir_figs = path + '/figures/'
dir_models = path + '/models/'
dir_results = path + '/results/'

# Experiment name and configs
exp = 'exp1'
configs = ['cteb_v0_a0_b0']

# Topics
d_topics = ['density1','narrowness1']
xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics]
ytopic = 'safety2'


# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 100

# Experiment start and end
start_ms = 10000
end_ms = 1000

# Import Bag files into pandas
X, y, groups = generate_dataset_all(configs,xtopics,ytopic,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling)
