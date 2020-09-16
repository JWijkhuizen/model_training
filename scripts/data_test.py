#!/usr/bin/env python3.6

# import rosbag
import rospy
import rosbag_pandas
import rospkg
import os
import glob
import matplotlib.pyplot as plt
import statistics as stat
from openpyxl import Workbook
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pickle

from functions_postprocess import *

# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
dir_bags = path + '/bags'

# Topics to get from bags
bag_topics = ['/metrics/safety1','/metrics/safety2','/metrics/density1','/metrics/narrowness1','/metrics/performance2']
d_topics = ['density1','narrowness1']

# Experiment and configs to import
exp = 'Experiment6'
configs = ['dwa1']

# Data postprocess
samplesize = 10		# [ms] resample size
rolling = 20		# [n]  for smooting the data

n = 10

# Bag files
os.chdir(dir_bags)
files = []
for config in configs:
	files = files + sorted(glob.glob("%s*%s.bag"%(exp,config)))

# Print all the files with idx
print('idx   File')
[print('%-5s %-s'%(idx, files[idx])) for idx in range(len(files))]

# Import bags and store in pandas
print("Import bags")
df = dict()
df2 = dict()
for idx in range(n):
	df[idx] = import_bag(files[idx], samplesize, rolling)
	df[idx] = add_derivs(df[idx],d_topics)

# topics to plot
topics = ['safety1', 'safety2']
labels = dict()
labels['safety1'] = 'safety [0,1]'
labels['safety2'] = 'safety without limit at 1'

# colors = ['tab:blue','tab:orange']
# idc = 0
# for idy in range(len(files)):
print("Plotting")
for config in configs:
	for idx in range(n):
		dfs1 = df[config][idx]
		graph21(dfs, titles, xlabel, ylabels)

plt.show()