#!/usr/bin/env python

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

from postprocess_functions import *

# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
dir_bags = path + '/bags'

# Topics to get from bags
bag_topics = ['/metrics/safety1','/metrics/safety2','/metrics/density1','/metrics/narrowness1','/metrics/performance2']
d_topics = ['density1','narrowness1']

# Experiment and configs to import
exp = 'Experiment4'
configs = ['dwa2','teb1']

# Data postprocess
samplesize = 10		# [ms] resample size
rolling = 20		# [n]  for smooting the data

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
for idx in range(len(files)):
	df[idx] = import_bag(files[idx], bag_topics, samplesize, rolling)
	df[idx] = add_derivs(df[idx],d_topics)

# topics to plot
topics = ['performance2', 'safety1']

# colors = ['tab:blue','tab:orange']
# idc = 0
# for idy in range(len(files)):
print("Plotting")
for idx in [3,13]:
	
	fig, ax = plt.subplots()
	# Predictions
	for topic in topics:
		ax.plot(df[idx].index.total_seconds(),df[idx][topic].values, label=df[idx][topic].name)
	ax.legend(loc=0)
	ax.set_title(files[idx])



plt.show()