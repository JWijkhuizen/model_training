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

from functions_postprocess import *

# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
dir_bags = path + '/bags'
dir_figs = path + '/figures/'

# Topics to get from bags
bag_topics = ['/metrics/safety1','/metrics/safety2','/metrics/density1','/metrics/narrowness1','/metrics/performance2','/metrics/time']
d_topics = ['density1','narrowness1']

# Experiment and configs to import
exp = 'Experiment1'
configs = ['dwa2']

# Data postprocess
samplesize = 10		# [ms] resample size
rolling = 1		# [n]  for smooting the data


# Import Bag files into pandas
os.chdir(dir_bags)
files = dict()
df = dict()
for config in configs:
    df[config] = dict()

    files[config] = sorted(glob.glob("%s*%s.bag"%(exp,config)))
    for idx in range(len(files[config])):
        df[config][idx] = import_bag(files[config][idx],samplesize,rolling)
        # df[config][idx] = add_derivs(df[config][idx],d_topics)
n_exp = len(files[configs[0]])

# Print all the files with idx
print('idx   File')
for idx in range(len(files[configs[0]])):
    print('%-5s %-s'%(idx, files[configs[0]][idx]))



for config in configs:
	for idx in [8]:
		fig = graph([df[config][idx]['safety2'],df[config][idx]['safety1']], 'Limited and unlimited safety metric', 'Time [s]', 'Safety level')
		# Save fig
		# fig.savefig(dir_figs + 'Limited and unlimited safety metric' + '.png')



plt.show()