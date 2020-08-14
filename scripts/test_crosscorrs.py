#!/usr/bin/env python

# import rosbag
import rospy
import rospkg
import os
import glob
import rosbag_pandas
import matplotlib.pyplot as plt
import statistics as stat
from openpyxl import Workbook
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
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
exp = 'Experiment1'
# configs = ['dwa1','dwa2','teb1','teb2']
configs = ['teb1']

d_topics = ['density1','narrowness1']

xtopics = ['density1','d_density1','narrowness1','d_narrowness1']
# xtopics = ['density1','d_density1']
# ytopic = 'safety2'
ytopic = 'performance2_3'

# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 100

# Import Bag files into pandas
os.chdir(dir_bags)
files = dict()
df = dict()
X = dict()
y = dict()
groups = dict()
X_shift = dict()
y_shift = dict()
groups_shift = dict()
X_shift2 = dict()
y_shift2 = dict()
groups_shift2 = dict()
lags = dict()
mean_lags = dict()
for config in configs:
    df[config] = dict()

    files[config] = sorted(glob.glob("%s*%s.bag"%(exp,config)))
    lags[config] = []
    for idx in range(len(files[config])):
    # for idx in range(5):
        df[config][idx] = import_bag(files[config][idx],samplesize,rolling)
        df[config][idx] = add_derivs(df[config][idx],d_topics)

        df[config][idx] = df[config][idx].iloc[(int(4000/samplesize)):]
        
        graph_xcorr(df[config][idx][ytopic],df[config][idx][xtopics[1]],samplesize)

        lags[config].append(determine_lags(df[config][idx],xtopics,ytopic,samplesize))
        print(lags[config][-1])
        # print(lags[config][-1])
        


n_exp = len(files[configs[0]])

# Print all the files with idx
print('idx   File')
for idx in range(len(files[configs[0]])):
    print('%-5s %-s'%(idx, files[configs[0]][idx]))




plt.show()