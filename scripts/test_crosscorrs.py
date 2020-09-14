#!/usr/bin/env python3.6

# import rosbag
import rospy
import rospkg
import os
import glob
import rosbag_pandas
import matplotlib.pyplot as plt
# import statistics as stat
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
exp = 'exp2'
configs = ['cdwa_v0_a0_b0']

d_topics = ['density1','narrowness1']
xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics]
xtopics = xtopics + ['density1_f']
ytopic = 'performance3'

runs_id = [0,1,2]

plottopics = ['density1','density2']

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
lags = dict()
mean_lags = dict()
for config in configs:
    df[config] = dict()

    files[config] = sorted(glob.glob("%s_%s*.bag"%(exp,config)))
    lags[config] = []
    # for idx in range(len(files[config])):
    for idx in runs_id:
        for plottopic in plottopics:
        
            df[config][idx] = import_bag(files[config][idx],samplesize,rolling)
            df[config][idx] = add_derivs(df[config][idx],d_topics)

            df[config][idx].drop(df[config][idx].head(int(10000/samplesize)).index,inplace=True)
            df[config][idx].drop(df[config][idx].tail(int(1000/samplesize)).index,inplace=True) # drop last n rows

            graph_xcorr(df[config][idx][ytopic],df[config][idx][plottopic],samplesize,"idx:%s, topic:%s"%(idx,plottopic))

            # lags_temp = determine_lags(df[config][idx],xtopics,ytopic,samplesize)
            # lags[config].append(lags_temp)
            # print(lags_temp)
            # print(lags[config][-1])
            # print(lags[config][-1])
        


plt.show()