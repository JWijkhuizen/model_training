#!/usr/bin/env python3.6

import rospkg
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

import pickle

from functions_postprocess import *

# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
path2 = rospack.get_path('simulation_tests')
dir_bags = path2 + '/bags/'
dir_figs = path + '/figures/'
dir_models = path + '/models/'
dir_results = path + '/results/'

# Experiment name and configs
exp = 't1'
# configs = ['teb_v0_a0_b0','dwa_v0_a0_b0','teb_v1_a0_b0','dwa_v1_a0_b0']
configs = ["dwa_v0_a0_b0", "dwa_v0_a1_b0", "dwa_v0_a1_b1", "dwa_v0_a0_b1", "dwa_v1_a0_b0", "dwa_v1_a1_b0", "dwa_v1_a1_b1", "dwa_v1_a0_b1", "teb_v0_a0_b0", "teb_v0_a1_b0", "teb_v0_a1_b1", "teb_v0_a0_b1", "teb_v1_a0_b0", "teb_v1_a1_b0", "teb_v1_a1_b1", "teb_v1_a0_b1", "dwa_v2_a0_b0", "dwa_v2_a1_b0", "dwa_v2_a1_b1", "dwa_v2_a0_b1", "teb_v2_a0_b0", "teb_v2_a1_b0", "teb_v2_a1_b1", "teb_v2_a0_b1"]
configs = ["dwa_v1_a0_b0", "dwa_v1_a1_b0", "dwa_v1_a1_b1", "dwa_v1_a0_b1", "teb_v1_a0_b0", "teb_v1_a1_b0", "teb_v1_a1_b1", "teb_v1_a0_b1", "dwa_v2_a0_b0", "dwa_v2_a1_b0", "dwa_v2_a1_b1", "dwa_v2_a0_b1", "teb_v2_a0_b0", "teb_v2_a1_b0", "teb_v2_a1_b1", "teb_v2_a0_b1"]

# Topics
xtopics = ['obstacle_density21','narrowness1']
ytopics = ['safety']

# Resamplesize and smoothing (rolling)
samplesize = 100
rolling = 1


# print('Load files variable with files to include')
# pkl_filename = dir_bags + "files_incl_%s"%(exp)
# with open(pkl_filename, 'rb') as file:
#     files = pickle.load(file)

colors = ['tab:blue','tab:orange']
os.chdir(dir_bags)
for config in configs:
    perf = []
    files = sorted(glob.glob("*%s_c%s*.bag"%(exp,config)))
    # print(files)
    for file in files:
        df,start,end = import_bag(file,samplesize,rolling)

        # Start and end time:
        df.drop(df.head(int((start*1000)/samplesize)).index,inplace=True)
        df.drop(df.tail(int((end*1000)/samplesize)).index,inplace=True) # drop last n rows

        performance = df['performance_old32'].mean()
        perf.append(performance)
    perf_average = sum(perf) / len(perf) 
    print(str([config,perf_average]))
        