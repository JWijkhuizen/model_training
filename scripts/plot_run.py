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

from functions_postprocess import *


# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
# path_st = rospack.get_path('simulation_tests')
# dir_bags = path_st + '/bags'
dir_bags = path + '/bags'
dir_figs = path + '/figures/'
dir_models = path + '/models/'
dir_results = path + '/results/'


# Experiment name and configs
exp = 'exp2'
# configs = ['dwa1','dwa2','teb1','teb2']
configs = ['cdwa_v0_a0_b0']
# configs = ['dwa1','dwa2']

d_topics = ['density1','narrowness1']

xtopics = ['density1','density1_f']
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics]
ytopics = ['safety1','safety2','performance4']

runs_id = [0]

# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 200

# Import Bag files into pandas
os.chdir(dir_bags)
files = dict()
df = dict()
lags = dict()
lags_m = dict()
mean_lags = dict()

for config in configs:
    df[config] = dict()

    files[config] = sorted(glob.glob("%s_%s*.bag"%(exp,config)))
    # print(files[config])
    lags[config] = []
    lags_m[config] = dict()
    # for idx in range(len(files[config])):
    for idx in runs_id:
        df[config][idx] = import_bag(files[config][idx],samplesize,rolling)
        df[config][idx] = add_derivs(df[config][idx],d_topics)
        df[config][idx] = df[config][idx].iloc[(int(4000/samplesize)):]

        df[config][idx].drop(df[config][idx].head(int(10000/samplesize)).index,inplace=True)
        df[config][idx].drop(df[config][idx].tail(int(1000/samplesize)).index,inplace=True) # drop last n rows

        dfs1 = []
        for ytopic in ytopics:
            dfs1.append(df[config][idx][ytopic])
        dfs2 = []
        for xtopic in xtopics:
            dfs2.append(df[config][idx][xtopic])
        graph21([dfs1,dfs2], ["Signals of the run %s, config %s"%(idx, config)], "time [s]", ["QA","EM"])

plt.show()