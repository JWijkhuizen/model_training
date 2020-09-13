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
exp = 'exp1'
# configs = ['dwa1','dwa2','teb1','teb2']
configs = ['cdwa_v0_a0_b0']
# configs = ['dwa1','dwa2']

d_topics = ['density1','narrowness1']

# xtopics = ['density4','d_density1','narrowness1','d_narrowness1']
xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics]
ytopic = 'safety2'

# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 100

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
    ide = 0
    for idx in range(10):
        df[config][ide] = import_bag(files[config][idx],samplesize,rolling)
        # if df[config][ide].index[-1].total_seconds() > 50:
        #     print("experiment %s failed, total time=%s"%(idx,df[config][ide].index[-1].total_seconds()))
        #     continue
        df[config][ide] = add_derivs(df[config][ide],d_topics)

        df[config][idx].drop(df[config][idx].head(int(10000/samplesize)).index,inplace=True)
        df[config][idx].drop(df[config][idx].tail(int(1000/samplesize)).index,inplace=True) # drop last n rows


        lags_temp = determine_lags(df[config][ide],xtopics,ytopic,samplesize)
        lags_temp = [element * samplesize for element in lags_temp]
        # print(files[config][idx])
        print(lags_temp)
        lags[config].append(lags_temp)
        lags_m[config][idx] = lags_temp
        # print(lags[config][-1])
        # df_shift = shift_lags(df[config][idx],xtopics,lags[config])
        # df_shift1 = shift_lags(df[config][idx],xtopics,-1*lags[config])
        ide += 1
    mean_lags[config] = np.array(lags[config]).mean(axis=0).astype(int)
    print(mean_lags[config])


os.chdir(dir_results)
wb = Workbook()
ws = wb.active
for config in configs:
    ws.append([config])
    ws.append(["","Lags for Environment metric [ms]"])
    ws.append(["run","OD1","N1","d_OD1","d_N1"])

    for idx in range(len(files[config])):
        ws.append([idx] + lags_m[config][idx] + [files[config][idx]])
    ws.append(["Mean"] + mean_lags[config].tolist())

# wb.save(filename = 'Lags_table.xlsx')