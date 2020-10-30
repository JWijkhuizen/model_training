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
configs = ['dwa_v2_a0_b0','dwa_v1_a0_b0','dwa_v2_a1_b0']
config_data = 'dwa_v2_a0_b0'

d_topics = []

xtopics = ['obstacle_density21','narrowness1']
ytopics = ['safety']

# Resamplesize and smoothing (rolling)
samplesize = 100
rolling = 1

# Import Bag files into pandas
os.chdir(dir_bags)
files = dict()
for config in configs:
    files[config] = sorted(glob.glob("*%s_c%s_w4_C4_r3*.bag"%(exp,config)))
# print(files)

colors = ['tab:blue','tab:orange']
for ytopic in ytopics:
    # Import Bag files into pandas
    X, y, groups = generate_dataset_all_selectedfiles(files,configs,xtopics,ytopic,d_topics,exp,dir_bags,samplesize,rolling)
    pf = PolynomialFeatures(degree=5)
    Xp = pf.fit_transform(X[config_data])

    ym = dict()
    for config in configs:
        print('Load model')
        dir_model = rospack.get_path(config)
        pkl_filename = dir_model + "/quality_models/" + ytopic + ".pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)

        ym[config] = model.predict(Xp)

    yr = y[config_data]

    t = np.linspace(0,(len(yr)-1)/10,len(yr))
    fig, ax = plt.subplots(figsize=[12.8,4.8])
    for config in configs:
        # if config not in config_data:
        ax.plot(t, ym[config], label='Predicted %s'%config)#, color=colors[1])#, score = %s'%(round(m1.score(df[idy][xtopics].values,df[idy][ytopic].values),2)))
    ax.plot(t, yr, label='Measured %s'%config_data, linestyle='--', color=colors[0])
    ax.legend()
    # ax.set_title('Best safety model and real %s \n trained on run 1, tested on run %s , config = %s \n rmse = %s'%(ytopic,idy,config,round(mean_squared_error(y, y1),5)))
    ax.set_ylim(0,1.1)
    ax.set_xlim(4,32)
    ax.set_ylabel('Safety level')
    ax.set_xlabel("Time (s)")
    ax.set_title("Model (%s) predictions"%(ytopic))
    plt.tight_layout()

    os.chdir(dir_figs)
    fig.savefig(dir_figs + 'model_%s_test_all'%(ytopic) + '.png')

plt.show()