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
dir_bags = path + '/bags/'
dir_figs = path + '/figures/'
dir_models = path + '/models/'
dir_results = path + '/results/'

# Experiment name and configs
exp = 'validation'
configs = ['teb_v1_a0_b0']
# configs = ['teb_v0_a0_b0']

d_topics = []

xtopics = ['obstacle_density21','narrowness1']
ytopics = ['safety']

# Resamplesize and smoothing (rolling)
samplesize = 100
rolling = 1

# Experiment start and end
start_ms = 1
end_ms = 1

# Import Bag files into pandas
os.chdir(dir_bags)
files = dict()
for config in configs:
    files[config] = sorted(glob.glob("*%s_c%s*.bag"%(exp,config)))
print(files)

colors = ['tab:blue','tab:orange']
for ytopic in ytopics:
    # Import Bag files into pandas
    X, y, groups = generate_dataset_all(configs,xtopics,ytopic,d_topics,exp,dir_bags,samplesize,rolling)

    for config in configs:
        print('Load model')
        dir_model = rospack.get_path(config)
        pkl_filename = dir_model + "/quality_models/" + ytopic + ".pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)


        pf = PolynomialFeatures(degree=5)
        Xp = pf.fit_transform(X[config])

        y1 = model.predict(Xp)
        for i in range(len(y1)):
            y1[i] = min(y1[i],1)
        yr = y[config]
        for i in range(len(yr)):
            yr[i] = min(yr[i],1)

        # Scoring metrics
        print("mean squared error       = %s"%mean_squared_error(yr,y1))
        print("explained variance score = %s"%explained_variance_score(yr,y1))
        print("r2 score                 = %s"%r2_score(yr,y1))

        print(y1)
        t = np.linspace(0,(len(yr)-1)/100,len(yr))
        # print(t[2]-t[1])
        # print(t)
        fig, ax = plt.subplots(figsize=[12.8,4.8])
        ax.plot(t, y1, label='Model', color=colors[1])#, score = %s'%(round(m1.score(df[idy][xtopics].values,df[idy][ytopic].values),2)))
        ax.plot(t, yr, label='real', linestyle='--', color=colors[0])
        ax.legend(loc=0)
        # ax.set_title('Best safety model and real %s \n trained on run 1, tested on run %s , config = %s \n rmse = %s'%(ytopic,idy,config,round(mean_squared_error(y, y1),5)))
        ax.set_ylim(0,1.2)
        ax.set_ylabel('QA')
        ax.set_xlabel("Time [s]")
        ax.set_title("Model (%s) validation \n for config: %s"%(ytopic,config))
        plt.tight_layout()

        os.chdir(dir_figs)
        fig.savefig(dir_figs + 'model_%s_validation_%s'%(ytopic,config) + '.png')

plt.show()