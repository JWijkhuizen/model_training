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
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.smoothers_lowess import lowess

from functions_postprocess import *


# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
path_st = rospack.get_path('simulation_tests')
dir_bags = path + '/bags'
dir_figs = path + '/figures/'
dir_models = path + '/models/'
dir_results = path + '/results/'

# Experiment name and configs
exp = 'test'
# configs = ['dwa1','dwa2','teb1','teb2']
configs = ['cdwa_v0_a0_b0']

d_topics = ['density1','narrowness1']

xtopics = ['density1','d_density1','narrowness1','d_narrowness1','performance1']
# xtopics = ['density1','d_density1']
# ytopic = 'safety2'
ytopic = 'performance2_2'

model = 'performance4_RF_cdwa_v0_a0_b0_best.pkl'
lags = [23, -26, -20,  -2,  73]

# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 100

# Import Bag files into pandas
os.chdir(dir_bags)
files = dict()
df = dict()
df_shift = dict()

for config in configs:
    df[config] = dict()
    df_shift[config] = dict()

    files[config] = sorted(glob.glob("%s*%s.bag"%(exp,config)))

    n_exp = len(files[configs[0]])
    # n_exp = 1

    for idx in range(n_exp):
        df[config][idx] = import_bag(files[config][idx],samplesize,rolling)
        df[config][idx] = add_derivs(df[config][idx],d_topics)

        df[config][idx] = df[config][idx].iloc[(int(4000/samplesize)):]

    for idx in range(n_exp):
        df_shift[config][idx] = shift_lags(df[config][idx],xtopics,lags)


# Print all the files with idx
print('idx   File')
for idx in range(n_exp):
    print('%-5s %-s'%(idx, files[configs[0]][idx]))



print('Load models')
pkl_filename = dir_models + model
with open(pkl_filename, 'rb') as file:
    m1 = pickle.load(file)


colors = ['tab:blue','tab:orange']
# for idy in range(len(files)):
for config in configs:
    for idy in range(n_exp):
    # for idy in [1]:
        print('Predict')
        X = df_shift[config][idy][xtopics].values
        y = df_shift[config][idy][ytopic].values

        y1 = m1.predict(X)
        for i in range(len(y1)):
            y1[i] = min(y1[i],1)
        # y1 = lowess(y1, df_shift[config][idy].index.total_seconds(), is_sorted=True, frac=0.025)
        # y2 = m2.predict(X)


        print("Plotting")
        fig, ax = plt.subplots()
        # Predictions
        ax.plot(df_shift[config][idy].index.total_seconds(),y1, label='Model', color=colors[1])#, score = %s'%(round(m1.score(df[idy][xtopics].values,df[idy][ytopic].values),2)))
        # ax.plot(df[idy].index.total_seconds(),y2, label='TEB', color='tab:orange')#, score = %s'%(round(m1.score(df[idy][xtopics].values,df[idy][ytopic].values),2)))
        # Real
        ax.plot(df_shift[config][idy].index.total_seconds(),y, label='real', linestyle='--', color=colors[0])
        ax.legend(loc=0)
        # ax.set_title('Best safety model and real %s \n trained on run 1, tested on run %s , config = %s \n rmse = %s'%(ytopic,idy,config,round(mean_squared_error(y, y1),5)))
        ax.set_ylim(0,1.2)
        # if idy == len(files_dwa2)-1:
        # print(y1)
        # print('rmse = %s'%(mean_squared_error(y, y1))
        plt.tight_layout()
        # Save fig
        # fig.savefig(dir_figs + 'Modelresult_teb1_train1_test%s'%idy + '.png')

plt.show()