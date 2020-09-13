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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import PolynomialFeatures
import pickle


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

# xtopics = ['density1','narrowness1']
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics]
xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics] + ['performance2']
ytopic = 'safety2'

model = 'safety2_Poly3_cdwa_v0_a0_b0_best.pkl'

# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 100

# Experiment start and end
start_ms = 10000
end_ms = 1000

# Import Bag files into pandas
X, y, groups = generate_dataset_all(configs,xtopics,ytopic,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling)

# Print all the files with idx
# for config in configs:
#     print('idx   File')
#     for idx in range(n_exp[config]):
#         print('%-5s %-s'%(idx, files[config][idx]))


print('Load model')
pkl_filename = dir_models + model
with open(pkl_filename, 'rb') as file:
    m1 = pickle.load(file)

colors = ['tab:blue','tab:orange']
for config in configs:  
    pf3 = PolynomialFeatures(degree=3)
    Xp3 = pf3.fit_transform(X[config])  

    gkf = GroupKFold(n_splits=int(groups[config][-1])+1)
    for train, test in gkf.split(X[config], y[config], groups=groups[config]):
        print('Predict')
        y1 = m1.predict(Xp3[test])
        for i in range(len(y1)):
            y1[i] = min(y1[i],1)
        for i in range(len(y[config])):
            y[config][i] = min(y[config][i],1)

        print("Plotting")
        fig, ax = plt.subplots()
        ax.plot(y1, label='Model', color=colors[1])#, score = %s'%(round(m1.score(df[idy][xtopics].values,df[idy][ytopic].values),2)))
        ax.plot(y[config][test], label='real', linestyle='--', color=colors[0])
        ax.legend(loc=0)
        # ax.set_title('Best safety model and real %s \n trained on run 1, tested on run %s , config = %s \n rmse = %s'%(ytopic,idy,config,round(mean_squared_error(y, y1),5)))
        ax.set_ylim(0,1.2)
        # if idy == len(files_dwa2)-1:
        # print(y1)
        # print('rmse = %s'%(mean_squared_error(y, y1))
        plt.tight_layout()

plt.show()