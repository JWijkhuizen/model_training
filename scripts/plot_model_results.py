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
from matplotlib.widgets import Button


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
exp = 'Experiment6'
# configs = ['dwa1','dwa2','teb1','teb2']
configs = ['dwa_v0_a0_b0']
# configs = ['dwa1','dwa2']

d_topics = ['density1_f','narrowness1']

# xtopics = ['density1','narrowness1']
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics]
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics] + ['performance2']
xtopics = ['density1_f','narrowness1','d_density1_f','performance2']
ytopic = 'safety2'

model = 'safety2_Poly6_cdwa1_4_best.pkl'

colors = dict()
colors['safety'] = 'tab:green'
colors['performance'] = 'tab:blue'

# Resamplesize and smoothing (rolling)
samplesize = 100
rolling = 1

# Experiment start and end
# start_ms = 10000
# end_ms = 1000

# Import Bag files into pandas
X, y, groups = generate_dataset_all(configs,xtopics,ytopic,d_topics,exp,dir_bags,samplesize,rolling)

# Print all the files with idx
# for config in configs:
#     print('idx   File')
#     for idx in range(n_exp[config]):
#         print('%-5s %-s'%(idx, files[config][idx]))

def fnext(event):
    plt.close()

print('Load model')
pkl_filename = dir_models + model
with open(pkl_filename, 'rb') as file:
    m1 = pickle.load(file)

colors = ['tab:blue','tab:orange']
for config in configs:  
    pf = PolynomialFeatures(degree=6)
    Xp = pf.fit_transform(X[config])  

    gkf = GroupKFold(n_splits=int(groups[config][-1])+1)
    idm = 0
    for train, test in gkf.split(X[config], y[config], groups=groups[config]):
        print('Predict')
        y1 = m1.predict(Xp[test])
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
        ax.set_title("test set: %s \n model: %s"%(idm,model))
        # if idy == len(files_dwa2)-1:
        # print(y1)
        # print('rmse = %s'%(mean_squared_error(y, y1))
        plt.tight_layout()

        plt.subplots_adjust(bottom=0.2)
        axnext = plt.axes([0.7, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(fnext)

        plt.show()

        idm+=1

