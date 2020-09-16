#!/usr/bin/env python

# import rosbag
import rospy
import rospkg
import os
import sys
import glob
import rosbag_pandas
import matplotlib.pyplot as plt
# import statistics as stat
from openpyxl import Workbook
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
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
dir_bags = path + '/bags/'
dir_figs = path + '/figures/'
dir_models = path + '/models/'
dir_results = path + '/results/'

# Experiment name and configs
exp = 'exp4'
configs = ['cdwa_v0_a0_b0','cdwa_v1_a0_b0','cteb_v0_a0_b0','cteb_v1_a0_b0']
# configs = ['cdwa1']

# Topics
d_topics = ['density1_f','narrowness1']
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics]
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics] + ['performance2']
xtopics = ['density1_f','narrowness1','d_density1_f','performance2']
# ytopic = 'performance3'
ytopic = 'safety2'

# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 100

# Experiment start and end
start_ms = 10000
end_ms = 1000

# Import Bag files into pandas
files, df = generate_dataset(configs,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling)
# os.chdir(dir_bags)
# files = dict()
# for config in configs:
#     files[config] = sorted(glob.glob("%s_%s*.bag"%(exp,config)))

print('idx   File')
for config in configs:
    for idx in range(len(files[config])):
        print('%-5s %-s'%(idx, files[config][idx]))

# Functions for buttons
def fyes(event):
    print("Yes, idx %s is usable"%idx)
    files_incl[config].append(files[config][idx])
    plt.close()

def fno(event):
    # files_incl[config].append(files[config][idx])
    print("No,  idx %s is not usable"%idx)
    plt.close()


files_incl = dict()
# topics = [['safety1','performance4'],[]]
print("Plotting")
for config in configs:
    files_incl[config] = []
    for idx in range(len(files[config])):   
        dfs1 = [df[config][idx]['safety2'],df[config][idx]['performance3']]
        dfs2 = [df[config][idx]['density1_f'],df[config][idx]['narrowness1'],df[config][idx]['d_density1_f'],df[config][idx]['performance2']]
        titles = ['Quality attributes \n %s %s'%(idx, files[config][idx]),'Environment metrics and robot states']
        xlabel = 'Time [s]'
        ylabels = ['QA value','metrics']
        fig, ax = graph21([dfs1,dfs2], titles, xlabel, ylabels)
        plt.subplots_adjust(bottom=0.2)
        axyes = plt.axes([0.3, 0.05, 0.1, 0.075])
        axno = plt.axes([0.7, 0.05, 0.1, 0.075])
        byes = Button(axyes, 'Use')
        bno = Button(axno, 'Not')
        byes.on_clicked(fyes)
        bno.on_clicked(fno)
        plt.show()
    print(files_incl[config])

os.chdir(dir_bags)
var_filename = dir_bags + "files_incl_%s"%(exp)
with open(var_filename, 'wb') as file:
    pickle.dump(files_incl, file)