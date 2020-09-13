#!/usr/bin/env python3.6

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

from functions_postprocess import *


# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
dir_bags = path + '/bags'
dir_figs = path + '/figures/'
dir_models = path + '/models/'
dir_results = path + '/results/'

# Experiment name and configs
exp = 'exp1'
# configs = ['cdwa_v0_a0_b0','cteb_v0_a0_b0']
configs = ['cdwa_v0_a0_b0']

# Topics
d_topics = ['density4','narrowness1']
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics]
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics] + ['performance2']
xtopics = ['density4','narrowness1','d_density4','performance2']
ytopic = 'safety2'

# Models
models = ['SGD','ElasticNet','Lasso','Ridge','SVR_linear','SVR_rbf','SVR_poly','RF']

# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 100

# Experiment start and end
start_ms = 10000
end_ms = 1000

# Import Bag files into pandas
X, y, groups = generate_dataset_all(configs,xtopics,ytopic,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling)
# X2, y2, groups2 = generate_dataset_all(configs,xtopics2,ytopic,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling)

# Train loop
os.chdir(dir_results)
# sys.stdout=open("test_modeltypes.txt","w")
for config in configs:
    pf2 = PolynomialFeatures(degree=2)
    Xp2 = pf2.fit_transform(X[config])
    pf3 = PolynomialFeatures(degree=3)
    Xp3 = pf3.fit_transform(X[config])
    print('Training for %s'%config)
    # Loop train models
    # for modelname in models:
    gkf = GroupKFold(n_splits=int(groups[config][-1])+1)
    print(int(groups[config][-1])+1)

    idm = 0   
    best_score = -10  
    for train, test in gkf.split(X[config], y[config], groups=groups[config]):
        # Fit model
        # m[modelname].fit(X[config][train],y[config][train])
        # lr = LinearRegression(normalize=True)
        # lr.fit(X[config][train],y[config][train])
        # r2score = lr.score(X[config][test],y[config][test])
        # r2score_own = lr.score(X[config][train],y[config][train])
        # print('idm:%s, test score:%s, own score:%s, model: Linear'%(idm,round(r2score,5),round(r2score_own,5)))

        # lr = LinearRegression(normalize=True)
        # lr.fit(Xp2[train],y[config][train])
        # r2score = lr.score(Xp2[test],y[config][test])
        # r2score_own = lr.score(Xp2[train],y[config][train])
        # print('idm:%s, test score:%s, own score:%s, model: Polynomial2'%(idm,round(r2score,5),round(r2score_own,5)))

        lr = LinearRegression(normalize=True)
        lr.fit(Xp3[train],y[config][train])
        r2score = lr.score(Xp3[test],y[config][test])
        r2score_own = lr.score(Xp3[train],y[config][train])
        print('idm:%s, test score:%s, own score:%s, model: Polynomial3'%(idm,round(r2score,5),round(r2score_own,5)))

        if r2score > best_score:
            best_filename = dir_models + "%s_Poly3_%s_best.pkl"%(ytopic,config)
            model_best = lr
            best_score = r2score

        idm += 1
        #     print('idm:%s, modelname:%s test score:%s, own score:%s'%(idm,modelname,round(r2score,5),round(r2score_own,5)))
        #     idm += 1

    with open(best_filename, 'wb') as file:
        pickle.dump(model_best, file)


# sys.stdout.close()