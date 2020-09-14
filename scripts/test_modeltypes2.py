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
exp = 'exp2'
configs = ['cdwa_v0_a0_b0','cteb_v0_a0_b0']
# configs = ['cdwa_v0_a0_b0']

# Topics
d_topics = ['density1_f','narrowness1']
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics]
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics] + ['performance2']
xtopics = ['density1_f','narrowness1','d_density1_f','performance2']
ytopic = 'performance4'

# Models
models = ['SGD','ElasticNet','Lasso','Ridge','SVR_linear','SVR_rbf','SVR_poly','RF']
polies = [1,2,3,4,5,6]

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
lr = LinearRegression(normalize=True)
for config in configs:
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
        bestpolyscore = 0
        bestpoly=0
        for poly in polies:
            pf = PolynomialFeatures(degree=poly)
            Xp = pf.fit_transform(X[config])
            lr.fit(Xp[train],y[config][train])
            r2score = lr.score(Xp[test],y[config][test])
            r2score_own = lr.score(Xp[train],y[config][train])
            print('idm:%s, test score:%s, own score:%s, model poly:%s'%(idm,round(r2score,5),round(r2score_own,5),poly))

            if r2score > best_score:
                best_filename = dir_models + "%s_Poly%s_%s_best.pkl"%(ytopic,poly,config)
                model_best = lr
                best_score = r2score
            if r2score > bestpolyscore:
                bestpolyscore=r2score
                bestpoly = poly
            idm += 1
        print("Best poly = %s"%bestpoly)
    with open(best_filename, 'wb') as file:
        pickle.dump(model_best, file)


# sys.stdout.close()