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

# Topics
d_topics = ['density1_f','narrowness1']
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics]
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics] + ['performance2']
xtopics = ['density1_f','narrowness1','d_density1_f','performance2']
ytopic = 'safety2'

# Models
# models = ['SGD','ElasticNet','Lasso','Ridge','SVR_linear','SVR_rbf','SVR_poly','RF']
models = ['RF']
# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 100

# Experiment start and end
start_ms = 10000
end_ms = 1000

# Import Bag files into pandas
X, y, groups = generate_dataset_all(configs,xtopics,ytopic,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling)
# X2, y2, groups2 = generate_dataset_all(configs,xtopics2,ytopic,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling)

m = dict()
m['SGD'] = GridSearchCV(SGDRegressor(),
                     param_grid={"loss": ['squared_loss','huber','epsilon_insensitive'],
                                 "penalty":['l2','elasticnet'],
                                 "alpha":   10.0**-np.arange(1,7),
                                 "epsilon":[0.01,0.05,0.1,0.2]}, 
                     n_jobs=-1)
m['ElasticNet'] = GridSearchCV(ElasticNet(),
                     param_grid={"alpha": [0.1, 1.0],
                                 "l1_ratio": [0.0,0.25,0.5,0.75,1.0]}, 
                     n_jobs=-1)
m['Lasso'] = GridSearchCV(Lasso(),
                     param_grid={"alpha": [0.1, 1.0]}, 
                     n_jobs=-1)
m['Ridge'] = GridSearchCV(Ridge(),
                     param_grid={"alpha": np.logspace(-6, 6, 13)}, 
                     n_jobs=-1)                                                       
m['SVR_linear'] = GridSearchCV(SVR(kernel='linear'),
                     param_grid={"C": [1, 25, 50, 75, 100, 125, 150],
                                 "gamma": np.logspace(-2, 2, 5)}, 
                     n_jobs=-1)
                     # scoring=scoring_metric)
m['SVR_rbf'] = GridSearchCV(SVR(kernel='rbf'),
                     param_grid={'C': [1, 10, 100, 1000],
                                 'gamma': [0.001, 0.0001]}, 
                     n_jobs=-1)
m['SVR_poly'] = GridSearchCV(SVR(kernel='poly',gamma=0.1),
                     param_grid={"C": [50, 100, 150, 200, 250, 300],
                                 # "gamma": np.logspace(-2, 2, 5),
                                 "degree":[2,3,4]}, 
                     n_jobs=-1)
m['RF'] = GridSearchCV(RandomForestRegressor(),
                     param_grid={"n_estimators": [5, 10, 50, 100, 150, 200]}, 
                     n_jobs=-1)


# Train loop
os.chdir(dir_results)
sys.stdout=open("test_modeltypes.txt","w")
for config in configs:
    print('Training for %s'%config)
    # Loop train models
    for modelname in models:
        gkf = GroupKFold(n_splits=int(groups[config][-1])+1)

        idm = 0   
        best_score = -10  
        for train, test in gkf.split(X[config], y[config], groups=groups[config]):
            # Fit model
            m[modelname].fit(X[config][train],y[config][train])
            # Score 
            r2score = m[modelname].score(X[config][test],y[config][test])
            r2score_own = m[modelname].score(X[config][train],y[config][train])
            
            # print('$%s_model= %s, idx=%s, own score = %s, average score = %s'%(ytopic,modelname,idx,round(r2score[modelname][idx][idx],4),round(np.average(r2score[modelname][idx]),4)))
            # Save model
            if r2score > best_score:
                best_filename = dir_models + "%s_%s_%s_best.pkl"%(ytopic,modelname,config)
                model_best = m[modelname]
                best_score = r2score

            print('idm:%s, modelname:%s test score:%s, own score:%s'%(idm,modelname,round(r2score,5),round(r2score_own,5)))
            idm += 1

        with open(best_filename, 'wb') as file:
            pickle.dump(model_best, file)
            

sys.stdout.close()