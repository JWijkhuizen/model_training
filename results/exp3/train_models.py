#!/usr/bin/env python

# import rosbag
import rospy
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
from sklearn.model_selection import GridSearchCV
import pickle
from functions import *


# Current directory
workspace = os.path.dirname(os.path.realpath(__file__))

# Topics to load from bagfile
bag_topics = ['/metrics/safety1','/metrics/safety2','/metrics/density1','/metrics/narrowness1']
# Topics to take the derivative from
d_topics = ['density1','narrowness1']
# configs
configs = ['dwa2','teb1']
# configs = ['dwa2']
# Import parameters
stepsize = 10
rolling = 100

# Dependent variables
xtopics = ['density1','d_density1','narrowness1','d_narrowness1']
# Independent variable
ytopic = 'safety2'

# models to train
m = dict()
models = ['SVR_linear','SVR_rbf','SVR_poly','RF','SGD']
# scoring_metric = 'neg_mean_squared_error'
scoring_metric = 'r2'

m[models[0]] = GridSearchCV(SVR(kernel='linear',gamma=0.1),
                     param_grid={"C": [1, 25, 50, 75, 100, 125, 150],
                                 "gamma": np.logspace(-2, 2, 5)}, 
                     n_jobs=-1,
                     scoring=scoring_metric)
m[models[1]] = GridSearchCV(SVR(kernel='rbf',gamma=0.1),
                     param_grid={"C": [1, 25, 50, 75, 100, 125, 150],
                                 "gamma": np.logspace(-2, 2, 5)}, 
                     n_jobs=-1,
                     scoring=scoring_metric)
m[models[2]] = GridSearchCV(SVR(kernel='poly',gamma=0.1),
                     param_grid={"C": [50, 100, 150, 200, 250, 300],
                                 # "gamma": np.logspace(-2, 2, 5),
                                 "degree":[2,3,4]}, 
                     n_jobs=-1,
                     scoring=scoring_metric)
# Random forrest
m[models[3]] = GridSearchCV(RandomForestRegressor(),
                     param_grid={"n_estimators": [5, 10, 50, 100, 150, 200]}, 
                     n_jobs=-1,
                     scoring=scoring_metric)
# Stochastic Gradient Descent
m[models[4]] = GridSearchCV(SGDRegressor(),
                     param_grid={"loss": ['squared_loss','huber','epsilon_insensitive'],
                                 "penalty":['l2','elasticnet'],
                                 "epsilon":[0.01,0.05,0.1,0.2]}, 
                     n_jobs=-1,
                     scoring=scoring_metric)
# GaussianProcessRegressor
# m[models[4]] = GridSearchCV(GaussianProcessRegressor(),
#                      param_grid={"loss": ['squared_loss','huber','epsilon_insensitive'],
#                                  "penalty":['l2','elasticnet'],
#                                  "epsilon":[0.01,0.05,0.1,0.2]}, 
#                      n_jobs=-1,
#                      scoring=scoring_metric)

# Train loop
wb = Workbook()
for config in configs:
    print('Training for %s'%config)
    files = sorted(glob.glob("*%s.bag"%config))
    # Import all files
    X = dict()
    y = dict()
    for idx in range(len(files)):
        df = import_bag(files[idx], bag_topics,stepsize,rolling)
        df = add_derivs(df,d_topics)
        X[idx] = df[xtopics].values
        y[idx] = df[ytopic].values

    # Create sheet for config
    ws = wb.create_sheet(title=config)
    
    # Loop train models
    r2score = dict()
    for modelname in models:
        r2score[modelname] = dict()
        for idx in range(len(files)):
            if idx == 5 and config == 'teb1' and modelname == 'SVR_poly':
                # I dont know why, but the script gets stuck with this combination. so skip it
                break
            # Fit models
            m[modelname].fit(X[idx],y[idx])
            # Score model on all datasets
            r2score[modelname][idx] = []
            for idy in range(len(files)):
                r2score[modelname][idx].append(m[modelname].score(X[idy],y[idy]))
            print('model= %s, idx=%s, average score = %s'%(modelname,idx,np.average(r2score[modelname][idx])))
            # Save model
            pkl_filename = "models/model%s_%s_%s.pkl"%(modelname,config,idx)
            with open(pkl_filename, 'wb') as file:
                pickle.dump(m[modelname], file)

            # Parameters to results sheets
            # column names, only on top
            if idx == 0:
                ws.append(['Model name','Trainset number'] + [x for x in m[modelname].best_params_] + ['average score','own score','All scores'])
            # Parameter values
            ws.append(["models/model%s_%s_%s"%(modelname,config,idx), idx] + [str(m[modelname].best_params_[x]) for x in m[modelname].best_params_] + [round(np.average(r2score[modelname][idx]),3),round(r2score[modelname][idx][idx],3)] + [round(num, 3) for num in r2score[modelname][idx]])
            wb.save(filename = 'Results.xlsx')

wb.save(filename = 'Results.xlsx')




