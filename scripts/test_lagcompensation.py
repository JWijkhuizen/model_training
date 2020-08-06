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

from functions_postprocess import *


# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
dir_bags = path + '/bags'
dir_figs = path + '/figures/'
dir_models = path + '/models/'
dir_results = path + '/results/'

# Experiment name and configs
exp = 'Experiment1'
# configs = ['dwa1','dwa2','teb1','teb2']
configs = ['teb1']

d_topics = ['density1','narrowness1']

xtopics = ['density1','d_density1','narrowness1','d_narrowness1']
# xtopics = ['density1','d_density1']
ytopic = 'safety2'
# ytopic = 'performance2_3'

# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 100

# Import Bag files into pandas
os.chdir(dir_bags)
files = dict()
df = dict()
X = dict()
y = dict()
groups = dict()
X_shift = dict()
y_shift = dict()
groups_shift = dict()
for config in configs:
    df[config] = dict()

    files[config] = sorted(glob.glob("%s*%s.bag"%(exp,config)))
    for idx in range(len(files[config])):
        df[config][idx] = import_bag(files[config][idx],samplesize,rolling)
        df[config][idx] = add_derivs(df[config][idx],d_topics)

        df[config][idx] = df[config][idx].iloc[(4000/samplesize):]

        lags_ms = determine_lags(df[config][idx],xtopics,ytopic,samplesize)
        print(lags_ms)
        df_shift = shift_lags(df[config][idx],xtopics,lags_ms)



        if idx == 0:
            # Unshifted
            X[config] = df[config][idx][xtopics].values
            y[config] = df[config][idx][ytopic].values
            groups[config] = np.full(len(X[config]),idx)
            # Shifted
            X_shift[config] = df_shift[xtopics].values
            y_shift[config] = df_shift[ytopic].values
            groups_shift[config] = np.full(len(df_shift[ytopic].values),idx)
        else:
            X[config] = np.concatenate((X[config], df[config][idx][xtopics].values))
            y[config] = np.concatenate((y[config], df[config][idx][ytopic].values))
            groups[config] = np.concatenate((groups[config], np.full(len(df[config][idx][ytopic].values),idx)))

            X_shift[config] = np.concatenate((X_shift[config], df_shift[xtopics].values))
            y_shift[config] = np.concatenate((y_shift[config], df_shift[ytopic].values))
            groups_shift[config] = np.concatenate((groups_shift[config], np.full(len(df_shift[ytopic].values),idx)))

    print(len(groups_shift[config]))
    print(len(X_shift[config]))
        # print(df[config][idx][xtopics].head())

n_exp = len(files[configs[0]])

# Print all the files with idx
print('idx   File')
for idx in range(len(files[configs[0]])):
    print('%-5s %-s'%(idx, files[configs[0]][idx]))

# Plot example of signals
idx = 0
for config in configs[0:]:
    topics = [[df[config][idx]['safety2'],df[config][idx]['performance2_3']],[df[config][idx]['narrowness1'],df[config][idx]['density1']]]
    titles = ['Experiment with config = %s'%(config),'Quality Attributes','Environment metrics']
    xlabel = 'Time [s]'
    ylabel = ['Quality Attributes','Environment Metrics']
    fig = graph21(topics, titles, xlabel, ylabel)
plt.show()

m = dict()
models = ['RF']
# scoring_metric = 'neg_mean_squared_error'
scoring_metric = 'r2'

m['SVR_linear'] = GridSearchCV(SVR(kernel='linear',gamma=0.1),
                     param_grid={"C": [1, 25, 50, 75, 100, 125, 150],
                                 "gamma": np.logspace(-2, 2, 5)}, 
                     n_jobs=-1)
                     # scoring=scoring_metric)
m['SVR_rbf'] = GridSearchCV(SVR(kernel='rbf',gamma=0.1),
                     param_grid={"C": [1, 25, 50, 75, 100, 125, 150],
                                 "gamma": np.logspace(-2, 2, 5)}, 
                     n_jobs=-1)
                     # scoring=scoring_metric)
m['SVR_poly'] = GridSearchCV(SVR(kernel='poly',gamma=0.1),
                     param_grid={"C": [50, 100, 150, 200, 250, 300],
                                 # "gamma": np.logspace(-2, 2, 5),
                                 "degree":[2,3,4]}, 
                     n_jobs=-1)
                     # scoring=scoring_metric)
# Random forrest
m['RF'] = GridSearchCV(RandomForestRegressor(),
                     param_grid={"n_estimators": [5, 10, 50, 100, 150, 200]}, 
                     n_jobs=-1)
                     # scoring=scoring_metric)
# Stochastic Gradient Descent
m['SGD'] = GridSearchCV(SGDRegressor(),
                     param_grid={"loss": ['squared_loss','huber','epsilon_insensitive'],
                                 "penalty":['l2','elasticnet'],
                                 "epsilon":[0.01,0.05,0.1,0.2]}, 
                     n_jobs=-1)
                     # scoring=scoring_metric)


# Train loop
# wb = Workbook()
# os.chdir(dir_results)
resultsfilename = 'Test_crossval_%s.xlsx'%exp
for config in configs:
    print('Training for %s'%config)

    # Create sheet for config
    # ws = wb.create_sheet(title=config)
    
    # Loop train models
    for modelname in models:
        # for idx in range(n_exp):
        gkf = GroupKFold(n_splits=n_exp)
        idm = 0
        # fig, ax = plt.subplots():
        for train, test in gkf.split(X[config], y[config], groups=groups[config]):
            # if idx == 5 and config == 'teb1' and modelname == 'SVR_poly':
            #     # I dont know why, but the script gets stuck with this combination. so skip it
            #     break

            # Fit model
            m[modelname].fit(X[config][train],y[config][train])
            # Score 
            r2score = m[modelname].score(X[config][test],y[config][test])
            r2score_own = m[modelname].score(X[config][train],y[config][train])
            
            # print('$%s_model= %s, idx=%s, own score = %s, average score = %s'%(ytopic,modelname,idx,round(r2score[modelname][idx][idx],4),round(np.average(r2score[modelname][idx]),4)))
            # Save model
            pkl_filename = dir_models + "%s_%s_%s_kfold%s.pkl"%(ytopic,modelname,config,idm)
            with open(pkl_filename, 'wb') as file:
                pickle.dump(m[modelname], file)

            print('idm:%s, modelname:%s test score:%s, own score:%s'%(idm,modelname,round(r2score,5),round(r2score_own,5)))
            idm += 1
        print('Shifted')
        # Shifted
        for train, test in gkf.split(X_shift[config], y_shift[config], groups=groups_shift[config]):
            # if idx == 5 and config == 'teb1' and modelname == 'SVR_poly':
            #     # I dont know why, but the script gets stuck with this combination. so skip it
            #     break

            # Fit model
            m[modelname].fit(X_shift[config][train],y_shift[config][train])
            # Score 
            r2score = m[modelname].score(X_shift[config][test],y_shift[config][test])
            r2score_own = m[modelname].score(X_shift[config][train],y_shift[config][train])
            
            # print('$%s_model= %s, idx=%s, own score = %s, average score = %s'%(ytopic,modelname,idx,round(r2score[modelname][idx][idx],4),round(np.average(r2score[modelname][idx]),4)))
            # Save model
            pkl_filename = dir_models + "%s_%s_%s_kfold%s_shifted.pkl"%(ytopic,modelname,config,idm)
            with open(pkl_filename, 'wb') as file:
                pickle.dump(m[modelname], file)

            print('idm:%s, modelname:%s test score:%s, own score:%s'%(idm,modelname,round(r2score,5),round(r2score_own,5)))
            idm += 1
            # # Parameters to results sheets
            # # column names, only on top
            # if idx == 0:
            #     parnames = ['','','','']
            #     i = 0
            #     for x in m[modelname].best_params_:
            #         parnames[i] = x
            #         i += 1
            #     ws.append(['Model name','Trainset number'] + parnames + ['average score','own score','All scores'])
            # # Parameter values
            # pars = ['','','','']
            # i = 0
            # for x in m[modelname].best_params_:
            #     pars[i] = str(m[modelname].best_params_[x])
            #     i += 1
            # ws.append(["%s_%s_%s_%s.pkl"%(ytopic,modelname,config,idx), idx] + pars + [round(np.average(r2score[modelname][idx]),3),round(r2score[modelname][idx][idx],3)] + [round(num, 3) for num in r2score[modelname][idx]])
            # wb.save(filename = resultsfilename)

# wb.save(filename = resultsfilename)
