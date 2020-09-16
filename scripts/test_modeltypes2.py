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
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
import pickle

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
# configs = ['cdwa_v0_a0_b0','cteb_v0_a0_b0']
configs = ['cdwa1']

# Topics
d_topics = ['density1_f','narrowness1']
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics]
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics] + ['performance2']
xtopics = ['density1_f','narrowness1','performance2']
# ytopic = 'performance3'
ytopic = 'performance4'

# Models
models = ['SGD','ElasticNet','Lasso','Ridge','SVR_linear','SVR_rbf','SVR_poly','RF']
polies = [1,2,3,4,5,6]

# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 1

# Experiment start and end
start_ms = 13000
end_ms = 1000

print('Load files variable with files to include')
pkl_filename = dir_bags + "files_incl_%s"%(exp)
with open(pkl_filename, 'rb') as file:
    files = pickle.load(file)

colors = ['tab:blue','tab:orange']

# Import Bag files into pandas
X, y, groups = generate_dataset_all_selectedfiles(files,configs,xtopics,ytopic,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling)
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
                best_model_name = "%s_Poly%s_%s_%s_best"%(ytopic,poly,config,idm)
                best_filename = dir_models + best_model_name + ".pkl"
                model_best = lr
                best_score = r2score
                idm_best = idm
            if r2score > bestpolyscore:
                bestpolyscore=r2score
                bestpoly = poly
        idm += 1
        print("Best poly = %s"%bestpoly)
    # path_config = rospack.get_path(config)
    with open(best_filename, 'wb') as file:
        pickle.dump(model_best, file)


    print("Best Model:")
    print(best_model_name)
    idm = 0
    for train, test in gkf.split(X[config], y[config], groups=groups[config]):
        if idm == idm_best:
            y1 = model_best.predict(Xp[test])
            for i in range(len(y1)):
                y1[i] = min(y1[i],1)
            yr = y[config][test]
            for i in range(len(yr)):
                yr[i] = min(yr[i],1)

            # Scoring metrics
            print("mean squared error       = %s"%mean_squared_error(yr,y1))
            print("explained variance score = %s"%explained_variance_score(yr,y1))
            print("r2 score                 = %s"%r2_score(yr,y1))

            print(len(yr))
            t = np.linspace(0,(len(yr)-1)/100,len(yr))
            print(t[2]-t[1])
            print(t)
            fig, ax = plt.subplots()
            ax.plot(t, y1, label='Model', color=colors[1])#, score = %s'%(round(m1.score(df[idy][xtopics].values,df[idy][ytopic].values),2)))
            ax.plot(t, yr, label='real', linestyle='--', color=colors[0])
            ax.legend(loc=0)
            # ax.set_title('Best safety model and real %s \n trained on run 1, tested on run %s , config = %s \n rmse = %s'%(ytopic,idy,config,round(mean_squared_error(y, y1),5)))
            ax.set_ylim(0,1.2)
            ax.set_ylabel('QA')
            ax.set_xlabel("Time [s]")
            ax.set_title("Comparison of modelled and real Performance")
            plt.tight_layout()

        idm+=1

plt.show()
