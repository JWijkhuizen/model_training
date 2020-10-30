#!/usr/bin/env python3.6

import rospkg
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

import pickle

from functions_postprocess import *

# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
path2 = rospack.get_path('simulation_tests')
dir_bags = path2 + '/bags/'
dir_figs = path + '/figures/'
dir_models = path + '/models/'
dir_results = path + '/results/'

# Experiment name and configs
exp = 't2'
# configs = ['teb_v0_a0_b0','dwa_v0_a0_b0','teb_v1_a0_b0','dwa_v1_a0_b0']
configs = ["dwa_v1_a0_b0", "dwa_v1_a1_b0", "dwa_v1_a0_b1", "dwa_v1_a1_b1", "dwa_v2_a0_b0", "dwa_v2_a1_b0", "dwa_v2_a0_b1", "dwa_v2_a1_b1", "teb_v1_a0_b0", "teb_v1_a1_b0", "teb_v1_a0_b1", "teb_v1_a1_b1", "teb_v2_a0_b0", "teb_v2_a1_b0", "teb_v2_a0_b1", "teb_v2_a1_b1"]
# configs = ["dwa_v0_a0_b0", "teb_v0_a0_b0"]
# Topics
d_topics = []
xtopics = ['obstacle_density21','narrowness1']
ytopics = ['safety']

# print_output = True
print_output = True
# plot_model = True
plot_model = True

# Models
# polies = [1,2,3,4,5,6]
polies = [5]

# Resamplesize and smoothing (rolling)
samplesize = 100
rolling = 1

save_models = True

# print('Load files variable with files to include')
# pkl_filename = dir_bags + "files_incl_%s"%(exp)
# with open(pkl_filename, 'rb') as file:
#     files = pickle.load(file)

wb = Workbook()
ws = wb.active


r2scores = dict()
colors = ['tab:blue','tab:orange']
for ytopic in ytopics:
    r2scores[ytopic] = dict()
    r2score_array = []
    mse_array = []
    # r2scores_std[ytopic] = dict()
    # Import Bag files into pandas
    X, y, groups = generate_dataset_all(configs,xtopics,ytopic,d_topics,exp,dir_bags,samplesize,rolling)

    # Train loop
    lr = LinearRegression(normalize=True)
    for config in configs:
        r2scores[ytopic] = []
        gkf = GroupKFold(n_splits=int(groups[config][-1])+1)

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
                if print_output: print('idm:%s, test score:%s, own score:%s, model poly:%s'%(idm,round(r2score,5),round(r2score_own,5),poly))

                if r2score > best_score:
                    best_model_name = "%s_Poly%s_%s_%s_best"%(ytopic,poly,config,idm)
                    model_best = lr
                    best_score = r2score
                    idm_best = idm
                if r2score > bestpolyscore:
                    bestpolyscore=r2score
                    bestpoly = poly
            idm += 1
            if print_output: print("Best poly = %s"%bestpoly)

        # Save model
        path_config = rospack.get_path(config)
        path_model = path_config + "/quality_models"
        if not os.path.exists(path_model):
            os.makedirs(path_model)
        # print(path_model)
        if save_models:
            best_filename = path_model + "/" + ytopic + ".pkl"
            with open(best_filename, 'wb') as file:
                pickle.dump(model_best, file)


        # print("Best Model:")
        # print(best_model_name)
        idm = 0
        for train, test in gkf.split(X[config], y[config], groups=groups[config]):
            if idm == idm_best:
                y1 = model_best.predict(Xp[test])
                # for i in range(len(y1)):
                #     y1[i] = min(y1[i],1)
                yr = y[config][test]
                # for i in range(len(yr)):
                #     yr[i] = min(yr[i],1)

                # Scoring metrics
                # print("mean squared error       = %s"%mean_squared_error(yr,y1))
                print("explained variance score = %s"%explained_variance_score(yr,y1))
                print("%s r2 score = %s, mse = %s"%(config,round(r2_score(yr,y1),4),round(mean_squared_error(yr,y1),4)))
                r2score_array.append(round(r2_score(yr,y1),4))
                mse_array.append(round(mean_squared_error(yr,y1),4))
                r2scores[ytopic].append(r2_score(yr,y1))

                if plot_model:
                    # if config == 'dwa_v1_a0_b0':
                    t = np.linspace(0,(len(yr)-1)/10,len(yr))
                    fig, ax = plt.subplots(figsize=[12.8,4.8])
                    ax.plot(t, y1, label='Predicted', color=colors[1])#, score = %s'%(round(m1.score(df[idy][xtopics].values,df[idy][ytopic].values),2)))
                    ax.plot(t, yr, label='Measured', linestyle='--', color=colors[0])
                    ax.legend(loc=0)
                    # ax.set_title('Best safety model and real %s \n trained on run 1, tested on run %s , config = %s \n rmse = %s'%(ytopic,idy,config,round(mean_squared_error(y, y1),5)))
                    ax.set_ylim(0,1.1)
                    # ax.set_xlim(0,15)
                    ax.set_ylabel('Safety level')
                    ax.set_xlabel("Time (s)")
                    ax.set_title("Safety model for %s"%(config))
                    plt.tight_layout()

                    # fig.savefig(dir_figs + 'model_%s_test_config_%s'%(ytopic,config) + '.png')
            idm+=1
    ws.append(["","r2score","mse"])
    idy = 0
    for config in configs:
        ws.append([config, r2score_array[idy], mse_array[idy]])
        idy+=1


    # fig, ax = plt.subplots()
    # ax.bar(configs, r2scores[ytopic])#, score = %s'%(round(m1.score(df[idy][xtopics].values,df[idy][ytopic].values),2)))
    # ax.set_title('r2scores of the %s model for all configurations'%(ytopic))
    # ax.set_ylabel('r2scores')
    # ax.set_ylim(0.85,1.0)
    # plt.xticks(rotation=90)

os.chdir(dir_results)
wb.save(filename = 'models_scores_table.xlsx')

plt.show()
