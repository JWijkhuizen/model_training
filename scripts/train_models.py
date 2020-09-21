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
dir_bags = path + '/bags/'
dir_figs = path + '/figures/'
dir_models = path + '/models/'
dir_results = path + '/results/'

# Experiment name and configs
exp = 'exp6'
configs = ['dwa_v0_a0_b0','teb_v0_a0_b0']

# Topics
d_topics = ['obstacle_density','narrowness']
xtopics = ['obstacle_density','narrowness']
# ytopics = ['safety', 
ytopics = ['performance_dir']

# Models
polies = [1,2,3,4,5,6]

# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 1

# Experiment start and end
start_ms = 8000
end_ms = 1000

print('Load files variable with files to include')
pkl_filename = dir_bags + "files_incl_%s"%(exp)
with open(pkl_filename, 'rb') as file:
    files = pickle.load(file)

colors = ['tab:blue','tab:orange']
for ytopic in ytopics:
    # Import Bag files into pandas
    X, y, groups = generate_dataset_all_selectedfiles(files,configs,xtopics,ytopic,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling)

    # Train loop
    lr = LinearRegression(normalize=True)
    for config in configs:
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
                print('idm:%s, test score:%s, own score:%s, model poly:%s'%(idm,round(r2score,5),round(r2score_own,5),poly))

                if r2score > best_score:
                    best_model_name = "%s_Poly%s_%s_%s_best"%(ytopic,poly,config,idm)
                    model_best = lr
                    best_score = r2score
                    idm_best = idm
                if r2score > bestpolyscore:
                    bestpolyscore=r2score
                    bestpoly = poly
            idm += 1
            print("Best poly = %s"%bestpoly)

        # Save model
        path_config = rospack.get_path(config)
        path_model = path_config + "/quality_models"
        if not os.path.exists(path_model):
            os.makedirs(path_model)
        print(path_model)
        best_filename = path_model + "/" + ytopic + ".pkl"
        with open(best_filename, 'wb') as file:
            pickle.dump(model_best, file)


        print("Best Model:")
        print(best_model_name)
        idm = 0
        for train, test in gkf.split(X[config], y[config], groups=groups[config]):
            # if idm == idm_best:
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
            ax.set_title("Model (%s) test on unseen data \n for config: %s"%(ytopic,config))
            plt.tight_layout()

            # fig.savefig(dir_figs + 'model_%s_test_config_%s'%(ytopic,config) + '.png')
            idm+=1

plt.show()
