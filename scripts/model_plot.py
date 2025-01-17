#!/usr/bin/env python3.6

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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle


def import_bag(file, bag_topics, print_head=False):
	df = rosbag_pandas.bag_to_dataframe(file, include=bag_topics)
	df.index -= df.index[0]
	df.index = pd.to_timedelta(df.index, unit='s')
	topics = [topic.replace('/metrics/','').replace('/data','') for topic in list(df.columns)]
	df.columns = topics

	df = df.groupby(level=0).mean()
	# df = df.resample('1ms').mean()
	df = df.resample('10ms').mean()
	df = df.interpolate(method='linear',limit_direction='both')
	df = df.rolling(100, min_periods=1).mean()

	if print_head:
		print(df.head())
	return df

def add_derivs(df,topics):
	for topic in topics:
		df['d_%s'%topic] = np.gradient(df[topic].values)/(df.index[1].total_seconds())
	return df

def graph_xcorr(df1,df2,title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2,1,1)
    ax.plot(df1.index.total_seconds(),df1.values/max(df1.dropna().values),label=df1.name)
    ax.plot(df2.index.total_seconds(),df2.values/max(df2.dropna().values),label=df2.name)
    ax.legend()
    # ax[0].set_title(title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signals')
    ax.set_title(title)

    ms = 2000
    step = 25
    lags = range(-int(ms),int(ms),step)
    rs = [crosscorr(df1,df2, lag) for lag in lags]
    ax = fig.add_subplot(2,1,2)
    ax.plot(lags,rs)
    ax.set_title('Cross correlation')
    ax.set_xlabel('lag [ms]')
    ax.set_ylabel('Correlation r')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,bottom=0.12)

    print(max(rs, key=abs))
    # fig.savefig(resultsfolder + title + '.png')

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

def train_linear_model(X,y):
	return LinearRegression().fit(X, y)


def determine_lags(df1,topics_shift,topic_ref):
	ms = 800
	step = 25
	lags = range(-int(ms),int(ms),step)

	lag_ms = []
	for topic_shift in topics_shift:
		rs = [abs(crosscorr(df1[topic_shift],df1[topic_ref], lag)) for lag in lags]
		rs_abs =  [abs(rsi) for rsi in rs] 
		lag_ms.append(np.argmax(rs_abs)*step - ms)
		print('For topic_shift: %s, Max rs = %s, at lag = %s'%(topic_shift,max(rs, key=abs),lag_ms[-1]))
	
	return lag_ms

def shift_lags(df,topics_shift,lag_ms):
	# Make a copy so the original will not be modified
	df1 = df.copy()

	for idx in range(len(lag_ms)):
		df1[topics_shift[idx]] = df1[topics_shift[idx]].shift(lag_ms[idx])
	df1 = df1.dropna()
	return df1



# import rosbag
import rospkg


from functions_postprocess import *


# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
dir_bags = path + '/bags'
dir_figs = path + '/figures/'
dir_models = path + '/models/'

# Experiment name and configs
exp = 'Experiment5'
# configs = ['dwa1','dwa2','teb1','teb2']
configs = ['teb1']

d_topics = ['density1','narrowness1']

xtopics = ['density1','d_density1','narrowness1','d_narrowness1','performance1']
ytopics = ['safety2','performance2_2']

# Resamplesize and smoothing (rolling)
samplesize = 10
rolling = 100

# TEB
model = dict()
model['safety2'] = 'safety2_RF_teb1_10_best.pkl'
model['performance2_2'] = 'performance2_2_SVR_rbf_teb1_0_best.pkl'

lags = dict()
lags['safety2'] = [-64, -39, 0, 65, 12]
lags['performance2_2'] = [ 72, 77, 84, 61, 113]

# Import Bag files into pandas
os.chdir(dir_bags)
files = dict()
df = dict()
df_shift = dict()
X = dict()
y = dict()
for config in configs:
    df[config] = dict()
    df_shift[config] = dict()
    X[config] = dict()
    y[config] = dict()

    files[config] = sorted(glob.glob("%s*%s.bag"%(exp,config)))
    for idx in range(len(files[config])):
        df[config][idx] = import_bag(files[config][idx],samplesize,rolling)
        df[config][idx] = add_derivs(df[config][idx],d_topics)
        df[config][idx] = df[config][idx].iloc[int(4000/samplesize):]

        X[config][idx] = dict()
        y[config][idx] = dict()
        df_shift[config][idx] = dict()
        for ytopic in ytopics:
            df_shift[config][idx][ytopic] = shift_lags(df[config][idx],xtopics,lags[ytopic])

            X[config][idx][ytopic] = df_shift[config][idx][ytopic][xtopics].values
            y[config][idx][ytopic] = df_shift[config][idx][ytopic][ytopic].values
n_exp = len(files[configs[0]])

# Print all the files with idx
print('idx   File')
for idx in range(len(files[configs[0]])):
    print('%-5s %-s'%(idx, files[configs[0]][idx]))

print('Load models')
m = dict()
for ytopic in ytopics:
	pkl_filename = dir_models + model[ytopic]
	with open(pkl_filename, 'rb') as file:
	    m[ytopic] = pickle.load(file)



colors = ['tab:blue','tab:orange']
# for idy in range(len(files)):
for config in configs:
	for idx in range(n_exp):
		for ytopic in ytopics:
			print('Predict')

			y1 = m[ytopic].predict(X[config][idx][ytopic])
			for i in range(len(y1)):
				y1[i] = min(y1[i],1)
			if ytopic == 'safety2':
				y0 = df_shift[config][idx][ytopic]['safety1'].values
			else:
				y0 = y[config][idx][ytopic]


			print("Plotting")
			fig, ax = plt.subplots()
			# Predictions
			ax.plot(df_shift[config][idx][ytopic].index.total_seconds(),y1, label='Model', color=colors[1])#, score = %s'%(round(m1.score(df[idy][xtopics].values,df[idy][ytopic].values),2)))
			# ax.plot(df[idy].index.total_seconds(),y2, label='TEB', color='tab:orange')#, score = %s'%(round(m1.score(df[idy][xtopics].values,df[idy][ytopic].values),2)))
			# Real
			ax.plot(df_shift[config][idx][ytopic].index.total_seconds(),y0, label='real', linestyle='--', color=colors[0])
			ax.legend(loc=0)
			ax.set_title('Model and real for %s \n, tested on run %s , config = %s \n rmse = %s'%(ytopic,idx,config,round(mean_squared_error(y0, y1),5)))
			# ax.set_ylim(0,1.2)
			# if idy == len(files_dwa2)-1:
			# print(y1)
			# print('rmse = %s'%(mean_squared_error(y, y1))
			plt.tight_layout()
			# Save fig
			# fig.savefig(dir_figs + 'Modelresult_teb1_train1_test%s'%idy + '.png')





plt.show()