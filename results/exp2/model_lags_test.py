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
from sklearn.svm import SVR
import pickle


def import_bag(file, bag_topics, print_head=False):
	df = rosbag_pandas.bag_to_dataframe(file, include=bag_topics)
	df.index -= df.index[0]
	df.index = pd.to_timedelta(df.index, unit='s')
	topics = [topic.replace('/metrics/','').replace('/data','') for topic in list(df.columns)]
	df.columns = topics

	df = df.groupby(level=0).mean()
	# df = df.resample('1ms').mean()
	df = df.resample('1ms').mean()
	df = df.interpolate(method='linear',limit_direction='both')
	df = df.rolling(800, min_periods=1).mean()

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


workspace = os.path.dirname(os.path.realpath(__file__))

bag_topics = ['/metrics/safety1','/metrics/safety2','/metrics/density1','/metrics/narrowness1']
d_topics = ['density1','narrowness1']


files = glob.glob("*.bag")
print(files)
n = len(files)


print("Import data")
df = dict()
for idx in range(n):
	print(files[idx])
	df[idx] = import_bag(files[idx], bag_topics)
	df[idx] = add_derivs(df[idx],d_topics)


print('Train and predict')	



xtopics = ['density1','d_density1','narrowness1','d_narrowness1']
ytopic = 'safety2'

idx = 1

m1 = SVR().fit(df[idx][xtopics].values,df[idx][ytopic].values)
print('Score = %s'%m1.score(df[idx][xtopics].values,df[idx][ytopic].values))
# y1 = m1.predict(df[idy][xtopics].values)

pkl_filename = "model_dwa.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(m1, file)
# pkl_filename = "model_dwa.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(m1, file)

# lag_ms = determine_lags(df[idx],xtopics,ytopic)
# df_lags[idx] = shift_lags(df[idx],xtopics,lag_ms)
# df_lags[idy] = shift_lags(df[idy],xtopics,lag_ms)
# m1_l = SVR().fit(df_lags[idx][xtopics].values,df_lags[idx][ytopic].values)
# print('Score = %s'%m1_l.score(df[idx][xtopics].values,df[idx][ytopic].values))
# y1_l = m1.predict(df_lags[idy][xtopics].values)

yv = df[idy][ytopic].values

print("Plotting")
# Plot all the signals
# for i in [idx,idy]:
# 	fig,ax = plt.subplots()
# 	for topic in xtopics:
# 		ax.plot(df[i][topic], label=topic)
# 	ax.plot(df[i][ytopic], label=ytopic)
# 	ax.legend()
# 	ax.set_title(files[i])

# Plot the cross correlations
# for topic in xtopics:
# 	graph_xcorr(df[idx][ytopic],df[idx][topic],topic)

# Predictions
# fig, ax = plt.subplots()
# ax.plot(df[idy].index.total_seconds(),y1, label='y1, score = %s'%(round(m1.score(df[idy][xtopics].values,df[idy][ytopic].values),2)))
# ax.plot(df_lags[idy].index.total_seconds(),y1_l, label='y1 lags, score = %s'%(round(m1_l.score(df[idy][xtopics].values,df[idy][ytopic].values),2)))
# # Real
# ax.plot(df[idy].index.total_seconds(),yv, label='real', linestyle='--')
# ax.legend(loc=0)
# ax.set_title(files[idv])
# ax.set_ylim(0,1.2)

print(m1.get_params())

plt.show()