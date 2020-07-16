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
from sklearn.svm import SVR


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

def graph_xcorr(df1,df2,title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2,1,1)
    ax.plot(df1.index.total_seconds(),df1.values,label=df1.name)
    ax.plot(df2.index.total_seconds(),df2.values,label=df2.name)
    ax.legend()
    # ax[0].set_title(title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signals')
    ax.set_title(title)

    ms = 800
    lags = range(-int(ms),int(ms),1)
    rs = [crosscorr(df1,df2, lag) for lag in lags]
    ax = fig.add_subplot(2,1,2)
    ax.plot(lags,rs)
    ax.set_title('Cross correlation')
    ax.set_xlabel('lag [ms]')
    ax.set_ylabel('Correlation r')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,bottom=0.12)

    print(max(max(rs),abs(min(rs))))
    # fig.savefig(resultsfolder + title + '.png')

def crosscorr(datax, datay, lag=0, wrap=False):
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

workspace = os.path.dirname(os.path.realpath(__file__))

bag_topics = ['/metrics/safety1','/metrics/safety2','/metrics/density1','/metrics/narrowness1']
# bag_topics = ['/metrics/density1']

# files_t = glob.glob("Experiment1*dwa2.bag")
files_v = glob.glob("Experiment4*.bag")
names_v = [file.replace('Experiment4_', '') for file in files_v]
files_t = ['Experiment1_2_dwa2.bag','Experiment1_0_teb2.bag']
files = files_t+files_v
print(files)
n = len(files)
n_t = len(files_t)
n_v = len(files_v)
print(n_t)


topicsb = ['safety1']
# topicsa = ['density1','density2','density3','density4']
# topicsa = ['narrowness1','narrowness2','narrowness3','narrowness4']
topicsa = ['density1','narrowness1']
topicss = []
for topica in topicsa:
	for topicb in topicsb:
		topicss.append([topica,topicb])

for file in files_t:
	df = import_bag(file,bag_topics)
	for topics in topicss:
		# title = 'Cross correlation of %s and %s \n for the config %s'%(topics[0],topics[1],config)
		title = ''
		graph_xcorr(df[topics[0]],df[topics[1]],title)

plt.show()
