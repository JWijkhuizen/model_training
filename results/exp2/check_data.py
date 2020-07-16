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

workspace = os.path.dirname(os.path.realpath(__file__))

bag_topics = ['/metrics/closeness','/metrics/safety1','/metrics/safety2','/metrics/density1','/metrics/narrowness1']
d_topics = ['closeness','density1','narrowness1']


files = glob.glob("*.bag")
print(files)
n = len(files)


print("Import data")
df = dict()
for idx in range(n):
	print(files[idx])
	df[idx] = import_bag(files[idx], bag_topics)
	df[idx] = add_derivs(df[idx],d_topics)
	




print("Plotting")

# for idx in [3]:
# 	fig, ax = plt.subplots()
# 	ax.plot(df[idx].index.total_seconds(), df[idx].closeness/max(df[idx].closeness.dropna()), label='closeness')
# 	ax.plot(df[idx].index.total_seconds(), df[idx].safety1, label='safety1')
# 	ax.legend(loc=0)


topics_i = ['closeness','d_closeness','density1','d_density1','narrowness1','d_narrowness1']
topics_o = ['safety1']
idx = 1
for topic_i in topics_i:
	for topic_o in topics_o:
		title = 'Cross correlation of %s and %s'%(topic_i,topic_o)
		graph_xcorr(df[idx][topic_o],df[idx][topic_i],title)



plt.show()