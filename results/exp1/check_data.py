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
from statsmodels.nonparametric.smoothers_lowess import lowess

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

# Experiment1_0_teb2.bag
# Experiment1_2_dwa2.bag

print("Import data")
#Import data
X = dict()
y1 = dict()
y2 = dict()
index = dict()
for idx in range(n):
	print(files[idx])
	df1 = import_bag(files[idx], bag_topics)
	d_density1 = pd.Series(np.gradient(df1['density1'].values), df1.index, name='d_density1')
	d_narrowness1 = pd.Series(np.gradient(df1['narrowness1'].values), df1.index, name='d_narrowness1')
	data = pd.concat([df1['safety1'],df1['safety2'], df1['density1'], df1['narrowness1'], d_narrowness1/0.001, d_density1/0.001], axis=1)
	# data = data.dropna()
	# X[idx] = data[['density1','narrowness1','d_density1','d_narrowness1']].values
	X[idx] = data[['density1','narrowness1','d_density1','d_narrowness1']].values
	y2[idx] = data['safety2'].values
	y1[idx] = data['safety1'].values
	index[idx] = data.index
	print(df1.head(100))



print("Plotting")
# ax.set_xticks(labels)
# ax.set_title('R2 = ' + str(reg.score(X[idx], y[idx])))

# for idv in range(n_t,n_t+n_v,1):
for idx in [3]:
	fig, ax = plt.subplots()
	# for idx in range(n_t):
	ax.plot(data.index.total_seconds(), data['density1'], label='density1')
	ax.plot(data.index.total_seconds(), data['d_density1'], label='d_density1')
	ax.legend(loc=0)

	fig, ax = plt.subplots()
	# for idx in range(n_t):
	ax.plot(data.index.total_seconds(), data['narrowness1'], label='density1')
	ax.plot(data.index.total_seconds(), data['d_narrowness1'], label='d_density1')
	ax.legend(loc=0)





plt.show()