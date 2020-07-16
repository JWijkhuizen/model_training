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
from sklearn.ensemble import RandomForestRegressor
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

def crosscorr(datax, datay, lag=0, wrap=False):
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))



workspace = os.path.dirname(os.path.realpath(__file__))

bag_topics = ['/metrics/closeness','/metrics/performance1','/metrics/time','/metrics/safety1','/metrics/safety2','/metrics/density1','/metrics/density2','/metrics/density3','/metrics/density4','/metrics/narrowness1']

# files_t = glob.glob("Experiment1*dwa2.bag")
files_v = ['Experiment2_1_dwa2.bag', 'Experiment2_1_teb2.bag']
names_v = [file.replace('Experiment4_', '') for file in files_v]
files_t = ['Experiment2_2_dwa2.bag','Experiment2_2_teb2.bag']
files = files_t+files_v
print(files)
n = len(files)
n_t = len(files_t)
n_v = len(files_v)
print(n_t)

# Experiment1_0_teb2.bag
# Experiment1_2_dwa2.bag

xtopics = ['closeness','density1','narrowness1','d_density1','d_narrowness1']
xtopics2 = ['density1','narrowness1','d_density1','d_narrowness1']

#Import data
X = dict()
X_lags = dict()
y1 = dict()
y2 = dict()
index = dict()

for idx in range(n):
	print(files[idx])
	df1 = import_bag(files[idx], bag_topics)
	d_density1 = pd.Series(np.gradient(df1['density1'].values), df1.index, name='d_density1')
	d_narrowness1 = pd.Series(np.gradient(df1['narrowness1'].values), df1.index, name='d_narrowness1')
	data = pd.concat([df1['closeness'],df1['safety1'],df1['safety2'],df1['performance1'], df1['density1'], df1['narrowness1'], d_narrowness1/0.001, d_density1/0.001], axis=1)
	if idx in [0,1]:
		ms = 800
		step = 25
		lags = range(-int(ms),int(ms),step)
		X_lags[idx] = []
		for xtopic in xtopics:
			rs = [abs(crosscorr(data[xtopic],data['safety1'], lag)) for lag in lags]
			max_rs = max(rs)
			max_rs_id = np.argmax(rs)*step - ms
			print('For %s, max correlation = %s, at lag = %s'%(xtopic,max_rs,max_rs_id))
			X_lags[idx].append(max_rs_id)

	X[idx] = data[xtopics]
	y2[idx] = data['safety2']
	y1[idx] = data['safety1']



m1 = dict()
m12 = dict()
m2 = dict()
m3 = dict()
m32 = dict()

index = dict()
pr1 = dict()
pr12 = dict()
pr2 = dict()
pr3 = dict()
pr32 = dict()

print('Training')
for idx in range(n_t):
	idrs = 0
	datat = pd.concat([X[idx], y1[idx], y2[idx]], axis=1)
	for xtopic in xtopics:
		datat[xtopic] = datat[xtopic].shift(X_lags[idx][idrs])
		idrs += 1
	datat = datat.dropna()

	Xt = datat[xtopics]
	y2t = datat['safety2']
	y1t = datat['safety1']

	print("Fitting with Linear Regression method")
	m1[idx] = LinearRegression().fit(Xt.values, y2t.values)
	m12[idx] = LinearRegression().fit(Xt[xtopics2].values, y2t.values)
	# print("Fitting with Ridge method")
	# m2[idx] = Ridge().fit(X[idx],y[idx])
	print("Fitting with SVR method")
	m3[idx] = SVR().fit(Xt.values,y2t.values)
	m32[idx] = SVR().fit(Xt[xtopics2].values,y2t.values)
	# print("Fitting with random forest thing")
	# m4[idx] = RandomForestRegressor(n_estimators=100).fit(X[idx], y2[idx])

	print('Predicting')
	pr1[idx] = dict()
	pr12[idx] = dict()
	pr2[idx] = dict()
	pr3[idx] = dict()
	pr32[idx] = dict()
	index[idx] = dict()
	for idv in [2, 3]:
		idrs = 0
		Xt = X[idv]
		for xtopic in xtopics:
			Xt[xtopic] = Xt[xtopic].shift(X_lags[idx][idrs])
			idrs += 1
		Xt = Xt.dropna()

		index[idx][idv] = Xt
		pr1[idx][idv] = m1[idx].predict(Xt.values)
		pr12[idx][idv] = m12[idx].predict(Xt[xtopics2].values)
		# pr2[idx][idv] = m2[idx].predict(Xt.values)
		pr3[idx][idv] = m3[idx].predict(Xt.values)
		pr32[idx][idv] = m32[idx].predict(Xt[xtopics2].values)


print("Plotting")
# for idv in range(n_t,n_t+n_v,1):
for idv in [2, 3]:
	fig, ax = plt.subplots()

	# Linear model
	# ax.plot(index[0][idv].index.total_seconds(),pr1[0][idv], label='Linear model DWA')
	# ax.plot(index[1][idv].index.total_seconds(),pr1[1][idv], label='Linear model TEB')
	
	# ax.plot(index[0][idv].index.total_seconds(),pr12[0][idv], label='Linear model DWA 2')
	# ax.plot(index[1][idv].index.total_seconds(),pr12[1][idv], label='Linear model TEB 2')
	
	# SVR model
	ax.plot(index[0][idv].index.total_seconds(),pr3[0][idv], label='SVR model DWA')
	ax.plot(index[1][idv].index.total_seconds(),pr3[1][idv], label='SVR model TEB')

	ax.plot(index[0][idv].index.total_seconds(),pr32[0][idv], label='SVR model DWA')
	ax.plot(index[1][idv].index.total_seconds(),pr32[1][idv], label='SVR model TEB')


	# Real
	ax.plot(y1[idv].index.total_seconds(),y1[idv].values, label='real %s'%names_v[idv-2], linestyle='--')
	
	ax.legend(loc=0)
	ax.set_title(files[idv])
	ax.set_ylim(0,1.2)

# for idx in [3]:
# 	fig, ax = plt.subplots()
# 	# for idx in range(n_t):
# 	ax.plot(data.index.total_seconds(), data['density1'], label='density1')
# 	ax.plot(data.index.total_seconds(), data['d_density1'], label='d_density1')
# 	ax.legend(loc=0)

# 	fig, ax = plt.subplots()
# 	# for idx in range(n_t):
# 	ax.plot(data.index.total_seconds(), data['narrowness1'], label='density1')
# 	ax.plot(data.index.total_seconds(), data['d_narrowness1'], label='d_density1')
# 	ax.legend(loc=0)


plt.show()