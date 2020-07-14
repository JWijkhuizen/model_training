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
	df = df.interpolate(method='linear')

	if print_head:
		print(df.head())
	return df

workspace = os.path.dirname(os.path.realpath(__file__))

bag_topics = ['/metrics/performance1','/metrics/time','/metrics/safety1','/metrics/safety2','/metrics/density1','/metrics/density2','/metrics/density3','/metrics/density4','/metrics/narrowness1']

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


#Import data
X = dict()
y1 = dict()
y2 = dict()
index = dict()
for idx in range(n):
	print(files[idx])
	df1 = import_bag(files[idx], bag_topics)
	d_density1 = pd.Series(np.gradient(df1['density1'].values), df1.index, name='d_density1')
	d_density3 = pd.Series(np.gradient(df1['density3'].values), df1.index, name='d_density3')
	d_narrowness1 = pd.Series(np.gradient(df1['narrowness1'].values), df1.index, name='d_narrowness1')
	data = pd.concat([df1['safety1'],df1['safety2'],df1['performance1'], df1['density1'], df1['narrowness1'], d_narrowness1/0.001, d_density1/0.001], axis=1)
	data = data.dropna()
	# X[idx] = data[['density1','narrowness1','d_density1','d_narrowness1']].values
	X[idx] = data[['density1','narrowness1','d_density1','d_narrowness1']].values
	y2[idx] = data['safety2'].values
	y1[idx] = data['safety1'].values
	index[idx] = data.index

	# fig, ax = plt.subplots()
	# ax.plot(data['density1'].values)



m1 = dict()
m2 = dict()
m3 = dict()

for idx in range(n_t):
	print("Fitting with Linear Regression method")
	m1[idx] = LinearRegression().fit(X[idx], y2[idx])
	# print("Fitting with Ridge method")
	# m2[idx] = Ridge().fit(X[idx],y[idx])
	print("Fitting with SVR method")
	m3[idx] = SVR().fit(X[idx],y2[idx])

print("Plotting")
# ax.set_xticks(labels)
# ax.set_title('R2 = ' + str(reg.score(X[idx], y[idx])))

# for idv in range(n_t,n_t+n_v,1):
for idv in [3, 5]:
	fig, ax = plt.subplots()
	# for idx in range(n_t):
	# ax.plot(m1[0].predict(X[idv]), label='Linear model DWA')
	# ax.plot(m1[1].predict(X[idv]), label='Linear model TEB')
	ax.plot(lowess(m3[0].predict(X[idv]),index[idv],frac=0.01), label='SVR model DWA')
	ax.plot(lowess(m3[1].predict(X[idv]),index[idv],frac=0.01), label='SVR model TEB')
	# ax.plot(m3[0].predict(X[idv]), label='SVR model DWA')
	# ax.plot(m3[1].predict(X[idv]), label='SVR model TEB')
		# ax.plot(m2[idx].predict(X[idv]), label='Ridge model')
		# ax.plot(m3[idx].predict(X[idv]), label='SVC model')
	ax.plot(y1[idv], label='real %s'%names_v[idv-2], linestyle='--')
	ax.legend(loc=0)
	# ax.set_title('model_n = %s , R2 = '%idx + str(m1[idx].score(X[idv], y1[idv])))
	ax.set_title(names_v[idv-2])
	ax.set_ylim(0,1.5)




plt.show()