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
	df = df.interpolate(method='linear')

	if print_head:
		print(df.head())
	return df

workspace = os.path.dirname(os.path.realpath(__file__))

bag_topics = ['/metrics/performance1','/metrics/time','/metrics/safety2','/metrics/density1','/metrics/density4','/metrics/narrowness1']

files_t = glob.glob("Experiment1*teb2.bag")
files_v = glob.glob("Experiment4*teb2.bag")
files = files_t+files_v
print(files)
n = len(files)
# n = 30

X = dict()
y = dict()
for idx in range(n):
	print(files[idx])
	df1 = import_bag(files[idx], bag_topics)
	d_density1 = pd.Series(np.gradient(df1['density1'].values), df1.index, name='d_density1')
	d_narrowness1 = pd.Series(np.gradient(df1['narrowness1'].values), df1.index, name='d_narrowness1')
	data = pd.concat([df1['safety2'],df1['performance1'], df1['density1'], df1['narrowness1'], d_narrowness1/0.001, d_density1/0.001], axis=1)
	data = data.dropna()
	# X[idx] = data[['density1','narrowness1','d_density1','d_narrowness1']].values
	X[idx] = data[['density1','narrowness1','d_density1','d_narrowness1']].values
	y[idx] = data['safety2'].values


n = 3
m1 = dict()
m2 = dict()
m3 = dict()

for idx in range(n):
	print("Fitting with Linear Regression method")
	m1[idx] = LinearRegression().fit(X[idx], y[idx])
	# print("Fitting with Ridge method")
	# m2[idx] = Ridge().fit(X[idx],y[idx])
	# print("Fitting with SVR method")
	# m3[idx] = SVR().fit(X[idx],y[idx])

print("Plotting")
# ax.set_xticks(labels)
# ax.set_title('R2 = ' + str(reg.score(X[idx], y[idx])))

for idv in [1]:
	for idx in [1]:
		fig, ax = plt.subplots()
		ax.plot(m1[idx].predict(X[idv]), label='Linear model')
		# ax.plot(m2[idx].predict(X[idv]), label='Ridge model')
		# ax.plot(m3[idx].predict(X[idv]), label='SVC model')
		ax.plot(y[idv], label='real')
		ax.legend(loc=0)
		ax.set_title('model_n = %s , R2 = '%idx + str(m1[idx].score(X[idv], y[idv])))
		ax.set_ylim(0,1.1)


plt.show()