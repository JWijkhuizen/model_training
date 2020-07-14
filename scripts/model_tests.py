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

files = glob.glob("*teb2.bag")
n = len(files)
# n = 30

X = dict()
y = dict()

# Import data
for idx in range(n):
	df1 = import_bag(files[idx], bag_topics)
	d_density1 = pd.Series(np.gradient(df1['density1'].values), df1.index, name='d_density1')
	d_narrowness1 = pd.Series(np.gradient(df1['narrowness1'].values), df1.index, name='d_narrowness1')
	data = pd.concat([df1['safety1'], df1['performance1'], df1['density1'], df1['narrowness1'], d_narrowness1/0.001, d_density1/0.001], axis=1)
	data = data.dropna()
	X[idx] = data[['performance1','density1','narrowness1','d_density1','d_narrowness1']].values
	y[idx] = data['safety1'].values



print("Fitting with Linear Regression method")
reg = dict()
r2 = dict()
r2_mean = []
r2_std = []
r2_id = []
for idx in [99]:
	reg[idx] = LinearRegression().fit(X[idx], y[idx])
	r2[idx] = []
	for idy in range(n):
		r2[idx].append(reg[idx].score(X[idy], y[idy]))
	print(min(r2[idx]))
	if stat.mean(r2[idx]) > 0:
		r2_mean.append(stat.mean(r2[idx]))
		r2_std.append(stat.stdev(r2[idx]))
		r2_id.append(idx)

print("Fitting with Ridge method")
reg2 = Ridge()
reg2.fit(X[99],y[99])

print("Fitting with SVR method")
m3 = SVR()
m3.fit(X[10],y[10])

print("Plotting")
# ax.set_xticks(labels)
# ax.set_title('R2 = ' + str(reg.score(X[idx], y[idx])))

idv = 63
for idx in [r2_id[np.argmax(r2_mean)]]:
	if idx > n-1:
		idx = 99
	fig, ax = plt.subplots()
	ax.plot(reg[idx].predict(X[idv]), label='Linear model')
	ax.plot(reg2.predict(X[idv]), label='Ridge model')
	ax.plot(m3.predict(X[idv]), label='SVC model')
	ax.plot(y[idv], label='real')
	ax.legend(loc=0)
	ax.set_title('model_n = %s , R2 = '%idx + str(reg[idx].score(X[idv], y[idv])))


plt.show()