#!/usr/bin/env python
import rosbag_pandas
import pandas as pd

def import_bag(file, bag_topics, print_head=False):
	df = rosbag_pandas.bag_to_dataframe(file, include=bag_topics)
	df.index -= df.index[0]
	df.index = pd.to_timedelta(df.index, unit='s')

	topics = sorted([topic.replace('/metrics/','').replace('/data','') for topic in bag_topics])
	df.columns = topics
	df = df.groupby(level=0).mean()
	df = df.resample('1ms').mean()
	df = df.interpolate(method='linear')

	if print_head:
		print(df.head())
	return df