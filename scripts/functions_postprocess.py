#!/usr/bin/env python

import rosbag
import rosbag_pandas
import rospy
import rosbag_pandas
import matplotlib.pyplot as plt
from openpyxl import Workbook
import pandas as pd
import numpy as np


def import_bag(file, samplesize, rolling, bag_topics=None, print_head=False):
    df = rosbag_pandas.bag_to_dataframe(file, include=bag_topics)
    df.index -= df.index[0]
    df.index = pd.to_timedelta(df.index, unit='s')
    topics = [topic.replace('/metrics/','').replace('/data','') for topic in list(df.columns)]
    df.columns = topics

    df = df.groupby(level=0).mean()
    # print('Amount of data points before resampling = %s'%len(df))
    df = df.resample('%sms'%samplesize).mean()
    # print('Amount of data points after resampling = %s'%len(df))
    df = df.interpolate(method='linear',limit_direction='both')
    df = df.rolling(rolling, min_periods=1).mean()

    if print_head:
        print(df.head())
    return df

def add_derivs(df,topics):
    for topic in topics:
        df['d_%s'%topic] = np.gradient(df[topic].values)/(df.index[1].total_seconds())
    return df

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

def graph(dfs, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    idx = 0
    for df in dfs:
    	ax.plot(df.index.total_seconds(), df.values, label=df.name)
    	idx += 1
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=0)
    # print(save)
    return fig
    # fig.savefig(resultsfolder + title + '.png')

def graph21(dfs, titles, xlabel, ylabels):
    n = len(dfs)
    fig, axes = plt.subplots(n, sharex=True)
    idx = 0
    for df in dfs:
        if type(df) == list:
            for df_i in df:
                axes[idx].plot(df_i.index.total_seconds(), df_i.values, label=df_i.name)
        else:
            axes[idx].plot(df.index.total_seconds(), df.values, label=df.name)
        axes[idx].set_ylabel(ylabels[idx])
        axes[idx].legend()
        # if len(titles) > idx: axes[idx].set_title(titles[idx])
        idx += 1
    axes[0].set_title(titles[0])
    axes[n-1].set_xlabel(xlabel)
    fig.tight_layout()

    return fig
    # fig.savefig(resultsfolder + title + '.png')

def graph22(x,y,dfs, titles, xlabels, ylabels, fig):
    n = len(dfs)
    idx = 1
    yi = 0
    for df in dfs:
        ax = fig.add_subplot(y, x, idx)
        ax.plot(df.dropna().index.total_seconds(), df.dropna().values, label=df.name)
        if (idx-1)%x == 0:
            ax.set_ylabel(ylabels[yi])
            yi += 1
        ax.set_title(titles[idx])
        ax.set_xlabel(xlabels[idx-1])
        idx += 1
    fig.suptitle(titles[0], fontsize=16)
    fig.tight_layout()

    return fig
    # fig.savefig(resultsfolder + titles[0] + '.png')

def graph_xcorr(df1,df2,title):
    fig = plt.figure()#figsize=(10, 10))
    ax = fig.add_subplot(2,1,1)
    ax.plot(df1.index.total_seconds(),df1.values,label=df1.name)
    ax.plot(df2.index.total_seconds(),df2.values,label=df2.name)
    ax.legend()
    # ax[0].set_title(title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signals')
    ax.set_title(title)

    ms = 3000
    step = 25
    lags = range(-int(ms),int(ms),step)
    rs = [crosscorr(df1,df2, lag) for lag in lags]
    ax = fig.add_subplot(2,1,2)
    ax.plot(lags,rs)
    ax.set_title('Cross correlation')
    ax.set_xlabel('lag [ms]')
    ax.set_ylabel('Correlation r')
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85,bottom=0.12)

    print(max(rs, key=abs))

    return fig
    # fig.savefig(resultsfolder + title + '.png')

def crosscorr(datax, datay, lag=0, wrap=False):
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))