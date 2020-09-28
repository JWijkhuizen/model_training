#!/usr/bin/env python

import rosbag
import rosbag_pandas
import rospy
import rosbag_pandas
import matplotlib.pyplot as plt
from openpyxl import Workbook
import pandas as pd
import numpy as np
import os
import glob


def import_bag(file, samplesize, rolling, bag_topics=None, print_head=False):
    df = rosbag_pandas.bag_to_dataframe(file, include=bag_topics)
    df.index -= df.index[0]
    df.index = pd.to_timedelta(df.index, unit='s')
    # print(df.columns)
    topics = [topic.replace('/metrics/','').replace('/data','') for topic in list(df.columns)]
    df.columns = topics
    # print(topics)
    # print("length OD=%s"%len(df['obstacle_density21'].dropna()))
    # print("length N=%s"%len(df['narrowness1'].dropna()))
    # print("length S=%s"%len(df['safety'].dropna()))
    # print(df['time'].dropna())
    # print(df['start_end'].dropna())
    start = df.loc[df['start_end'] == "start"].index[0].total_seconds()
    end = df.loc[df['start_end'] == "end"].index[0].total_seconds()


    df = df.groupby(level=0).mean()
    df = df.resample('%sms'%samplesize).mean()
    df = df.interpolate(method='linear',limit_direction='both')
    df = df.rolling(rolling, min_periods=1).mean()

    # print("length OD=%s"%len(df['obstacle_density21'].dropna()))
    # print("length N=%s"%len(df['narrowness1'].dropna()))
    # print("length S=%s"%len(df['safety'].dropna()))
    # print(df['safety'].dropna().head(20))

    if print_head:
        print(df.head())
    return df, start, end


def generate_dataset_all(configs,xtopics,ytopic,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling):
    os.chdir(dir_bags)
    files = dict()
    df = dict()
    X = dict()
    y = dict()
    groups = dict()
    n_exp = dict()
    for config in configs:
        df[config] = dict()

        files[config] = sorted(glob.glob("*%s_c%s*.bag"%(exp,config)))
        print(files[config])
        for idx in range(len(files[config])):
            df[config][idx] = import_bag(files[config][idx],samplesize,rolling)
            df[config][idx] = add_derivs(df[config][idx],d_topics)

            # Start and end time:
            df[config][idx].drop(df[config][idx].head(int(start_ms/samplesize)).index,inplace=True)
            df[config][idx].drop(df[config][idx].tail(int(end_ms/samplesize)).index,inplace=True) # drop last n rows
        # n = len(files[config])
        # print(df[config][0].head())
        # All the data in one set
        for idx in range(len(files[config])):
            n_group = files[config][idx][-5]    # Group number is the run number
            # print(n_group)
            if idx == 0:
                X[config] = df[config][idx][xtopics].values
                y[config] = df[config][idx][ytopic].values
                groups[config] = np.full(len(X[config]),n_group)
            else:
                X[config] = np.concatenate((X[config], df[config][idx][xtopics].values))
                y[config] = np.concatenate((y[config], df[config][idx][ytopic].values))
                groups[config] = np.concatenate((groups[config], np.full(len(df[config][idx][ytopic].values),n_group)))
    return X, y, groups

def generate_dataset_all_selectedfiles(files,configs,xtopics,ytopic,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling):
    os.chdir(dir_bags)
    df = dict()
    X = dict()
    y = dict()
    groups = dict()
    n_exp = dict()
    for config in configs:
        df[config] = dict()
        for idx in range(len(files[config])):
            df[config][idx] = import_bag(files[config][idx],samplesize,rolling)
            df[config][idx] = add_derivs(df[config][idx],d_topics)

            # Start and end time:
            df[config][idx].drop(df[config][idx].head(int(start_ms/samplesize)).index,inplace=True)
            df[config][idx].drop(df[config][idx].tail(int(end_ms/samplesize)).index,inplace=True) # drop last n rows
        # n = len(files[config])
        # check_shittyness(df,xtopics,ytopic,configs,n,samplesize)

        # All the data in one set
        for idx in range(len(files[config])):
            n_group = files[config][idx][-5]    # Group number is the run number
            # print(n_group)
            if idx == 0:
                X[config] = df[config][idx][xtopics].values
                y[config] = df[config][idx][ytopic].values
                groups[config] = np.full(len(X[config]),n_group)
            else:
                X[config] = np.concatenate((X[config], df[config][idx][xtopics].values))
                y[config] = np.concatenate((y[config], df[config][idx][ytopic].values))
                groups[config] = np.concatenate((groups[config], np.full(len(df[config][idx][ytopic].values),n_group)))
    return X, y, groups

def generate_dataset_shifted(configs,xtopics,ytopic,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling):
    os.chdir(dir_bags)
    files = dict()
    df = dict()
    X_shift = dict()
    y_shift = dict()
    groups_shift = dict()
    lags = dict()
    mean_lags = dict()
    n_exp = dict()
    for config in configs:
        df[config] = dict()
        files[config] = sorted(glob.glob("%s_c%s*.bag"%(exp,config)))
        lags[config] = []
        for idx in range(len(files[config])):
            # Import bags
            df[config][idx] = import_bag(files[config][idx],samplesize,rolling)
            df[config][idx] = add_derivs(df[config][idx],d_topics)
            # Start and end time:
            df[config][idx].drop(df[config][idx].head(int(start_ms/samplesize)).index,inplace=True)
            df[config][idx].drop(df[config][idx].tail(int(end_ms/samplesize)).index,inplace=True) # drop last n rows
            # Lags
            lags[config].append(determine_lags(df[config][idx],xtopics,ytopic,samplesize))
        # Determine mean lags for shift
        mean_lags[config] = np.array(lags[config]).mean(axis=0).astype(int)

        # All the data in one set
        for idx in range(len(files[config])):
            df_shift = shift_lags(df[config][idx],xtopics,mean_lags[config])
            n_group = files[config][idx][-5]    # Group number is the run number
            if idx == 0:
                # Shifted
                X_shift[config] = df_shift[xtopics].values
                y_shift[config] = df_shift[ytopic].values
                groups_shift[config] = np.full(len(df_shift[ytopic].values),n_group)
            else:
                X_shift[config] = np.concatenate((X_shift[config], df_shift[xtopics].values))
                y_shift[config] = np.concatenate((y_shift[config], df_shift[ytopic].values))
                groups_shift[config] = np.concatenate((groups_shift[config], np.full(len(df_shift[ytopic].values),n_group)))
    return X_shift, y_shift, groups_shift


def generate_dataset(configs,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling):
    os.chdir(dir_bags)
    files = dict()
    df = dict()
    X = dict()
    y = dict()
    groups = dict()
    n_exp = dict()
    for config in configs:
        df[config] = dict()

        files[config] = sorted(glob.glob("*%s_c%s*.bag"%(exp,config)))

        for idx in range(len(files[config])):
            df[config][idx],start,end = import_bag(files[config][idx],samplesize,rolling)
            print(end)
            print(df[config][idx].index[-1].total_seconds())
            end = df[config][idx].index[-1].total_seconds() - end
            print(end)
            df[config][idx] = add_derivs(df[config][idx],d_topics)

            # Start and end time:
            df[config][idx].drop(df[config][idx].head(int((start*1000)/samplesize)).index,inplace=True)
            df[config][idx].drop(df[config][idx].tail(int((end*1000)/samplesize)).index,inplace=True) # drop last n rows
    return files, df

def add_derivs(df,topics):
    for topic in topics:
        df['d_%s'%topic] = np.gradient(df[topic].values)/(df.index[1].total_seconds())
    return df


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

    return fig, axes
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

def graph_xcorr(df1,df2,samplesize,title='Cross correlation'):
    fig = plt.figure()#figsize=(10, 10))
    ax = fig.add_subplot(2,1,1)
    ax.plot(df1.index.total_seconds(),df1.values,label=df1.name)
    ax.plot(df2.index.total_seconds(),df2.values,label=df2.name)
    ax.legend()
    # ax[0].set_title(title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signals')
    ax.set_title(title)

    ms = 1500
    n = ms/samplesize
    lags = range(-int(n),int(n),1)
    lags_ms = np.array(range(-int(n),int(n),1))*samplesize
    rs = [crosscorr(df1,df2, lag) for lag in lags]
    # rs = [crosscorr(df1[topic_shift],df1[topic_ref], lag) for lag in lags]
        
    ax = fig.add_subplot(2,1,2)
    ax.plot(lags_ms,rs)
    ax.set_title(title)
    ax.set_xlabel('lag [ms]')
    ax.set_ylabel('Correlation r')
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85,bottom=0.12)

    # print(max(rs, key=abs),lags[rs.index(max(rs, key=abs))])

    return fig
    # fig.savefig(resultsfolder + title + '.png')

def crosscorr(datax, datay, lag=0, wrap=False):
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))


def determine_lags(df1,topics_shift,topic_ref,samplesize):
    ms = 1500
    n = ms/samplesize
    if topic_ref == 'performance2_3':
        lags = range(0,int(n),1)
    else:
        lags = range(-int(n),int(n),1)

    lag_n = []
    corrs = []
    for topic_shift in topics_shift:
        rs = [crosscorr(df1[topic_ref],df1[topic_shift], lag) for lag in lags]
        if topic_shift == 'd_narrowness1':
            lag_n.append(lags[rs.index(max(rs))])
            corrs.append(max(rs))
        else:
            lag_n.append(lags[rs.index(max(rs, key=abs))])
            corrs.append(max(rs, key=abs))
        # print('For topic_shift: %s, Max rs = %s, at lag = %s'%(topic_shift,max(rs, key=abs),samplesize*lag_n[-1]))
    
    return corrs, lag_n

def corrs_lags(df1,topics_shift,topic_ref,samplesize):
    ms = 1500
    n = ms/samplesize
    lags = range(-int(n),int(n),1)

    lag_n = []
    rs_max = []
    for topic_shift in topics_shift:
        rs = [crosscorr(df1[topic_ref],df1[topic_shift], lag) for lag in lags]
        if topic_shift == 'd_narrowness1':
            rs_max.append(round(max(rs),2))
            lag_n.append(lags[rs.index(max(rs))])
        else:
            rs_max.append(round(max(rs, key=abs),4))
            lag_n.append(lags[rs.index(max(rs, key=abs))])
        # print('For topic_shift: %s, Max rs = %s, at lag = %s'%(topic_shift,max(rs, key=abs),samplesize*lag_n[-1]))
    
    return rs_max, lag_n

def shift_lags(df,topics_shift,lag_n):
    # Make a copy so the original will not be modified
    df1 = df.copy()

    for idx in range(len(lag_n)):
        df1[topics_shift[idx]] = df1[topics_shift[idx]].shift(lag_n[idx])
    df1 = df1.dropna()
    return df1