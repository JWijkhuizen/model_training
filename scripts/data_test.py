#!/usr/bin/env python3.6

# import rosbag
import rospy
import rospkg
import os
import sys
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from matplotlib.widgets import Button

from functions_postprocess import *


# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
path2 = rospack.get_path('simulation_tests')
dir_bags = path2 + '/bags/'
dir_figs = path + '/figures/'
dir_models = path + '/models/'
dir_results = path + '/results/'

# Experiment name and configs
exp = '14'
configs = ['dwa_v0_a0_b0','teb_v0_a0_b0']
# configs = ['cdwa1']

# Topics
d_topics = []
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics]
# xtopics = d_topics + ['d_%s'%d_topic for d_topic in d_topics] + ['performance2']
# xtopics = ['obstacle_density11','narrowness','d_obstacle_density','performance']
xtopics = ['obstacle_density21','narrowness1']
# ytopic = 'performance3'
ytopics = ['safety']

# Resamplesize and smoothing (rolling)
samplesize = 100
rolling = 1

# Experiment start and end
start_ms = 1
end_ms = 1

# Import Bag files into pandas
files, df = generate_dataset(configs,d_topics,exp,dir_bags,start_ms,end_ms,samplesize,rolling)
# os.chdir(dir_bags)
# files = dict()
# for config in configs:
#     files[config] = sorted(glob.glob("%s_%s*.bag"%(exp,config)))

print('idx   File')
for config in configs:
    for idx in range(len(files[config])):
        print('%-5s %-s'%(idx, files[config][idx]))

# Functions for buttons
def fnext(event):
    global idx
    idx += 1
    plt.close()

def fprev(event):
    global idx
    idx -= 1
    plt.close()

def fsave(event):
	axprev.set_visible(False)
	axnext.set_visible(False)
	axsave.set_visible(False)
	axexit.set_visible(False)

	fig.savefig(dir_figs + 'data_plot_exp%s_config%s_idx%s'%(exp,config,idx) + '.png')
	plt.close()

def fexit(event):
    global idx
    idx = 100
    plt.close()




files_incl = dict()
# topics = [['safety1','performance4'],[]]
print("Plotting")
for config in configs:
    files_incl[config] = []
    idx = 10
    while idx in range(len(files[config])):   
        dfs1 = [df[config][idx][ytopic] for ytopic in ytopics]
        dfs2 = [df[config][idx][xtopic] for xtopic in xtopics]
        titles = ['Experiment:%s, idx:%s, config:%s \n Quality attributes'%(exp,idx,config),'Environment metrics and robot states']
        xlabel = 'Time [s]'
        ylabels = ['Quality \n level','Environment \n measure']
        fig, ax = graph21([dfs1,dfs2], titles, xlabel, ylabels)
        plt.subplots_adjust(bottom=0.2)
        axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.3, 0.05, 0.1, 0.075])
        axsave = plt.axes([0.6, 0.05, 0.1, 0.075])
        axexit = plt.axes([0.8, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bprev = Button(axprev, 'Prev')
        bsave = Button(axsave, 'Save')
        bexit = Button(axexit, 'Exit')
        bnext.on_clicked(fnext)
        bprev.on_clicked(fprev)
        bsave.on_clicked(fsave)
        bexit.on_clicked(fexit)
        plt.show()

