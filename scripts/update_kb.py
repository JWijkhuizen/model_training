#!/usr/bin/env python3.6

import rospkg
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

import pickle
from owlready2 import *
from threading import Lock

from functions_postprocess import *

# Paths
rospack = rospkg.RosPack()
path = rospack.get_path('model_training')
path2 = rospack.get_path('simulation_tests')
dir_bags = path2 + '/bags/'
dir_figs = path + '/figures/'
dir_models = path + '/models/'
dir_results = path + '/results/'
path3 = rospack.get_path('mros1_reasoner')
path_kb = path3 + '/scripts/'
path_tomasys = rospack.get_path('mc_mdl_tomasys') + '/'
# Experiment name and configs
exp = 't1'
# configs = ['dwa_v1_a0_b0','teb_v1_a0_b0']
configs = ["dwa_v1_a0_b0", "dwa_v1_a1_b0", "dwa_v1_a0_b1", "dwa_v1_a1_b1", "dwa_v2_a0_b0", "dwa_v2_a1_b0", "dwa_v2_a0_b1", "dwa_v2_a1_b1", "teb_v1_a0_b0", "teb_v1_a1_b0", "teb_v1_a0_b1", "teb_v1_a1_b1", "teb_v2_a0_b0", "teb_v2_a1_b0", "teb_v2_a0_b1", "teb_v2_a1_b1"]

save = False

# Topics
d_topics = []
xtopics = ['obstacle_density21','narrowness1']
ytopics = ['safety','performance_old31']
qa_name = dict()
qa_name['safety'] = 'safety'
qa_name['performance_old31'] = 'performance'


# print_output = True
print_output = False
# plot_model = True
plot_model = True
plot = True
save = True
# Models
# polies = [1,2,3,4,5,6]
polies = [5]

# Resamplesize and smoothing (rolling)
samplesize = 100
rolling = 1

tomasys_file = path_tomasys + 'tomasys.owl'
onto_file = path_kb + 'kb_nav_quality_v2.owl'
save_onto_file = onto_file

# Load kb
tomasys = get_ontology(tomasys_file).load()
onto = get_ontology(onto_file).load()
lock = Lock()

if tomasys is not None:
    print("Succesfully loaded tomasys")
else:
    print("Failed to load tomasys from %s"%tomasys_file)
if onto is not None:
    print("Succesfully loaded ontology")
else:
    print("Failed to load ontology from %s"%onto_file)


wb = Workbook()
ws = wb.active
ws.append([""] + configs)


# Determine averages and update kb
colors = ['tab:blue','tab:orange']
ytopic_av_array = dict()
for ytopic in ytopics:
    ytopic_av_array[ytopic] = []
    # Import Bag files into pandas
    X, y, groups = generate_dataset_all(configs,xtopics,ytopic,d_topics,exp,dir_bags,samplesize,rolling)

    print('*******************************')
    print(ytopic)
    for config in configs:
        print('************')
        print(config)
        ytopic_av = sum(y[config])/len(y[config])
        ytopic_av_array[ytopic].append(round(ytopic_av,3))
        print("average = %s"%(ytopic_av))

        # Update kb
        fd = next((fd for fd in tomasys.FunctionDesign.instances() if fd.name == config), None)
        qa_type = onto.search_one(iri="*{}".format(qa_name[ytopic]))
        if qa_type != None:
            with lock:
                qas = fd.hasQAestimation
                if qas == []: # for the first qa value received
                    print("No QAestimation found")
                else:
                    for qa in qas:
                        if qa.isQAtype == qa_type:
                            qa.hasValue = float(ytopic_av)

ws.append([""] + ytopics)
idy = 0
for config in configs:
    ws.append([config, ytopic_av_array[ytopics[0]][idy], ytopic_av_array[ytopics[1]][idy]])
    idy+=1


def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 1:
            value = round(height,1)
        elif height < 1:
            value = round(height,2)
        ax.annotate('{}'.format(value),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

if plot:
    fig, ax = plt.subplots(figsize=[12.8,4.8])
    x = np.arange(len(configs))  # the label locations
    width = 0.4  # the width of the bars
    idb = 0
    for ytopic in ytopics:
        rects = ax.bar(x-0.5*width+idb*width, ytopic_av_array[ytopic] , width, label=qa_name[ytopic])
        autolabel(rects,ax)
        idb+=1

    ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    ax.set_ylabel('Average quality level')
    ax.set_xlabel("Configuration")
    # ax.set_title('Time to completion of the SA systems and benchmarks')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    plt.xticks(rotation=45, ha='right')
    ax.set_ylim([0,1])
    ax.yaxis.grid()
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # plt.subplots_adjust(top=0.85)
    plt.subplots_adjust(bottom=0.25)
    title = 'Average_quality_configs'
    fig.savefig(dir_figs + title + '.png')


if save:
    os.chdir(dir_results)
    wb.save(filename = 'QA_average_table.xlsx')
    onto.save(file=save_onto_file, format="rdfxml")

print("Knowledge base succesfully updated!")
print("Kb saved to %s"%save_onto_file)


plt.show()