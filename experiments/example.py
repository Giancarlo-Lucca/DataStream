#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 09:27:46 2023

@author: asier
"""
import os, sys
from pathlib import Path
sys.path.append(os.path.abspath("."))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.d_fuzzstream import DFuzzStreamSummarizer
from src.functions.merge import AllMergers
from src.functions.distance import EuclideanDistance
from src.functions.membership import FuzzyCMeansMembership
from src.functions import metrics


#sm = 20
sm = 1
min_fmics = 5
max_fmics = 100
thresh = 0.8
chunksize = 1000
color = {'1': 'Red', '2': 'Blue', '3': 'Green', '4': 'pink', 'nan': 'Gray'}
figure = plt.figure()
scatter = plt.scatter('x', 'y', s='radius', data={'x': [], 'y': [], 'radius': []})

datasetName = 'Benchmark1_11000' # Benchmark1_11000, RBF1_40000, 'DS1' #

if (datasetName == 'Benchmark1_11000'):
    datasetPath = "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/Bench1_11k/Benchmark1_11000.csv"
    numChunks = 11
elif (datasetName == 'RBF1_40000'):
    datasetPath = "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/RBF1_40k/RBF1_40000.csv"
    numChunks = 40
elif (datasetName == 'DS1'):
    datasetPath = "SamplesFile_b_4C2D800Linear.csv"
    numChunks = 40

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
currentPath = Path.cwd()
output_path = currentPath / "output"/ datasetName
Path(output_path).mkdir(parents=True,exist_ok=True)

df = pd.DataFrame(columns = ['Chunk', 'Purity', 'pCoefficient', 'pEntropy', 'XieBeni','MPC','FukuyamaSugeno_1','FukuyamaSugeno_2'])
summarizer = DFuzzStreamSummarizer(
    distance_function=EuclideanDistance.distance,
    merge_threshold = thresh,
    merge_function=AllMergers[sm](sm, thresh, max_fmics),
    membership_function=FuzzyCMeansMembership.memberships,
    chunksize = chunksize,
    n_macro_clusters=2,
    time_gap=10000,
)

summary = {'x': [], 'y': [], 'radius' : [], 'color': [], 'weight': [], 'class': []}
timestamp = 0

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
currentPath = Path.cwd()
output_path = currentPath / "output"/ datasetName
Path(output_path).mkdir(parents=True,exist_ok=True)

fhand = open(output_path / 'chunkFMICs.txt', 'a')

# Read files in chunks
with pd.read_csv(datasetPath,
# with pd.read_csv("https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/Bench1_11k/Benchmark1_11000.csv",
# with pd.read_csv("https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/RBF1_40k/RBF1_40000.csv",
                dtype={"X1": float, "X2": float, "class": str},
                chunksize=chunksize) as reader:
    for chunk in reader:
        print(f"Summarizing examples from {timestamp} to {timestamp + 999} -> sim {sm} and thrsh {thresh}")
        fhand.write(f"Summarizing examples from {timestamp} to {timestamp + 999} -> sim {sm} and thrsh {thresh}\n")
        for index, example in chunk.iterrows():
            # Summarizing example
            summarizer.summarize(example[0:2], example[2], timestamp)
            timestamp += 1
        summarizer.offline()


        # TODO: Obtain al metrics and create the row
        all_metrics = metrics.all_online_metrics(summarizer.summary(), chunksize)
        metrics_summary = ""
        for name, value in all_metrics.items():
            metrics_summary += f"{name}: {round(value,3)}\n"
        metrics_summary = metrics_summary[:-1]

        row_metrics = list(all_metrics.values())
        row_timestamp = ["["+str(timestamp)+" to "+str(timestamp + 999)+"]"]

        new_row = pd.DataFrame([row_timestamp + row_metrics],
                               columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)

        fhand.write("Total de Fmics = "+str(len(summarizer.summary())))
        
        for fmic in summarizer.summary():
            for k, v in fmic.sumPointsPerClassd.items(): 
                fhand.write(f"\nTotal pontos classe {k} = {v} \n")
            fhand.write("------------------")

            summary['x'].append(fmic.center[0])
            summary['y'].append(fmic.center[1])
            summary['radius'].append(fmic.radius * 100000)
            summary['color'].append(color[max(fmic.tags, key=fmic.tags.get)])
            summary['weight'].append(fmic.m)
            summary['class'].append(max(fmic.tags, key=fmic.tags.get))

        if not os.path.isdir("./Img/"):
            os.mkdir("./Img/")

        fig = plt.figure()
        plt.scatter('x', 'y', s='radius', color='color',
                    data=summary, alpha=0.1)
        # Plot centroids
        plt.scatter('x', 'y', s=1, color='color', data=summary)
        # plt.legend(["color blue", "color green"], loc ="lower right")
        # plt.legend(["Purity"+str(summarizer.Purity()),"PartitionCoefficient"+str(summarizer.PartitionCoefficient()),"PartitionEntropy"+str(summarizer.PartitionEntropy()),"XieBeni"+str(summarizer.XieBeni()), "FukuyamaSugeno_1"+str(summarizer.FukuyamaSugeno_1()),"FukuyamaSugeno_2"+str(summarizer.FukuyamaSugeno_2())], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.figtext(.8, .8, "T = 4K")

        side_text = plt.figtext(.91, .8, metrics_summary)
        fig.subplots_adjust(top=1.0)
        # plt.show()
        fig.savefig("./Img/KK[Chunk "+str(timestamp - 1000)+" to "+str(timestamp - 1)+"] Sim("+str(sm)+")_Thresh("+str(thresh)+").png", bbox_extra_artists=(side_text,), bbox_inches='tight')
        plt.close()


    # Transforming FMiCs into dataframe
    for fmic in summarizer.summary():
        summary['x'].append(fmic.center[0])
        summary['y'].append(fmic.center[1])
        summary['radius'].append(fmic.radius * 100000)
        summary['color'].append(color[max(fmic.tags, key=fmic.tags.get)])
        summary['weight'].append(fmic.m)
        summary['class'].append(max(fmic.tags, key=fmic.tags.get))

    print("==== Approach ====")
    print("Similarity = ", sm)
    print("Threshold = ", thresh)
    print("==== Summary ====")
    print(summary)
    print("==== Metrics ====")
    print(summarizer.metrics)
    print("\n")
    print(df)
    print("------")

    output = "\n==== Approach ===="
    output = output + str("\n Similarity ="+str(sm))
    output = output + str("\n Threshold ="+str(thresh))
    output = output + str("\n ==== Summary ====")
    output = output + str("\n "+str(summary))
    output = output + str("\n ==== Metrics ====")
    output = output + str("\n "+str(summarizer.metrics))
    output = output + str("\n ")
    output = output + str("\n ==== Evaluation ====")
    # output = output + str("\n "+df.to_markdown(tablefmt='greed'))
    output = output + str("\n *****")
    df = df[0:0]
    # df.drop(df.index, inplace=True)
    with open('directOUTPUT.txt', 'a') as f:
        f.write(output)

    print("Final clusters:")
    if timestamp % 10000 == 0:
        final_clusters, mu = summarizer.final_clustering()
        for fc in final_clusters:
            print(fc)



with open(output_path / 'directOUTPUT.txt', 'a') as f:
    f.write("\n------------------------------------------------------------")
print("--- End of execution --- ")
f.close()
fhand.close()


# Visualization
import matplotlib.pyplot as plt

summary = {'x': [], 'y': [], 'radius' : [], 'color': [], 'weight': [], 'class': []}
for fmic in summarizer.summary():
    summary['x'].append(fmic.center[0])
    summary['y'].append(fmic.center[1])
    summary['radius'].append(fmic.radius * 100000)
    summary['color'].append(color[max(fmic.tags, key=fmic.tags.get)])
    summary['weight'].append(fmic.m)
    summary['class'].append(max(fmic.tags, key=fmic.tags.get))


final_clusters, mu = summarizer.final_clustering()
fig0, ax0 = plt.subplots()
for label in range(len(summary)):
    ax0.plot(summary['x'], summary['y'], '.',
             color=summary['color'][label])
for vi in final_clusters:
    ax0.plot(vi[0], vi[1], 'rs',  color='red')
ax0.set_title('Clustering')

