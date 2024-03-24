import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.d_fuzzstream import DFuzzStreamSummarizer
from src.functions.merge import AllMergers
from src.functions.distance import EuclideanDistance
from src.functions.membership import FuzzyCMeansMembership
from src.functions import metrics

sm = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
min_fmics = 5

start, end = 0, 31
datasetName = 'RBF1_40000'  # 'RBF1_40000',  'Benchmark1_11000'

if (datasetName == 'Benchmark1_11000'):
    datasetPath = "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/Bench1_11k/Benchmark1_11000.csv"
    threshList = [0.8, 0.9, 0.25, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.8, 0.25, 0.25, 0.65, 0.65, 0.8, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
    numChunks = 11
    chunksize = 1000
    n_clusters = 2
    max_fmics = 50
elif (datasetName == 'RBF1_40000'):
    datasetPath = "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/RBF1_40k/RBF1_40000.csv"
    threshList = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
    numChunks = 40
    chunksize = 1000
    n_clusters = 3
    max_fmics = 100
elif (datasetName == 'Gaussian_4C2D800'):
    datasetPath = "../datasets/DS1.csv" # https://gitlab.citius.usc.es/david.gonzalez.marquez/GaussianMotionData/-/raw/master/SamplesFile_b_4C2D800Linear.csv?ref_type=heads
    threshList = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
    numChunks = 8
    chunksize = 100
    n_clusters = 4
    max_fmics = 100
output_path = "".join(("./output/", datasetName, "/"))

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

for vecIndex, simIDX in enumerate(sm[start:end]):
    threshIDX = threshList[vecIndex]
    summarizer = DFuzzStreamSummarizer(
        max_fmics=max_fmics,
        distance_function=EuclideanDistance.distance,
        merge_threshold=threshIDX,
        merge_function=AllMergers[simIDX](simIDX, threshIDX, max_fmics),
        membership_function=FuzzyCMeansMembership.memberships,
        chunksize=chunksize,
        n_macro_clusters=n_clusters,
        time_gap=chunksize
    )

    timestamp = 0

        # Read files in chunks
    with pd.read_csv(datasetPath,
                     dtype={"X1": float, "X2": float, "class": str},
                     chunksize=chunksize) as reader:
        for chunk in reader:
            log_text = (f"Summarizing examples from {timestamp} to "
                        f"{timestamp + chunksize-1} -> sim {simIDX} "
                        f"and thrsh {threshIDX}")

            for index, example in chunk.iterrows():
                # Summarizing example
                ex_data = example[0:2]
                ex_class = example[2]
                summarizer.summarize(ex_data, ex_class, timestamp)
                timestamp += 1

                # Offline - Evaluation
                if (timestamp) % summarizer.time_gap == 0:
                    ari, sil = metrics.offline_stats(summarizer, chunk)
                    purity = metrics.offline_purity(summarizer._Vmm)
                    print(simIDX,timestamp,ari,sil, purity)


print("--- End of execution --- ")

