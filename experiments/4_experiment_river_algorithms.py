#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:59:50 2024

@author: asier.urio
"""

import argparse
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath("."))
import math
import statistics
from river import cluster
from river import stream
import pandas as pd
from sklearn.metrics import silhouette_score,adjusted_rand_score


def clu_stream_gen(n_classes, chunksize):
    maxmcs = [50, 100, 200]
    halflifes = [0.1, 0.2, 0.4, 0.5, 0.8, 1]
    for mmc in maxmcs:
        for hl in halflifes:
            clustream = cluster.CluStream(
                n_macro_clusters=n_classes,
                max_micro_clusters=mmc,
                time_gap=chunksize,
                time_window=chunksize,
                halflife=hl
            )
            yield clustream, [mmc, hl]



def db_stream_gen(n_classes=None, chunksize=None):
    cleanups = [1, 5 , 10]
    thresholds = [0.1, 0.2, 0.4, 0.5, 0.8, 1]
    factors = [0.01, 0.05, 0.1, 0.3, 0.5, 1]
    mweights = [0.1, 0.5, 1, 1.5]
    for cth in thresholds:
        for ffa in factors:
            for ifa in thresholds:
                for ci in cleanups:
                    for mw in mweights:
                        dbstream = cluster.DBSTREAM(
                            clustering_threshold=cth,
                            fading_factor=ffa,
                            cleanup_interval=ci,
                            intersection_factor=ifa,
                            minimum_weight=mw
                        )
                        yield dbstream, [cth, ffa, ci, ifa, mw]

def den_stream_gen(n_classes=None, chunksize=None):
    samples = [10, 50 , 100]
    thresholds = [0.1, 0.2, 0.4, 0.5, 0.8, 1]
    factors = [0.01, 0.05, 0.1, 0.3, 0.5, 1]
    mus = [1, 2.5, 5, 10]
    n_comb = len(factors) * len(thresholds) * len(mus) * len(thresholds) * len(samples)
    curr = 0
    for ffa in factors:
        for b in thresholds:
            for m in mus:
                for ep in thresholds:
                    for samp in samples:
                        print(f"{curr}/{n_comb}")
                        curr += 1
                        if m*b <= 1:
                            continue
                        denstream = cluster.DenStream(
                            decaying_factor=ffa,
                            beta=b,
                            mu=m,
                            epsilon=ep,
                            n_samples_init=samp
                        )
                        yield denstream, [ffa, b, m, ep, samp]


def skm_stream_gen(n_classes=2, chunksize=1000):
    halflifes = [0.1, 0.2, 0.4, 0.5, 0.8, 1]
    sigmas  = [0.5, 1, 1.5, 2, 5]
    for hl in halflifes:
        for sg in sigmas:
            streamkmeans = cluster.STREAMKMeans(
                chunk_size=chunksize,
                n_clusters=n_classes,
                halflife=hl,
                sigma=sg,
                seed=0
            )
            yield streamkmeans, [hl, sg]

def ikm_stream_gen(n_classes=2, chunksize=1000):
    halflifes = [0.1, 0.2, 0.4, 0.5, 0.8, 1]
    sigmas  = [0.5, 1, 1.5, 2, 5]
    for hl in halflifes:
        for sg in sigmas:
            ik_means = cluster.KMeans(
                n_clusters=n_classes,
                halflife=hl,
                sigma=sg,
                seed=0
            )
            yield ik_means, [hl, sg]

def run(algorithm, datasetPath, chunksize):
    ARI = []
    classes = []
    points = []
    timestamp = 0
    with pd.read_csv(datasetPath,
                     dtype={ "class": str},
                     chunksize=chunksize) as reader:
        timestamp = 1
        for chunk in reader:
            for index, example in chunk.iterrows():
                X = example[example.index[0:-1]].to_dict()
                C = example[example.index[[-1]]].to_dict()
                classes.append(C)
                points.append(X)
                algorithm.learn_one(X)

                if timestamp % chunksize == 0:
                    YC = []
                    Y = []
                    for XTest, YTest in zip(points, classes):
                        resultC = algorithm.predict_one(XTest)
                        YC.append(resultC)
                        Y.append(-1 if math.isnan(float(YTest.get("class"))) else YTest.get("class"))

                    ARI.append(adjusted_rand_score(Y, YC))

                    classes = []
                    points = []
                timestamp += 1
    ari_i = statistics.mean(ARI)
    return ari_i


def main(algorithm, dataset):
    datasetPath = f"datasets/{dataset}.csv"

    if dataset == "Benchmark1_11000":
        n_classes = 2
        chunksize = 1000
        numChunks = 11
    elif dataset == "RBF1_40000":
        n_classes = 3
        chunksize = 1000
        numChunks = 40
    elif dataset == "DS1":
        n_classes = 4
        chunksize = 100
        numChunks = 8
    elif dataset == "powersupply":
        n_classes = 24
        chunksize = 1000
        numChunks = 29
    elif dataset == "NOAA":
        n_classes = 2
        chunksize = 1000
        numChunks = 18

    if algorithm == "clustream":
        alg = clu_stream_gen
    elif algorithm == "dbstream":
        alg = db_stream_gen
    elif algorithm == "denstream":
        alg = den_stream_gen
    elif algorithm == "skmeans":
        alg = skm_stream_gen
    elif algorithm == "ikmeans":
        alg = ikm_stream_gen

    best = 0
    best_params =  []
    for alg, params in alg(n_classes, chunksize):
        ari_i = run(alg,datasetPath, chunksize)

        if ari_i > best:
            best = ari_i
            best_params = params

    print(f"# Best result for {algorithm} in {dataset} dataset: ARI = {best:.3f}. For {best_params=}")
    print(f"# {algorithm},{dataset},{best:.4f},{best_params=}")

    with open("../output/results_river_tune.csv", "a") as rfile:
        rfile.write(f"\n# {algorithm},{dataset},{best:.4f},{best_params=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter parse of this project")
    parser.add_argument('--dataset', type=str, default='Benchmark1_11000',
                        help='Dataset: Benchmark1_11000 (d) or RBF1_40000')

    parser.add_argument('--algorithm', type=str, default="clustream",
                        help='algorithm to run')

    args = parser.parse_args()

    dataset = args.dataset
    algorithm = args.algorithm

    main(algorithm, dataset)
