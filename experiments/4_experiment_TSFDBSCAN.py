#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:56:54 2023

@author: asier
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scluster.TSF_DBSCAN import TSF_DBSCAN, p_object


def plot(X, C, M, outliers=[], ranges=None, title="title", file_name=""):
    if any(outliers):
        Xo = X[outliers]

        notoutliers = np.logical_not(outliers)
        Xc = X[notoutliers]
        Mc = M[notoutliers]
        Cc = C[notoutliers]
    else:
        Xc = X
        Mc = M
        Cc = C

    uC = [c for c in np.unique(Cc)]

    for i in range(len(np.unique(uC))):
        Cc[np.where(Cc == uC[i])] = i

    if ranges is not None:
        x1_min, x2_min = ranges[0]
        x1_max, x2_max = ranges[1]
    else:
        x1_min, x2_min = [round(x) for x in np.min(X[:, :-1], axis=0)]
        x1_max, x2_max = [round(x) for x in np.max(X[:, :-1], axis=0)]

    cmap = plt.get_cmap("tab20b", int(np.max(C)) - int(np.min(C)) + 1)
    plt.figure()
    if any(outliers):
        plt.scatter(Xo[:, 0], Xo[:, 1], marker='x')
    if len(Xc > 0):
        plt.scatter(Xc[:, 0], Xc[:, 1], c=Cc, cmap=cmap, alpha=Mc)  # alpha=M  # for fuzzy borders
    # plt.xlim(x1_min - 1, x1_max + 1)
    # plt.ylim(x2_min - 1, x2_max + 1)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.colorbar()
    plt.title(title)
    if file_name == "":
        file_name = title+".png"
    plt.savefig(file_name)
    

def main():
    currentPath = Path.cwd()
    dataset = "Benchmark1_11000.csv" # "RBF1_40000.csv"
    datasetPath = currentPath / "datasets" / dataset
    outputPath = currentPath / "output" / dataset.split(".")[0]
    dtypes = {"X1": float, "X2": float, "class": str}
    chunksize = 1000 # 800 banana, 100 DS1, 1000 covertype
    R = 11  # 6 banana, 8 DS1,
    emin_set = [0.01,0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    emax_set = [1, 2, 3, 4, 5]
    emin = 0.1
    emax = 0.3
    alpha = 0.0015
    omega = 0.3
    wmin = 1
    for emin in emin_set:
        for emax_prod in emax_set:
            for wmin in emax_set:            
                run_tsf_dbscan(datasetPath, outputPath, dtypes, chunksize, emin, emax_prod*emin, alpha, omega, wmin)

def run_tsf_dbscan(dataset, outputPath, dtypes, chunksize, emin, emax, alpha, omega, wmin):
    tsf = TSF_DBSCAN(emin, emax, alpha, omega, wmin, chunksize) 
    name = f"TSF DBSCAN {dataset.name}-{emin}-{emax}-{alpha}-{omega}-{wmin}"
    
    # Read files in chunks
    with pd.read_csv(dataset,
                     # names=['X1','X2','class'],  # For Gaussian dataset only
                     dtype=dtypes,
                     chunksize=chunksize) as reader:
        timestamp = 0
        ARI = []
        SIL = []
        for chunk in reader:
            print(f"Summarizing examples from {timestamp} to {timestamp + chunksize - 1}")
            for index, example in chunk.iterrows():
                # Summarizing example
                point = p_object(example[0:2].tolist(), t=timestamp)
                tsf.tsfdbscan(point)
                timestamp += 1
                
            print(f"{len(tsf.clusters)} Clusters found. ({index})")

            results = np.array([x.x + list(x.get_max_cluster_membership()) for x in tsf.plist])
            X = results[:, :-2]
            labels_tsf = results[:, -2]
            M = results[:, -1]

            label_real = chunk.iloc[:,-1]
            label_real[label_real.isnull()] = -1
            ARI.append(adjusted_rand_score(label_real.iloc[-chunksize:].astype('int'), labels_tsf[-chunksize:]))
            if len(X) - 1 >= len(np.unique(labels_tsf)) > 1:
                SIL.append(silhouette_score(X, labels_tsf))
            else:
                SIL.append(np.nan)
            print(ARI[-1], SIL[-1])
            plot(X, labels_tsf, M, labels_tsf==-1,ranges=[[0,0],[1,1]],
                 title=f"{name}-\n-{timestamp}",file_name=f"{outputPath}/{name}-{timestamp}.png")
            with open(f"{outputPath}/results11k.csv",mode='a') as res_file:
                res_file.write(f"{dataset.name},{emin},{emax},{alpha},{omega},{wmin},{timestamp}, {ARI[-1]},{SIL[-1]}")
            
        print(np.mean(ARI))
        print(np.mean(SIL))
if __name__ == "__main__":
    main()
