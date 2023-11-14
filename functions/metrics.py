#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:02:34 2023

@author: asier
"""

import numpy as np


def all_online_metrics(fmics, chunksize):
    return {
        "Purity": Purity(fmics),
        "PartitionCoefficent": PartitionCoefficient(fmics, chunksize),
        "PartitionEntropy": PartitionEntropy(fmics, chunksize),
        "XieBeni": XieBeni(fmics, chunksize),
        "ModifiedPartitionCoefficent": ModifiedPartitionCoefficient(fmics,
                                                                    chunksize),
        "FukuyamaSugeno_1": FukuyamaSugeno_1(fmics),
        "FukuyamaSugeno_2": FukuyamaSugeno_2(fmics, chunksize)
        }


def ARI(data):
    '''
    data:
    [[a11, .., a1n, class1, cluster1]
    ...
    [am1, .., amn, classm, clusterm]
    ]
    '''
    # FIXME: Maybe just pass the class and cluster arrays
    classes = data[:, -2]
    classes_n = np.unique(classes)
    clusters = data[:, -1]
    clusters_n = np.unique(clusters)

    comb = lambda n, k: np.math.comb(n, k)

    n = len(data)
    a = [len(np.where(classes == cla)[0]) for cla in classes_n]
    b = [len(np.where(clusters == clu)[0]) for clu in clusters_n]
    nij = [[len(np.where((clusters == clu) & (classes == cla))[0]) for clu in clusters_n] for cla in classes_n]
    n2 = comb(n, 2)
    Eai2 = np.sum([comb(ai, 2) for ai in a])
    Ebj2 = np.sum([comb(bj, 2) for bj in b])
    Eij2 = np.sum([[comb(n, 2) for n in row] for row in nij])
    ARI = (Eij2-(Eai2*Ebj2)/n2)/(0.5*(Eai2+Ebj2)-(Eai2*Ebj2)/n2)

    return ARI


def FS(x, c, mu, alpha=1):
    '''
    Fuzzy Silhouette

    Parameters
    ----------
    x : Array data points
        DESCRIPTION.
    c : Cluster's centroids
        DESCRIPTION.
    mu : Membership matrix
        DESCRIPTION.
    alpha : Fuzzy weighted coefficient
        Default 1

    Returns
    -------
    Float [0,1].

    '''
    w = [1 for _ in range(len(x))]
    return WFS(x, c, w, mu, alpha)


def WFS(x, c, w, mu, alpha=1):
    '''
    Weighted Fuzzy Silhouette

    Parameters
    ----------
    x : Array data points
        DESCRIPTION.
    c : Cluster's centroids
        DESCRIPTION.
    w : Weigths of examples
        DESCRIPTION.
    mu : Membership matrix
        DESCRIPTION.
    alpha : Fuzzy weighted coefficient
        Default 1

    Returns
    -------
    Float [0,1].

    '''
    n, s = x.shape
    nc = len(c)
    dist = np.zeros([n, nc])
    for i in range(nc):
        dist[:, i] = np.sqrt(np.sum((x-c[i, :])**2, axis=1))  # Euclidean
    labels = np.argmin(dist, axis=1)
    NC = [sum(labels == i) for i in range(nc)]  # points per cluster
    dm = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            dm[i][j] = np.sqrt(sum((x[i] - x[j])**2))
            dm[j][i] = dm[i][j]

    s = []
    for i in range(n):
        # aij
        aij = 0
        cluster = labels[i]
        same_cluster_idx = np.argwhere(labels == cluster)[:, 0]
        for j in same_cluster_idx:
            if i != j:
                aij += dm[i, j]
        aij = aij/(NC[cluster]-1)

        # bij
        bij = []
        for p in range(nc):
            if p != cluster:
                bij.append(0)
                other_cluster_idx = np.argwhere(labels == p)[:, 0]
                for j in other_cluster_idx:
                    bij[-1] += dm[i, j]
                bij[-1] = bij[-1]/NC[p]
        bij = min(bij)

        si = (bij - aij)/max(aij, bij)
        s.append(si)

    # https://www.w3resource.com/python-exercises/numpy/advanced-numpy-exercise-11.php
    second_largest = np.partition(mu, -2, axis=0)
    muijp = second_largest[-1] - second_largest[-2]

    wfs = sum((muijp**alpha)*s*w)/sum((muijp**alpha)*w)
    return wfs

def Purity(fmics):
    partialpur = 0
    for idxFMIC, fmic in enumerate(fmics):
        majorityClass = max(fmic.tags.values())
        totalPoints = sum(fmic.tags.values())
        partialpur += majorityClass/totalPoints
    return (partialpur/len(fmics))

# def Purity(fmics):
#     majorityClass = 0
#     totalPoints = 0
#     for idxFMIC, fmic in enumerate(fmics):
#         # Asier: Changed to dict in fmic.py
#         majorityClass += np.max(list(fmic.sumPointsPerClassd.values()))
#         totalPoints += np.sum(list(fmic.sumPointsPerClassd.values()))

#     return (1/totalPoints * majorityClass)


def PartitionCoefficient(fmics, chunksize):
    mSquare = 0
    for idxFMIC, fmic in enumerate(fmics):
        mSquare += fmic.mSquare

    return (1/chunksize * mSquare)


def ModifiedPartitionCoefficient(fmics, chunksize):
    mSquare = 0
    for idxFMIC, fmic in enumerate(fmics):
        mSquare += fmic.mSquare

    return 1 - ((len(fmics)/len(fmics)-1) * (1 - (1/chunksize * mSquare)))


def PartitionEntropy(fmics, chunksize):
    mLog = 0
    for idxFMIC, fmic in enumerate(fmics):
        mLog += fmic.mLog

    return (- 1/chunksize * mLog)


def XieBeni(fmics, chunksize):
    sumaSSD = 0
    centroidList = np.ones((len(fmics), 2))*1000000
    menorDistancia = 1000000
    # storing the distances among all Fmics
    for idxFMIC, fmic in enumerate(fmics):
        sumaSSD += fmic.ssd
        centroidList[idxFMIC, :] = fmic.center

    MinDist = np.min(np.linalg.norm(centroidList, axis=1))

    return (1/chunksize * sumaSSD)/MinDist


def FukuyamaSugeno_1(fmics):
    sumaSSD = 0
    centroidList = np.ones((len(fmics), 2))
    membershipList = np.ones(len(fmics))

    for idxFMIC, fmic in enumerate(fmics):
        sumaSSD += fmic.ssd
        centroidList[idxFMIC, :] = fmic.center
        membershipList[idxFMIC] = fmic.m

    V1 = np.sum(centroidList/len(fmics), axis=0)

    return sumaSSD - np.sum(membershipList * np.linalg.norm(centroidList - V1, axis=1))


def FukuyamaSugeno_2(fmics, chunksize):
    sumaSSD = 0
    sumaValues = 0
    centroidList = np.ones((len(fmics), 2))
    membershipList = np.ones(len(fmics))

    for idxFMIC, fmic in enumerate(fmics):
        sumaSSD += fmic.ssd
        centroidList[idxFMIC, :] = fmic.center
        membershipList[idxFMIC] = fmic.m
        sumaValues += 1/chunksize * fmic.values

    V2 = sumaValues

    return sumaSSD - np.sum(membershipList * np.linalg.norm(centroidList - V2, axis=1))
