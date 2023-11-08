from fmic import FMiC
from functions import distance
from functions import membership
from functions import merge
from functions.WFCM import WFCM  # Asier: WFCM for offline step
import numpy as np
import pandas as pd  # Asier

class DFuzzStreamSummarizer:

    def __init__(
            self,
            min_fmics=5,
            max_fmics=100,
            merge_threshold=0.8,
            radius_factor=1.0,
            m=2.0,
            distance_function=distance.EuclideanDistance.distance,
            membership_function=membership.FuzzyCMeansMembership.memberships,
            # merge_function=merge.FuzzyDissimilarityMerger.merge
            merge_function=merge.FuzzyDissimilarityMerger(1, 100).merge,
            chunksize=1000,
            n_macro_clusters=20, # Asier: number of cluster for the WFCM
            time_gap=10000, # Asier: When to apply the WFCM
    ):
        self.min_fmics = min_fmics
        self.max_fmics = max_fmics
        self.merge_threshold = merge_threshold
        self.radius_factor = radius_factor
        self.m = m
        self.__fmics = []
        self.__memberships = []  # membership degree of the new point to all FMics
        self.__distance_function = distance_function
        self.__membership_function = membership_function
        self.__merge_function = merge_function
        self.metrics = {'creations': 0, 'absorptions': 0, 'removals': 0, 'merges': 0}
        self.chunksize = chunksize
        self.n_macro_clusters=n_macro_clusters # Asier
        self.time_gap = time_gap # Asier
        self._V = []   # Asier: Centroids of the offline step
        self._Vmm = []  # Asier: Membership matrix of the online step

    def summarize(self, values, tag, timestamp):
        if len(self.__fmics) < self.min_fmics:
            self.__fmics.append(FMiC(values, tag, timestamp))
            self.metrics['creations'] += 1
            return

        distance_from_fmics = [self.__distance_function(fmic.center, values) for fmic in self.__fmics]
        is_outlier = True

        for idx, fmic in enumerate(self.__fmics):
            if fmic.radius == 0.0:
                # Minimum distance from another FMiC
                radius = min([
                    self.__distance_function(fmic.center, another_fmic.center)
                    for another_idx, another_fmic in enumerate(self.__fmics)
                    if another_idx != idx
                ])
            else:
                radius = fmic.radius * self.radius_factor

            if distance_from_fmics[idx] <= radius:
                is_outlier = False
                fmic.timestamp = timestamp

        if is_outlier:
            if len(self.__fmics) >= self.max_fmics:
                oldest = min(self.__fmics, key=lambda f: f.timestamp)
                self.__fmics.remove(oldest)
                self.metrics['removals'] += 1
            self.__fmics.append(FMiC(values, tag, timestamp))
            self.__memberships.append(1)
            self.metrics['creations'] += 1
        else:
            memberships = self.__membership_function(distance_from_fmics, self.m)
            self.__memberships = memberships
            for idx, fmic in enumerate(self.__fmics):
                fmic.assign(values, tag, memberships[idx], distance_from_fmics[idx])
            self.metrics['absorptions'] += 1
            number_of_fmics = len(self.__fmics)
            self.__fmics = self.__merge_function(self.__fmics, self.merge_threshold, self.__memberships)
            self.metrics['merges'] += number_of_fmics - len(self.__fmics)

        # Asier: Update the actual clusters
        if timestamp % self.time_gap == 0:
            self.offline()

    # Asier
    def offline(self):
        data = np.array([fm.center.to_list() for fm in self.__fmics])
        w = [fm.n for fm in self.__fmics]
        self._V, self._Vmm = WFCM(data, w, c=self.n_macro_clusters)

    def final_clustering(self):
        if self._V == []:
            self.offline()

        return self._V, self._Vmm


    # Asier: Metrics better in functions.metrics
    # def Purity(self):
    #     majorityClass = 0
    #     totalPoints = 0
    #     for idxFMIC, fmic in enumerate(self.__fmics):
    #         majorityClass += np.max(fmic.sumPointsPerClass)
    #         totalPoints += np.sum(fmic.sumPointsPerClass)

    #     return (1/totalPoints * majorityClass)

    # def PartitionCoefficient(self):
    #     mSquare = 0
    #     for idxFMIC, fmic in enumerate(self.__fmics):
    #         mSquare += fmic.mSquare

    #     return (1/self.chunksize * mSquare)

    # def ModifiedPartitionCoefficient(self):
    #     mSquare = 0
    #     for idxFMIC, fmic in enumerate(self.__fmics):
    #         mSquare += fmic.mSquare

    #     return 1 - ((len(self.__fmics)/len(self.__fmics)-1) * (1 - (1/self.chunksize * mSquare)))

    # def PartitionEntropy(self):
    #     mLog = 0
    #     for idxFMIC, fmic in enumerate(self.__fmics):
    #         mLog += fmic.mLog

    #     return (- 1/self.chunksize * mLog)

    # def XieBeni(self):
    #     sumaSSD = 0
    #     centroidList = np.ones((len(self.__fmics), 2))*1000000
    #     menorDistancia = 1000000
    #     # storing the distances among all Fmics
    #     for idxFMIC, fmic in enumerate(self.__fmics):
    #         sumaSSD += fmic.ssd
    #         centroidList[idxFMIC, :] = fmic.center

    #     MinDist = np.min(np.linalg.norm(centroidList, axis=1))

    #     return (1/self.chunksize * sumaSSD)/MinDist

    # def FukuyamaSugeno_1(self):
    #     sumaSSD = 0
    #     centroidList = np.ones((len(self.__fmics), 2))
    #     membershipList = np.ones(len(self.__fmics))

    #     for idxFMIC, fmic in enumerate(self.__fmics):
    #         sumaSSD += fmic.ssd
    #         centroidList[idxFMIC, :] = fmic.center
    #         membershipList[idxFMIC] = fmic.m

    #     V1 = np.sum(centroidList/len(self.__fmics), axis=0)

    #     return sumaSSD - np.sum(membershipList * np.linalg.norm(centroidList - V1, axis=1))

    # def FukuyamaSugeno_2(self):
    #     sumaSSD = 0
    #     sumaValues = 0
    #     centroidList = np.ones((len(self.__fmics), 2))
    #     membershipList = np.ones(len(self.__fmics))

    #     for idxFMIC, fmic in enumerate(self.__fmics):
    #         sumaSSD += fmic.ssd
    #         centroidList[idxFMIC, :] = fmic.center
    #         membershipList[idxFMIC] = fmic.m
    #         sumaValues += 1/self.chunksize * fmic.values

    #     V2 = sumaValues

    #     return sumaSSD - np.sum(membershipList * np.linalg.norm(centroidList - V2, axis=1))

    def summary(self):
        return self.__fmics.copy()
