from fmic import FMiC
from functions import distance
from functions import membership
from functions import merge
from functions.WFCM import WFCM  # Asier: WFCM for offline step
import numpy as np


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
            n_macro_clusters=20,  # Asier: number of cluster for the WFCM
            time_gap=1000,  # Asier: When to apply the WFCM
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
        self.n_macro_clusters = n_macro_clusters  # Asier
        self.time_gap = time_gap  # Asier
        self._V = []   # Asier: Centroids of the offline step
        self._Vmm = []  # Asier: Membership matrix of the online step

    def summarize(self, values, tag, timestamp):
        if len(self.__fmics) < self.min_fmics:
            self.__fmics.append(FMiC(values, tag, timestamp))
            self.metrics['creations'] += 1

            # Asier: Update the actual clusters. It seems that in some cases
            # can happen that there is only a few mic and a new is created
            # on the offline evaluation timestamp, so it was not called due
            # to the return statement. So I add here the code
            if (timestamp + 1) % self.time_gap == 0:
                self.offline()

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
        if (timestamp + 1) % self.time_gap == 0:
            self.offline()

    # Asier
    def offline(self):
        data = np.array([fm.center.to_list() for fm in self.__fmics])
        w = [fm.m for fm in self.__fmics]  # Sum of membership
        f = [fm for fm in self.__fmics]
        self._V, self._Vmm = WFCM(data, w, c=self.n_macro_clusters)

    def final_clustering(self):
        if self._V == []:
            self.offline()

        return self._V, self._Vmm

    def summary(self):
        return self.__fmics.copy()
