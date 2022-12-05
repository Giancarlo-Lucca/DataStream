from fmic import FMiC
from functions import distance
from functions import membership
from functions import merge


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
            #merge_function=merge.FuzzyDissimilarityMerger.merge
            merge_function=merge.FuzzyDissimilarityMerger(4, 100).merge
    ):
        self.min_fmics = min_fmics
        self.max_fmics = max_fmics
        self.merge_threshold = merge_threshold
        self.radius_factor = radius_factor
        self.m = m
        self.__fmics = []
        self.__memberships = []  #membership degree of the new point to all FMics
        self.__distance_function = distance_function
        self.__membership_function = membership_function
        self.__merge_function = merge_function
        self.metrics = {'creations': 0, 'absorptions': 0, 'removals': 0, 'merges': 0}

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

    def summary(self):
        return self.__fmics.copy()
