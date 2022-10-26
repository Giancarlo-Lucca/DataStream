from fmic import FMiC
from functions.distance import EuclideanDistance
import numpy as np


class FuzzyDissimilarityMerger:
    def __init__(self, sm = 1):
        self.similMatrix = np.zeros((5, 5, 2))
        self.sm = 2

    def merge(self, fmics, threshold, memberships):
        fmics_to_merge = []
        
        for i in range(0, len(fmics) - 1):
            for j in range(i + 1, len(fmics)):
                if (sm == 1):
                    dissimilarity = EuclideanDistance.distance(fmics[i].center, fmics[j].center)
                    sum_of_radius = fmics[i].radius + fmics[j].radius
                    if dissimilarity != 0:
                        similarity = sum_of_radius / dissimilarity
                    else:
                        # Highest value possible
                        similarity = 1.7976931348623157e+308

                elif (self.sm == 2):
                    self.similMatrix[i, j, 0] += np.minimum(memberships[i], memberships[j])
                    self.similMatrix[i, j, 1] += np.maximum(memberships[i], memberships[j])
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]
                
                
                if similarity >= threshold:
                    fmics_to_merge.append([i, j, similarity])

        # Sort by most similar
        fmics_to_merge.sort(reverse=True, key=lambda k: k[2])
        merged_fmics_idx = []
        merged_fmics = []

        for (i, j, _) in fmics_to_merge:
            if i not in merged_fmics_idx and j not in merged_fmics_idx:
                merged_fmics.append(FMiC.merge(fmics[i], fmics[j]))
                merged_fmics_idx.append(i)
                merged_fmics_idx.append(j)

        merged_fmics_idx.sort(reverse=True)
        for idx in merged_fmics_idx:
            fmics.pop(idx)

        return fmics + merged_fmics
