from fmic import FMiC
from functions.distance import EuclideanDistance
import numpy as np


class FuzzyDissimilarityMerger:

    def merge(fmics, threshold):
        fmics_to_merge = []
        similMatrix = np.zeros((5,5,2))

        
        for i in range(0, len(fmics) - 1):
            for j in range(i + 1, len(fmics)):
                similMatrix[i, j, 0] += np.minimum(fmics[i].__memberships, fmics[j].__memberships)      

                dissimilarity = EuclideanDistance.distance(fmics[i].center, fmics[j].center)
                sum_of_radius = fmics[i].radius + fmics[j].radius

                if dissimilarity != 0:
                    similarity = sum_of_radius / dissimilarity
                else:
                    # Highest value possible
                    similarity = 1.7976931348623157e+308

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
