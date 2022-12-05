from fmic import FMiC
from functions.distance import EuclideanDistance
import numpy as np


class FuzzyDissimilarityMerger:
    def __init__(self, sm, max_fmics):
        self.similMatrix = np.zeros((max_fmics, max_fmics, 2))
        #self.similMatrix.flat[0::6] = 1
        self.sm = sm

    def merge(self, fmics, threshold, memberships):
        fmics_to_merge = []
        for i in range(0, len(fmics) - 1):
            for j in range(i + 1, len(fmics)):
                if (self.sm == 1):
                    dissimilarity = EuclideanDistance.distance(fmics[i].center, fmics[j].center)
                    sum_of_radius = fmics[i].radius + fmics[j].radius
                    if dissimilarity != 0:
                        similarity = sum_of_radius / dissimilarity
                    else:
                        # Highest value possible
                        similarity = 1.7976931348623157e+308
 
                elif(self.sm == 2):
                    
                    self.similMatrix[i, j, 0] += np.minimum(memberships[i], memberships[j])
                    self.similMatrix[i, j, 1] += np.maximum(memberships[i], memberships[j])
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]
                
                #S(A,B) = AM(REF(x_1,y_1), ... REF(x_n, y_n))
                elif(self.sm  == 3):
                    t = 10
                    self.similMatrix[i, j, 0] += np.power(1 - np.absolute(memberships[i] - memberships[j]), 1/t)
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]
                    #print(similarity)

                #S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))
                # O = product   #G = probabilistic sum -> x1 + x2 − x1 · x2
                elif(self.sm == 4):
                    overlap = memberships[i] * memberships[j]                      
                    self.similMatrix[i, j, 0] = self.similMatrix[i, j, 0] + overlap - self.similMatrix[i, j, 0] * overlap 
                    similarity = self.similMatrix[i, j, 0]
                    #print(similarity)

                #S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))
                #O = min #G = max
                elif(self.sm == 5):                     
                    min = np.minimum(memberships[i], memberships[j])
                    self.similMatrix[i, j, 0] = np.maximum(self.similMatrix[i, j, 0], min)
                    similarity = self.similMatrix[i, j, 0]

                    
                #S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))
                #O = GM        #G = 
                elif(self.sm == 6):
                    GM = np.sqrt(memberships[i] * memberships[j])
                    #self.similMatrix[i, j, 0] += 1 - ()
                    #self.similMatrix[i, j, 1] += 1
                #S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))
                #O = OB        #G = 
                elif(self.sm == 7):
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum( memberships[i], memberships[j]))
                    #self.similMatrix[i, j, 0] +=

                #S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))
                #O = Div       #G = 
                elif(self.sm == 8):
                    ODiv = np.sqrt(memberships[i] * memberships[j])    
                
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
