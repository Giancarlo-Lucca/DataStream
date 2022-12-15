from fmic import FMiC
from functions.distance import EuclideanDistance
import numpy as np


class FuzzyDissimilarityMerger:
    def __init__(self, sm, max_fmics):
        self.similMatrix = np.zeros((max_fmics, max_fmics, 3))
        self.auxMatrix = np.zeros((max_fmics, max_fmics, 2))
        
        #self.similMatrix.flat[0::6] = 1
        self.sm = sm

    def merge(self, fmics, threshold, memberships):
        fmics_to_merge = []
        for i in range(0, len(fmics) - 1):
            for j in range(i + 1, len(fmics)):
                # Similarity S1 - euclidean
                if (self.sm == 1):
                    dissimilarity = EuclideanDistance.distance(fmics[i].center, fmics[j].center)
                    sum_of_radius = fmics[i].radius + fmics[j].radius
                    if dissimilarity != 0:
                        similarity = sum_of_radius / dissimilarity
                    else:
                        # Highest value possible
                        similarity = 1.7976931348623157e+308


                # Similarity S2 - SUMmin/SUMmax
                elif(self.sm == 2):
                    self.similMatrix[i, j, 0] += np.minimum(memberships[i], memberships[j])
                    self.similMatrix[i, j, 1] += np.maximum(memberships[i], memberships[j])
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]
                
                #================================================================================================================================
                            #S(A,B) = AM(REF(x_1,y_1), ... REF(x_n, y_n))
                #================================================================================================================================                
                elif(self.sm  == 3):
                    t = 10
                    self.similMatrix[i, j, 0] += np.power(1 - np.absolute(memberships[i] - memberships[j]), 1/t)
                    self.similMatrix[i, j, 1] += 1 #NAO È A MÉDIA É O NUMERO DE PONTOS!!!
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]
                    #print(similarity)

                #================================================================================================================================
                            #S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))    -> G = Probabilistic Sum (idx 4 to 8)
                #================================================================================================================================
                #O = Product
                elif(self.sm == 4):
                    Prod = memberships[i] * memberships[j]                      
                    self.similMatrix[i, j, 0] = self.similMatrix[i, j, 0] + Prod - self.similMatrix[i, j, 0] * Prod 
                    similarity = self.similMatrix[i, j, 0]

                #O = MIN
                elif(self.sm == 5):                     
                    min = np.minimum(memberships[i], memberships[j])
                    self.similMatrix[i, j, 0] = self.similMatrix[i, j, 0] + min - self.similMatrix[i, j, 0] * min
                    similarity = self.similMatrix[i, j, 0]

                #O = GM
                elif(self.sm == 6):                     
                    GM = np.sqrt(memberships[i] * memberships[j])
                    self.similMatrix[i, j, 0] = self.similMatrix[i, j, 0] + GM - self.similMatrix[i, j, 0] * GM
                    similarity = self.similMatrix[i, j, 0]

                #O = OB
                elif(self.sm == 7):                     
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum( memberships[i], memberships[j]))
                    self.similMatrix[i, j, 0] = self.similMatrix[i, j, 0] + OB - self.similMatrix[i, j, 0] * OB
                    similarity = self.similMatrix[i, j, 0]

                #O = ODiv
                elif(self.sm == 8):                     
                    ODiv = (memberships[i] * memberships[j] + np.minimum(memberships[i], memberships[j]))/2 
                    self.similMatrix[i, j, 0] = self.similMatrix[i, j, 0] + ODiv - self.similMatrix[i, j, 0] * ODiv
                    similarity = self.similMatrix[i, j, 0]

                #================================================================================================================================
                            #S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))    -> G = Maximum (idx 9 to 13)
                #================================================================================================================================
                #O = Product
                elif(self.sm == 9):
                    Prod = memberships[i] * memberships[j]                      
                    self.similMatrix[i, j, 0] = np.maximum(self.similMatrix[i, j, 0], Prod)
                    similarity = self.similMatrix[i, j, 0]

                #O = MIN
                elif(self.sm == 10):                     
                    min = np.minimum(memberships[i], memberships[j])
                    self.similMatrix[i, j, 0] = np.maximum(self.similMatrix[i, j, 0], min)
                    similarity = self.similMatrix[i, j, 0]

                #O = GM
                elif(self.sm == 11):                     
                    GM = np.sqrt(memberships[i] * memberships[j])
                    self.similMatrix[i, j, 0] = np.maximum(self.similMatrix[i, j, 0], GM)
                    similarity = self.similMatrix[i, j, 0]

                #O = OB
                elif(self.sm == 12):                     
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum( memberships[i], memberships[j]))
                    self.similMatrix[i, j, 0] = np.maximum(self.similMatrix[i, j, 0], OB)
                    similarity = self.similMatrix[i, j, 0]

                #O = ODiv
                elif(self.sm == 13):                     
                    ODiv = (memberships[i] * memberships[j] + np.minimum(memberships[i], memberships[j]))/2 
                    self.similMatrix[i, j, 0] = np.maximum(self.similMatrix[i, j, 0], ODiv)
                    similarity = self.similMatrix[i, j, 0]

                #================================================================================================================================
                            #S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))    -> G = G(GM) (idx 14 to 18)
                            #G(GM) = 1 - ((1-GM(x,y))^n * (1 - atual))^1/n+1
                #================================================================================================================================
                #O = Product
                elif(self.sm == 14):
                    Prod = memberships[i] * memberships[j]                 
                    if (self.similMatrix[i, j, 1] == 0):
                        # n = 0
                        self.auxMatrix[i, j, 1] = 1 - Prod
                        self.similMatrix[i, j, 0] = 1 - self.auxMatrix[i, j, 1] 
                        
                    else:
                        #similarity
                        self.auxMatrix[i, j, 1] *=  1 - Prod
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] , 1/self.similMatrix[i, j, 1]+1)
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0]
                    #print(similarity)
 

                #O = MIN
                elif(self.sm == 15):                     
                    min = np.minimum(memberships[i], memberships[j])                 
                    if (self.similMatrix[i, j, 1] == 0):
                        # n = 0
                        self.auxMatrix[i, j, 1] = 1 - Min
                        self.similMatrix[i, j, 0] = 1 - self.auxMatrix[i, j, 1] 
                        
                    else:
                        #similarity
                        self.auxMatrix[i, j, 1] *=  1 - Min
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] , 1/self.similMatrix[i, j, 1]+1)
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0]
                #O = GM
                elif(self.sm == 16):                     
                    GM = np.sqrt(memberships[i] * memberships[j])
                    if (self.similMatrix[i, j, 1] == 0):
                        # n = 0
                        self.auxMatrix[i, j, 1] = 1 - GM
                        self.similMatrix[i, j, 0] = 1 - self.auxMatrix[i, j, 1] 
                        
                    else:
                        #similarity
                        self.auxMatrix[i, j, 1] *=  1 - GM
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] , 1/self.similMatrix[i, j, 1]+1)
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0]

                #O = OB
                elif(self.sm == 17):                     
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum( memberships[i], memberships[j]))
                    if (self.similMatrix[i, j, 1] == 0):
                        # n = 0
                        self.auxMatrix[i, j, 1] = 1 - OB
                        self.similMatrix[i, j, 0] = 1 - self.auxMatrix[i, j, 1] 
                        
                    else:
                        #similarity
                        self.auxMatrix[i, j, 1] *=  1 - OB
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] , 1/self.similMatrix[i, j, 1]+1)
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0]

                #O = ODiv
                elif(self.sm == 18):                     
                    ODiv = (memberships[i] * memberships[j] + np.minimum(memberships[i], memberships[j]))/2 
                    if (self.similMatrix[i, j, 1] == 0):
                        # n = 0
                        self.auxMatrix[i, j, 1] = 1 - ODiv
                        self.similMatrix[i, j, 0] = 1 - self.auxMatrix[i, j, 1] 
                        
                    else:
                        #similarity
                        self.auxMatrix[i, j, 1] *=  1 - ODiv
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] , 1/self.similMatrix[i, j, 1]+1)
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0]

                #================================================================================================================================
                            #S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))    -> G = Dual(OB) (idx 19 to 13)
                            #GB = 1 - np.power(np.Prod(1 - memberships[i], 1 - memberships[j]) * np.minimum(1 - memberships[i], 1 - memberships[j]), 1/2)
                #================================================================================================================================
                #O = Product
                elif(self.sm == 19):
                    Prod = memberships[i] * memberships[j]   
                    min = np.minimum(memberships[i], memberships[j]) 
                    #Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):     
                        self.auxMatrix[i, j, 0] = min
                        self.auxMatrix[i, j, 1] = 1 - Prod
                    #Senão n>=1    
                    else:
                        self.auxMatrix[i, j, 0] = np.minimum(self.auxMatrix[i, j, 0], min)
                        self.auxMatrix[i, j, 1] *= 1 - Prod
                        #GB =  1 - (PROD * MIN)^{1/2}
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] * self.auxMatrix[i, j, 0], 1/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]
                    #print(similarity)
   
                #O = MIN
                elif(self.sm == 20):                     
                    min = np.minimum(memberships[i], memberships[j])

                #O = GM
                elif(self.sm == 21):                     
                    GM = np.sqrt(memberships[i] * memberships[j])
                    
                #O = OB
                elif(self.sm == 22):                     
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum( memberships[i], memberships[j]))
                    
                #O = ODiv
                elif(self.sm == 23):                     
                    ODiv = (memberships[i] * memberships[j] + np.minimum(memberships[i], memberships[j]))/2 
                    
                    
                #================================================================================================================================
                            #S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))    -> G = Dual(ODiv) (idx 24 to 28)
                #================================================================================================================================
                #O = Product
                elif(self.sm == 24):
                    Prod = memberships[i] * memberships[j]   
                    min = np.minimum(memberships[i], memberships[j])
                    #Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):     
                        self.auxMatrix[i, j, 0] = min
                        self.auxMatrix[i, j, 1] = 1 - Prod
                    #Senão n>=1    
                    else:
                        self.auxMatrix[i, j, 0] = np.minimum(self.auxMatrix[i, j, 0], min)
                        self.auxMatrix[i, j, 1] *= 1 - Prod
                        #GDiv =  1 - (PROD * MIN)^{1/2}
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]


                #O = MIN
                elif(self.sm == 25):                     
                    min = np.minimum(memberships[i], memberships[j])
                    #Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):     
                        self.auxMatrix[i, j, 0] = min
                        self.auxMatrix[i, j, 1] = 1 - min
                    #Senão n>=1    
                    else:
                        self.auxMatrix[i, j, 0] = np.minimum(self.auxMatrix[i, j, 0], min)
                        self.auxMatrix[i, j, 1] *= 1 - min
                        #GDiv =  1 - (PROD * MIN)^{1/2}
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]  

                #O = GM
                elif(self.sm == 26):                     
                    GM = np.sqrt(memberships[i] * memberships[j])
                    min = np.minimum(memberships[i], memberships[j])
                    #Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):     
                        self.auxMatrix[i, j, 0] = min
                        self.auxMatrix[i, j, 1] = 1 - GM
                    #Senão n>=1    
                    else:
                        self.auxMatrix[i, j, 0] = np.minimum(self.auxMatrix[i, j, 0], min)
                        self.auxMatrix[i, j, 1] *= 1 - GM
                        #GDiv =  1 - (PROD * MIN)^{1/2}
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]  
 

                #O = OB
                elif(self.sm == 27):                     
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum( memberships[i], memberships[j]))
                    min = np.minimum(memberships[i], memberships[j])
                    #Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):     
                        self.auxMatrix[i, j, 0] = min
                        self.auxMatrix[i, j, 1] = 1 - OB
                    #Senão n>=1    
                    else:
                        self.auxMatrix[i, j, 0] = np.minimum(self.auxMatrix[i, j, 0], min)
                        self.auxMatrix[i, j, 1] *= 1 - OB
                        #GDiv =  1 - (PROD * MIN)^{1/2}
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]  


                #O = ODiv
                elif(self.sm == 28):                     
                    ODiv = (memberships[i] * memberships[j] + np.minimum(memberships[i], memberships[j]))/2 
                    min = np.minimum(memberships[i], memberships[j])
                    #Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):     
                        self.auxMatrix[i, j, 0] = min
                        self.auxMatrix[i, j, 1] = 1 - ODiv
                    #Senão n>=1    
                    else:
                        self.auxMatrix[i, j, 0] = np.minimum(self.auxMatrix[i, j, 0], min)
                        self.auxMatrix[i, j, 1] *= 1 - ODiv
                        #GDiv =  1 - (PROD * MIN)^{1/2}
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]  

                #================================================================================================================================
                            #S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))    -> G = MÉDIA (idx 29 to 33)
                #================================================================================================================================
                #O = Product
                elif(self.sm == 29):
                    Prod = memberships[i] * memberships[j]
                    self.similMatrix[i, j, 0] += Prod
                    self.similMatrix[i, j, 1] += 1 
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]                                          

                #O = MIN
                elif(self.sm == 30):                     
                    min = np.minimum(memberships[i], memberships[j])
                    self.similMatrix[i, j, 0] += Min
                    self.similMatrix[i, j, 1] += 1 
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]    

                #O = GM
                elif(self.sm == 31):                     
                    GM = np.sqrt(memberships[i] * memberships[j])
                    self.similMatrix[i, j, 0] += GM
                    self.similMatrix[i, j, 1] += 1 
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]   

                #O = OB
                elif(self.sm == 32):                     
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum( memberships[i], memberships[j]))
                    self.similMatrix[i, j, 0] += OB
                    self.similMatrix[i, j, 1] += 1 
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]  

                #O = ODiv
                elif(self.sm == 33):                     
                    ODiv = (memberships[i] * memberships[j] + np.minimum(memberships[i], memberships[j]))/2 
                    self.similMatrix[i, j, 0] += ODiv
                    self.similMatrix[i, j, 1] += 1 
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]  


                if similarity >= threshold:
                    #print("inside threshold")
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
