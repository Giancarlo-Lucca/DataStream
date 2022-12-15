import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from d_fuzzstream import DFuzzStreamSummarizer
from functions.merge import FuzzyDissimilarityMerger
from functions.distance import EuclideanDistance
from functions.membership import FuzzyCMeansMembership
import pandas as pd


sm = 1
min_fmics = 5
max_fmics = 100
thresh = 0.5
#threshList = [0.5, 0.65, 0.8, 0.95]
threshList = [0.95]
chunksize=1000

df = pd.DataFrame(columns = ['Chunk', 'Purity', 'pCoefficient', 'pEntropy', 'XieBeni'])
new_row = {'Chunk':12, 'Purity':12, 'pCoefficient':12, 'pEntropy':12, 'XieBeni':12}
df2 = df.append(new_row, ignore_index=True)
print(df2)
print('**********************')
for simIDX in range (1, sm+1):
    for threshIDX in threshList:
        summarizer = DFuzzStreamSummarizer(
            distance_function=EuclideanDistance.distance,
            merge_threshold = threshIDX,
            merge_function=FuzzyDissimilarityMerger(14, max_fmics).merge,
            membership_function=FuzzyCMeansMembership.memberships,
            chunksize = chunksize
        )

        summary = {'x': [], 'y': [], 'weight': [], 'class': []}
        timestamp = 0

        # Read files in chunks
        with pd.read_csv("https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/Bench1_11k/Benchmark1_11000.csv",
                        dtype={"X1": float, "X2": float, "class": str},
                        chunksize = chunksize) as reader:       
            for chunk in reader:
                print(f"Summarizing examples from {timestamp} to {timestamp + 999} -> sim {simIDX} and thrsh {threshIDX}")

                for index, example in chunk.iterrows():
                    #Summarizing example
                    summarizer.summarize(example[0:2], example[2], timestamp)
                    timestamp += 1

                print("Purity: "+str(summarizer.Purity()))
                print("Partition Coeficient:"+str(summarizer.PartitionCoefficient()))
                print("Entropy: "+str(summarizer.PartitionEntropy()))
                print("Xie Benie:\n")

            # Transforming FMiCs into dataframe
            for fmic in summarizer.summary():
                summary['x'].append(fmic.center[0])
                summary['y'].append(fmic.center[1])
                summary['weight'].append(fmic.m)
                summary['class'].append(max(fmic.tags, key=fmic.tags.get))

        print("==== Approach ====")
        print("Similarity = ",simIDX)
        print("Threshold = ",threshIDX)
        print("==== Summary ====")
        print(summary)
        print("==== Metrics ====")
        print(summarizer.metrics)
        print("\n")
        output = "\n==== Approach ===="
        output = output + str("\n Similarity ="+str(simIDX))
        output = output + str("\n Threshold ="+str(threshIDX))
        output = output + str("\n ==== Summary ====")
        output = output + str("\n "+str(summary))
        output = output + str("\n ==== Metrics ====")
        output = output + str("\n "+str(summarizer.metrics))
        output = output + str("\n *** \n")
 
        with open('directOUTPUT.txt', 'a') as f:
            f.write(output)
    with open('directOUTPUT.txt', 'a') as f:
        f.write("\n------------------------------------------------------------")
print("--- End of execution --- ")
f.close()
