import pandas as pd
from d_fuzzstream import DFuzzStreamSummarizer
from functions.merge import FuzzyDissimilarityMerger
from functions.distance import EuclideanDistance
from functions.membership import FuzzyCMeansMembership


summarizer = DFuzzStreamSummarizer(
    distance_function=EuclideanDistance.distance,
    merge_function=FuzzyDissimilarityMerger.merge,
    membership_function=FuzzyCMeansMembership.memberships
)

summary = {'x': [], 'y': [], 'weight': [], 'class': []}
timestamp = 0

# Read files in chunks
with pd.read_csv("https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/Bench1_11k/Benchmark1_11000.csv",
                 dtype={"X1": float, "X2": float, "class": str},
                 chunksize=1000) as reader:
    for chunk in reader:
        print(f"Summarizing examples from {timestamp} to {timestamp + 999}")
        for index, example in chunk.iterrows():
            # Summarizing example
            summarizer.summarize(example[0:2], example[2], timestamp)
            timestamp += 1

    # Transforming FMiCs into dataframe
    for fmic in summarizer.summary():
        summary['x'].append(fmic.center[0])
        summary['y'].append(fmic.center[1])
        summary['weight'].append(fmic.m)
        summary['class'].append(max(fmic.tags, key=fmic.tags.get))

print("==== Summary ====")
print(summary)
print("==== Metrics ====")
print(summarizer.metrics)
