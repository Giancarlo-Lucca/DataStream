import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from d_fuzzstream import DFuzzStreamSummarizer
from functions.merge import FuzzyDissimilarityMerger
from functions.distance import EuclideanDistance
from functions.membership import FuzzyCMeansMembership
from functions import metrics

sm = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
min_fmics = 5
max_fmics = 100
thresh = 0.5

color = {'1': 'Red', '2': 'Blue', '3': 'Green', 'nan': 'pink'}
figure = plt.figure()
scatter = plt.scatter('x', 'y', s='radius', data={'x': [], 'y': [], 'radius': []})

datasetName = 'Benchmark1_11000'  # 'RBF1_40000',  'Benchmark1_11000'

if (datasetName == 'Benchmark1_11000'):
    datasetPath = "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/Bench1_11k/Benchmark1_11000.csv"
    threshList = [0.9, 0.9, 0.9, 0.9, 0.8, 0.9, 0.65, 0.1, 0.25, 0.1, 0.25, 0.25, 0.8, 0.8, 0.65, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.8, 0.9, 0.05, 0.05, 0.05, 0.05, 0.05]
    numChunks = 11
    chunksize = 1000
elif (datasetName == 'RBF1_40000'):
    datasetPath = "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/RBF1_40k/RBF1_40000.csv"
    threshList = [0.9, 0.9, 0.65, 0.9, 0.9, 0.8, 0.9, 0.1, 0.25, 0.1, 0.25, 0.25, 0.65, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.8, 0.9, 0.05, 0.05, 0.05, 0.05, 0.05]
    numChunks = 40
    chunksize = 1000
output_path = "".join(("./output/", datasetName, "/"))

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
Path(output_path).mkdir(exist_ok=True)

tabRes = pd.DataFrame(np.zeros((32, (numChunks * 2) + 2)))

for vecIndex, simIDX in enumerate(sm):
    # tabRes = pd.DataFrame(np.zeros((numChunks+4,5)))
    # tabRes.columns = [0.25,0.5,0.65,0.8,0.9]
    threshIDX = threshList[vecIndex]
    df = pd.DataFrame(columns=['Chunk', 'Purity', 'pCoefficient',
                               'pEntropy', 'XieBeni', 'MPC',
                               'FukuyamaSugeno_1', 'FukuyamaSugeno_2'])
    summarizer = DFuzzStreamSummarizer(
        distance_function=EuclideanDistance.distance,
        merge_threshold=threshIDX,
        merge_function=FuzzyDissimilarityMerger(simIDX, max_fmics).merge,
        membership_function=FuzzyCMeansMembership.memberships,
        chunksize=chunksize,
        n_macro_clusters=2
    )

    summary = {'x': [], 'y': [], 'radius': [], 'color': [], 'weight': [], 'class': []}
    timestamp = 0

    fhand = open('chunkFMICs.txt', 'a')

    # Read files in chunks
    with pd.read_csv(datasetPath,
                     dtype={"X1": float, "X2": float, "class": str},
                     chunksize=chunksize) as reader:
        for chunk in reader:
            log_text = (f"Summarizing examples from {timestamp} to "
                        f"{timestamp + chunksize-1} -> sim {simIDX} "
                        f"and thrsh {threshIDX}")
            print(log_text)
            fhand.write(log_text + "\n")
            for index, example in chunk.iterrows():
                # Summarizing example
                summarizer.summarize(example[0:2], example[2], timestamp)
                timestamp += 1

                # Offline - Evaluation
                if (timestamp) % summarizer.time_gap == 0:
                    # FIMXE: Error in Sim 6 and thrsh 0.8
                    om = metrics.all_offline_metrics(summarizer._V,
                                                     summarizer._Vmm,
                                                     summarizer.summary())
                    max_memb = np.max(summarizer._Vmm, axis=0)
                    tot_memb = np.sum(summarizer._Vmm, axis=0)
                    purity = np.sum(max_memb / tot_memb) / len(max_memb)
                    print(f"Offline purity for {timestamp}: {purity}")
                    print(om)

                    summarymc = {'x': [], 'y': [], 'class': []}
                    for fmic in summarizer.summary():
                        summarymc['x'].append(fmic.center[0])
                        summarymc['y'].append(fmic.center[1])
                        summarymc['class'].append(color[max(fmic.tags,
                                                        key=fmic.tags.get)])

                    mccenters = {'x': [], 'y': []}
                    for cen in summarizer._V:
                        mccenters['x'].append(cen[0])
                        mccenters['y'].append(cen[1])

                    te = ''
                    for k, v in om.items():
                        te += f"{k}: {v:.6f}\n"

                    fig = plt.figure()
                    # Plot centroids
                    plt.scatter('x', 'y', color='class', data=summarymc)
                    plt.scatter('x', 'y', color='green', data=mccenters)
                    fig_title = (f"{summarizer.n_macro_clusters}  MacroClusters"
                                 f" - Timestamp {timestamp - 1000}"
                                 f"to {timestamp - 1}")
                    plt.title(fig_title)
                    side_text = plt.figtext(.92, .5, te)
                    fig.subplots_adjust(top=1.0)
                    # plt.show()
                    fig_name = (f"./Img/MC[Chunk {timestamp - 1000} to "
                                f"{timestamp - 1}] "
                                f"Sim({simIDX})_Thresh({threshIDX}).png")
                    fig.savefig(fig_name,
                                bbox_extra_artists=(side_text,),
                                bbox_inches='tight')
                    plt.close()

            # TODO: Obtain al metrics and create the row
            all_metrics = metrics.all_online_metrics(summarizer.summary(),
                                                     chunksize)
            metrics_summary = ""
            for name, value in all_metrics.items():
                metrics_summary += f"{name}: {round(value,3)}\n"
            metrics_summary = metrics_summary[:-1]

            row_metrics = list(all_metrics.values())
            row_timestamp = ["[" + str(timestamp) + " to " + str(timestamp + chunksize - 1) + "]"]

            new_row = pd.DataFrame([row_timestamp + row_metrics],
                                   columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)

            fhand.write("Total de Fmics = " + str(len(summarizer.summary())))
            # print("Total de Fmics = "+str(len(summarizer.summary())))
            for fmic in summarizer.summary():
                for k, v in fmic.sumPointsPerClassd.items():  # FIXME: Not sorted, but sorted() has problems with nan
                    # print(f"Total pontos classe {k} = {v}")
                    fhand.write(f"\nTotal pontos classe {k} = {v} \n")
                fhand.write("------------------")

                summary['x'].append(fmic.center[0])
                summary['y'].append(fmic.center[1])
                summary['radius'].append(fmic.radius * 100000)
                summary['color'].append(color[max(fmic.tags, key=fmic.tags.get)])
                summary['weight'].append(fmic.m)

            if not os.path.isdir("./Img/"):
                os.mkdir("./Img/")

            fig = plt.figure()
            # Plot radius
            plt.scatter('x', 'y', s='radius', color='color',
                        data=summary, alpha=0.1)
            # Plot centroids
            plt.scatter('x', 'y', s=1, color='color', data=summary)
            # plt.legend(["color blue", "color green"], loc ="lower right")
            # plt.legend(["Purity"+str(summarizer.Purity()),"PartitionCoefficient"+str(summarizer.PartitionCoefficient()),"PartitionEntropy"+str(summarizer.PartitionEntropy()),"XieBeni"+str(summarizer.XieBeni()), "FukuyamaSugeno_1"+str(summarizer.FukuyamaSugeno_1()),"FukuyamaSugeno_2"+str(summarizer.FukuyamaSugeno_2())], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            plt.figtext(.8, .8, "T = 4K")
            side_text = plt.figtext(.91, .8, metrics_summary)
            fig.subplots_adjust(top=1.0)
            # plt.show()
            fig_name = (f"./Img/[Chunk {timestamp - 1000} to {timestamp - 1})]"
                        f" Sim({simIDX})_Thresh({threshIDX}).png")
            fig.savefig(fig_name, bbox_extra_artists=(side_text,),
                        bbox_inches='tight')
            plt.close()

        # Transforming FMiCs into dataframe
        for fmic in summarizer.summary():
            summary['x'].append(fmic.center[0])
            summary['y'].append(fmic.center[1])
            summary['radius'].append(fmic.radius * 100000)
            summary['color'].append(color[max(fmic.tags, key=fmic.tags.get)])
            summary['weight'].append(fmic.m)
            summary['class'].append(max(fmic.tags, key=fmic.tags.get))

        print("==== Approach ====")
        print("Similarity = ", simIDX)
        print("Threshold = ", threshIDX)
        # print("==== Summary ====")
        # print(summary)
        print("==== Metrics ====")
        print(summarizer.metrics)
        print("\n")
        print(df)
        print("------")

        tabRes.iloc[vecIndex, list(range(0, numChunks * 2, 2))] = df['XieBeni']
        tabRes.iloc[vecIndex, list(range(1, numChunks * 2, 2))] = df['MPC']
        tabRes.iloc[vecIndex, -2] = df['XieBeni'].mean()
        tabRes.iloc[vecIndex, -1] = df['MPC'].mean()

        output = "\n==== Approach ===="
        output = output + str("\n Similarity =" + str(simIDX))
        output = output + str("\n Threshold =" + str(threshIDX))
        output = output + str("\n ==== Summary ====")
        output = output + str("\n " + str(summary))
        output = output + str("\n ==== Metrics ====")
        output = output + str("\n " + str(summarizer.metrics))
        output = output + str("\n ")
        output = output + str("\n ==== Evaluation ====")
        output = output + str("\n " + df.to_markdown(tablefmt='greed'))
        output = output + str("\n *****")
        df = df[0:0]
        # df.drop(df.index, inplace=True)
        with open('directOUTPUT.txt', 'a') as f:
            f.write(output)

tabRes.to_excel("".join((output_path, "Example_Table.xlsx")))
# tabRes.to_excel("./output/XieBeni_sm" + str(simIDX) + ".xlsx")
with open('directOUTPUT.txt', 'a') as f:
    f.write("\n------------------------------------------------------------")
print("--- End of execution --- ")
f.close()
fhand.close()
