import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
sys.path.append(os.path.abspath("."))
from src.d_fuzzstream import DFuzzStreamSummarizer
from src.functions.merge import FuzzyDissimilarityMerger
from src.functions.merge import AllMergers
from src.functions.distance import EuclideanDistance
from src.functions.membership import FuzzyCMeansMembership
from src.functions import metrics

sm = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
min_fmics = 5

start, end = 0,32

color = {'1': 'Red', '2': 'Blue', '3': 'Green', 'nan': 'gray', '-1':'gray'}
figure = plt.figure()
scatter = plt.scatter('x', 'y', s='radius', data={'x': [], 'y': [], 'radius': []})

datasetName = 'Benchmark1_11000'  # 'RBF1_40000',  'Benchmark1_11000'

if (datasetName == 'Benchmark1_11000'):
    datasetPath = "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/Bench1_11k/Benchmark1_11000.csv"
    threshList = [0.8, 0.9, 0.25, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.8, 0.25, 0.25, 0.65, 0.65, 0.8, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
    numChunks = 11
    chunksize = 1000
    n_clusters = 2
    max_fmics = 50
elif (datasetName == 'RBF1_40000'):
    datasetPath = "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/RBF1_40k/RBF1_40000.csv"
    threshList = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
    numChunks = 10
    chunksize = 4000
    n_clusters = 3
    max_fmics = 100
elif (datasetName == 'Insects'):
    datasetPath = "../datasets/INSECTS-incremental_balanced_norm.csv"  # https://gitlab.citius.usc.es/david.gonzalez.marquez/GaussianMotionData/-/raw/master/SamplesFile_b_4C2D800Linear.csv?ref_type=heads
    threshList = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                  0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
    numChunks = 19
    chunksize = 3000
    n_clusters = 6
    max_fmics = 100


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
currentPath = Path.cwd()
output_path = currentPath / "output"/ datasetName
Path(output_path).mkdir(parents=True,exist_ok=True)

tabRes = pd.DataFrame(np.zeros((32, (numChunks * 3) + 3)))

for vecIndex, simIDX in enumerate(sm[start:end]):
    # tabRes = pd.DataFrame(np.zeros((numChunks+4,5)))
    # tabRes.columns = [0.25,0.5,0.65,0.8,0.9]
    threshIDX = threshList[start+vecIndex]
    df = pd.DataFrame(columns=['Chunk', 'Purity', 'pCoefficient',
                               'pEntropy', 'XieBeni', 'MPC',
                               'FukuyamaSugeno_1', 'FukuyamaSugeno_2'])
    summarizer = DFuzzStreamSummarizer(
        max_fmics=max_fmics,
        distance_function=EuclideanDistance.distance,
        merge_threshold=threshIDX,
        # merge_function=FuzzyDissimilarityMerger(simIDX, threshIDX, max_fmics).merge,
        merge_function=AllMergers[simIDX](simIDX, threshIDX, max_fmics),
        membership_function=FuzzyCMeansMembership.memberships,
        chunksize=chunksize,
        n_macro_clusters=n_clusters,
        time_gap=chunksize
    )

    summary = {'x': [], 'y': [], 'radius': [], 'color': [], 'weight': [], 'class': []}
    timestamp = 0

    fhand = open(f'{output_path}/chunkFMICs.txt', 'a')

    # Read files in chunks
    with pd.read_csv(datasetPath,
                     dtype={"X1": float, "X2": float, "class": str},
                     chunksize=chunksize) as reader:
        for chunk in reader:
            log_text = (f"Summarizing examples from {timestamp} to "
                        f"{timestamp + chunksize-1} -> sim {simIDX} "
                        f"and thrsh {threshIDX}")
            # print(log_text)
            fhand.write(log_text + "\n")
            for index, example in chunk.iterrows():
                # Summarizing example
                ex_data = example[0:-1]
                ex_class = example[-1]
                summarizer.summarize(ex_data, ex_class, timestamp)
                timestamp += 1

                # Offline - Evaluation
                if (timestamp) % summarizer.time_gap == 0:
                    ari, sil = metrics.offline_stats(summarizer, chunk)
                    print(simIDX,timestamp,ari,sil)

                    # om = metrics.all_offline_metrics(summarizer._V,
                    #                                  summarizer._Vmm,
                    #                                  summarizer.summary())
                    # max_memb = np.max(summarizer._Vmm, axis=0)
                    # tot_memb = np.sum(summarizer._Vmm, axis=0)
                    # purity = np.sum(max_memb / tot_memb) / len(max_memb)
                    # print(f"Offline purity for {timestamp}: {purity}")
                    # print(om)
                    #
                    # summarymc = {'x': [], 'y': [], 'class': []}
                    # for fmic in summarizer.summary():
                    #     summarymc['x'].append(fmic.center[0])
                    #     summarymc['y'].append(fmic.center[1])
                    #     summarymc['class'].append(color[max(fmic.tags,
                    #                                     key=fmic.tags.get)])
                    #
                    # mccenters = {'x': [], 'y': []}
                    # for cen in summarizer._V:
                    #     mccenters['x'].append(cen[0])
                    #     mccenters['y'].append(cen[1])
                    #
                    # te = ''
                    # for k, v in om.items():
                    #     te += f"{k}: {v:.6f}\n"
                    #
                    # fig = plt.figure()
                    # # Plot centroids
                    # plt.scatter('x', 'y', color='class', data=summarymc)
                    # plt.scatter('x', 'y', color='green', data=mccenters)
                    # fig_title = (f"{summarizer.n_macro_clusters}  MacroClusters"
                    #              f" - Timestamp {timestamp - 1000}"
                    #              f"to {timestamp - 1}")
                    # plt.title(fig_title)
                    # side_text = plt.figtext(.92, .5, te)
                    # fig.subplots_adjust(top=1.0)
                    # # plt.show()
                    # fig_name = (f"{output_path}/Img/MC[Chunk {timestamp - 1000} to "
                    #             f"{timestamp - 1}] "
                    #             f"Sim({simIDX})_Thresh({threshIDX}).png")
                    # fig.savefig(fig_name,
                    #             bbox_extra_artists=(side_text,),
                    #             bbox_inches='tight')
                    # plt.close()

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

            if not os.path.isdir(f"{output_path}/Img/"):
                os.mkdir(f"{output_path}/Img/")

            fig = plt.figure()
            # Plot radius
            plt.scatter('x', 'y', s='radius', color='color',
                        data=summary, alpha=0.1)
            # Plot centroids
            plt.scatter('x', 'y', s=1, color='color', data=summary)
            # plt.legend(["color blue", "color green"], loc ="lower right")
            # plt.legend(["Purity"+str(summarizer.Purity()),"PartitionCoefficient"+str(summarizer.PartitionCoefficient()),"PartitionEntropy"+str(summarizer.PartitionEntropy()),"XieBeni"+str(summarizer.XieBeni()), "FukuyamaSugeno_1"+str(summarizer.FukuyamaSugeno_1()),"FukuyamaSugeno_2"+str(summarizer.FukuyamaSugeno_2())], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            # plt.figtext(.8, .8, "T = 4K")
            # side_text = plt.figtext(.91, .8, metrics_summary)
            fig.subplots_adjust(top=1.0)
            # plt.show()
            fig_name = (f"{output_path}/Img/[Chunk {timestamp - 1000} to {timestamp - 1})]"
                        f" Sim({simIDX})_Thresh({threshIDX}).png")
            fig.savefig(fig_name, # bbox_extra_artists=(side_text,),
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

        tabRes.iloc[vecIndex, list(range(0, numChunks * 3, 3))] = df['XieBeni']
        tabRes.iloc[vecIndex, list(range(1, numChunks * 3, 3))] = df['pCoefficient']
        tabRes.iloc[vecIndex, list(range(2, numChunks * 3, 3))] = df['MPC']
        tabRes.iloc[vecIndex, -3] = df['XieBeni'].mean()
        tabRes.iloc[vecIndex, -2] = df['pCoefficient'].mean()
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
        with open(f'{output_path}/directOUTPUT.txt', 'a') as f:
            f.write(output)


    tabRes.to_excel(output_path / f"Example_Table_11K_08-04-1k-50F_{start}_{end}.xlsx")
# tabRes.to_excel("./output/XieBeni_sm" + str(simIDX) + ".xlsx")
with open(f'{output_path}/directOUTPUT.txt', 'a') as f:
    f.write("\n------------------------------------------------------------")
print("--- End of execution --- ")


def generate_best():
    import pandas as pd
    sm = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    t_sm = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    names = ["Comp1","Comp2","Comp3","Comp4","Comp5","Comp6","Comp7","Comp8","Comp9","Comp10",
             "Comp11","Comp12","Comp13","Comp14","Comp15","Comp16","Comp17","Comp18","Comp19","Comp20",
             "Comp21","Comp22","Comp23","Comp24","Comp25","Comp26","Comp27","Comp28","Comp29","Comp30","Comp31","Comp32"]
    threshList_40k = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
    threshList_11k = [0.9, 0.9, 0.25, 0.9, 0.5, 0.9, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.8, 0.5, 0.65, 0.65, 0.8, 0.9, 0.9, 0.8, 0.9, 0.25, 0.25, 0.25, 0.25, 0.25]
    em = [1999,3999,5999,7999,10999]
    df = pd.read_csv("data.csv")
    df.SimMeasure = df.SimMeasure.replace(sm, t_sm)
    df['Name'] = df['SimMeasure'].apply(lambda x: names[x-1])
    df_selected_th = df[df[['SimMeasure','Threshold']].apply(tuple, axis=1).isin(zip(t_sm, threshList_11k))]
    df_selected_th = df_selected_th[df_selected_th['EndChunk'].isin(em)]
    df_selected_th[["Name","XieBeni"]].groupby("Name").mean().sort_values(by="XieBeni")
    df_selected_th[["Name","XieBeni"]].groupby("Name").mean().sort_values(by="XieBeni").rank(method="dense", ascending=True)
    df_selected_th[["Name","pCoefficient"]].groupby("Name").mean().sort_values(by="pCoefficient")
    sel_40k = ["Comp1","Comp4","Comp5","Comp6","Comp7","Comp19","Comp20","Comp23","Comp24","Comp25","Comp26","Comp27"]
    sel_11k = ["Comp1","Comp3","Comp4","Comp7","Comp19","Comp21","Comp22","Comp23","Comp25","Comp26","Comp27"]
    df_selected_th[df_selected_th.Name.isin(sel_11k)].to_csv("best_sel_11k.csv")

def merge_files():
    dataset = "Benchmark1_11000" # "RBF1_40000"
    path = f"../output/{dataset}"
    path = f"output/{dataset}"
    data = []
    with open(f"{path}/files.csv", 'r') as f:
        for i, line in enumerate(f):
            fname = line.rstrip('\n')
            df = pd.read_excel(f"{path}/{fname}", nrows=1)
            SimMeasure = f"Comp{i+1}"
            df.insert(loc=0, column='SimMeasure', value=SimMeasure)
            data.append(df.iloc[0].tolist())
    df = pd.DataFrame(data)
    df = df.drop(1, axis=1)
    df = df.set_index(df.columns[0])

    df.to_excel("all.xlsx")

    nChunks = len(df.columns)/3 - 1
    nSel = 5
    step = nChunks/nSel
    names = ["XB","PC","MPC"]
    start = 0
    selected_cols1 = [3 * (i) * int(step) - 3 + 0 for i in range(1, nSel + 1)]
    selected_cols2 = [3 * (i) * int(step) - 3 + 1 for i in range(1, nSel + 1)]
    selected_cols3 = [3 * (i) * int(step) - 3 + 2 for i in range(1, nSel + 1)]
    selected_cols = sum(list(map(list, zip(selected_cols1, selected_cols2, selected_cols3))),[])
    selected_cols.extend([-3,-2,-1])

    c_names = sum([[f"{t}{int(i * step)}" for t in names] for i in range(1, int(nSel) + 2)] ,[])

    df = df.iloc[:, selected_cols]
    df = df.set_axis(c_names,axis=1)
    df.to_excel(f"{path}_processed.xlsx")