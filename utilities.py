import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm
from load_data import load
import glob
import os
import numpy as np
from sklearn.cluster import KMeans



def merge_spreadsheets(pick_df, outlier_df, log, lof, interval, interpreter1, interpreter2):
    interval = str(interval)

    if lof is True:
        add = "lof"
    else:
        add = "if"

    interpreter_list = list()
    well_list = list()
    for i in outlier_df["Well"]:
        w, interp = i.split("_")
        interpreter_list.append(interp)
        well_list.append(w)

    try:
        _ = pick_df["INTERPRETER2"]
    except KeyError:
        interpreter_list2 = list()
        for i in pick_df["INTERPRETER"]:
            if i == interpreter2:
                interpreter_list2.append(i)
            else:
                interpreter_list2.append(interpreter1)
        pick_df["INTERPRETER2"] = interpreter_list2

    temp_df = pd.DataFrame()
    temp_df["UWI"] = well_list
    temp_df["INTERPRETER2"] = interpreter_list
    temp_df["Label"] = outlier_df["Label"].copy(deep=True)

    temp_df.rename(columns={"Label": "Label_%s_%s_%s" % (log, add, interval)}, inplace=True)
    merged_df = pd.merge(pick_df, temp_df, on=["UWI", "INTERPRETER2"], how="left")

    return merged_df


def split_into_clusters(fm, df, n_clusters):
    df = df[["UWI", "X", "Y", "INTERPRETER"]]
    x = df["X"]
    y = df["Y"]

    X = np.column_stack((x, y))
    model = KMeans(n_clusters=n_clusters, n_jobs=-1)
    model.fit(X)
    clusters = model.labels_
    df["Clusters"] = clusters
    n_clusters = len(set(clusters))
    cmap = cm.get_cmap("jet")

    for i in range(0, n_clusters):
        sub = df[df["Clusters"] == i]
        x = sub["X"]
        y = sub["Y"]
        plt.scatter(x, y, c=cmap(i / float(n_clusters)), label="Cluster %i" % i)
    plt.legend(fontsize="xx-small", markerscale=.3)
    plt.savefig("outlier_plots/clusters/%s_clusters.pdf" % fm)
    plt.clf()

    return df
