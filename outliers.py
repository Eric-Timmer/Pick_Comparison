from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def outlier_detection(distances, wells, lof=True):
    if lof is True:
        n_neighbors = 100
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto", n_jobs=-1)
        try:
            labels = model.fit_predict(distances)
            labels = list(labels.flatten())
        except ValueError:
            labels = np.NaN
    else:
        model = IsolationForest(behaviour="new", bootstrap=True, n_jobs=-1, verbose=1, contamination="auto")
        try:
            model.fit(distances)
            labels = model.predict(distances)
            labels = list(labels.flatten())
        except ValueError:
            labels = np.NaN

    df = pd.DataFrame()
    df["Well"] = wells
    df["Label"] = labels

    # make outliers = 1, inliers = 0
    df["Label"].replace(1, 0, inplace=True)
    df["Label"].replace(-1, 1, inplace=True)
    return df


def plot_outliers_xy(labels, df, interpreter2, fm, log, interval, lof=False):
    """TODO not very clean code, fix."""
    df1 = df[df["INTERPRETER"] != interpreter2]
    df1 = df1[["UWI", "X", "Y"]]
    cols = ["gray", "red"]

    interpreter1_inlier_x = list()
    interpreter1_outlier_x = list()
    interpreter2_inlier_x = list()
    interpreter2_outlier_x = list()
    interpreter1_inlier_y = list()
    interpreter1_outlier_y = list()
    interpreter2_inlier_y = list()
    interpreter2_outlier_y = list()

    for i, w in enumerate(list(labels["Well"].values)):
        w = w.split("_")[0]
        lab = labels["Label"].iloc[i]
        current = df1[df1["UWI"] == w]
        try:
            x = current["X"].values[0]
            y = current["Y"].values[0]
        except IndexError:
            continue
        if lab == 0:
            interpreter1_inlier_x.append(x)
            interpreter1_inlier_y.append(y)
        else:
            interpreter1_outlier_x.append(x)
            interpreter1_outlier_y.append(y)

    df2 = df[df["INTERPRETER"] == interpreter2]
    df2 = df2[["UWI", "X", "Y"]]
    for i, w in enumerate(list(labels["Well"].values)):
        w = w.split("_")[0]
        lab = labels["Label"].iloc[i]
        current = df2[df2["UWI"] == w]
        try:
            x = current["X"].values[0]
            y = current["Y"].values[0]
        except IndexError:
            continue
        if lab == 0:
            interpreter2_inlier_x.append(x)
            interpreter2_inlier_y.append(y)
        else:
            interpreter2_outlier_x.append(x)
            interpreter2_outlier_y.append(y)

    plt.scatter(interpreter1_inlier_x, interpreter1_inlier_y, c=cols[0], alpha=0.3, marker="^", label="A inlier")
    plt.scatter(interpreter2_inlier_x, interpreter2_inlier_y, c=cols[0], alpha=0.3, marker="+", label="B inlier")
    plt.scatter(interpreter1_outlier_x, interpreter1_outlier_y, c=cols[1], alpha=0.8, marker="^", label="A outlier")
    plt.scatter(interpreter2_outlier_x, interpreter2_outlier_y, c=cols[1], alpha=0.8, marker="+", label="B outlier")
    plt.legend(fontsize="xx-small")
    if lof is True:
        plt.savefig("outlier_plots/clusters/%s_%s_%s_lof.pdf" % (fm, log, str(interval)))
    else:
        plt.savefig("outlier_plots/clusters/%s_%s_%s_if.pdf" % (fm, log, str(interval)))
    plt.clf()