from load_data import load
from outliers import outlier_detection, plot_outliers_xy
from distance import compute_similarities
from utilities import merge_spreadsheets
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def process(fm, log, pick_df, raw_pick_df, interpreter1, interpreter2, n_clusters, interval, testing=False):
    print("Loading data Interpreter1")
    df = load(fm, log, interpreter1, interval)
    cols1 = dict()
    for i in df.columns.values:
        cols1[i] = i + "_%s" % interpreter1
    df.rename(inplace=True, columns=cols1)
    # rename headers

    print("Loading data Interpreter2")
    df2 = load(fm, log, interpreter2, interval)
    cols2 = dict()
    for i in df2.columns.values:
        cols2[i] = i + "_%s" % interpreter2
    df2.rename(inplace=True, columns=cols2)

    # combine df and df2
    print("Concatenating data")
    df = df.join(df2)

    # TESTING DATA
    if testing is True:
        print("Loading Testing data")
        testing_df = load(fm, log, interpreter1, interval, testing="top")
        testing_df = testing_df.sample(frac=0.005)
        cols_testing = dict()
        for i in testing_df.columns.values:
            cols_testing[i] = i + "_testing"
        testing_df.rename(inplace=True, columns=cols_testing)
        print("Concatenating testing data")
        df = df.join(testing_df)

    outlier_lists = list()
    cluster_outlier_df = pd.DataFrame()

    outlier_lists_lof = list()
    cluster_outlier_df_lof = pd.DataFrame()

    cluster_labels = list()
    cluster_labels_lof = list()

    for c in tqdm(range(0, n_clusters), desc="%s, %s, interval = %s Processing clusters" % (fm, log, str(interval))):
        cluster_labels.append("Cluster %i IF" % c)
        cluster_labels_lof.append("Cluster %i LOF" % c)

        list1 = list(pick_df["UWI"][pick_df["Clusters"] == c].values)
        interpreters = list(pick_df["INTERPRETER"][pick_df["Clusters"] == c].values)

        # Add interpreter tags to well names
        temp = list()
        for i, interp in enumerate(interpreters):
            interp = interpreters[i]
            if testing is True:
                if interp == interpreter2:
                    temp.append("%s_%s" % (list1[i], interpreter2))
                else:
                    temp.append("%s_%s" % (list1[i], interpreter1))
                temp.append("%s_%s" % (list1[i], "testing"))
                continue
            if interp == interpreter2:
                temp.append("%s_%s" % (list1[i], interpreter2))
            else:
                temp.append("%s_%s" % (list1[i], interpreter1))

        list1 = temp
        list2 = list(df.columns.values)

        # select subset containing both interpreters' wells within given cluster
        subset = list(set(list1).intersection(list2))

        sub_df = df[subset]
        wells = list(sub_df.columns.values)
        shape = (len(wells), len(wells))

        # COMPUTE DISTANCES
        distances = compute_similarities(sub_df.values, shape)

        # ISOLATION FOREST
        outlier_df = outlier_detection(distances, wells, lof=False)
        outlier_df["Cluster"] = c
        outlier_lists.append(outlier_df["Label"])

        if cluster_outlier_df.empty:
            cluster_outlier_df = outlier_df
        else:
            cluster_outlier_df = pd.concat((cluster_outlier_df, outlier_df), keys=("Well", "Label", "Cluster"),
                                           ignore_index=True)

        # LOCAL OUTLIER FACTOR SCORE
        outlier_df_lof = outlier_detection(distances, wells, lof=True)
        outlier_df_lof["Cluster"] = c
        outlier_lists_lof.append(outlier_df_lof["Label"])

        if cluster_outlier_df_lof.empty:
            cluster_outlier_df_lof = outlier_df_lof
        else:
            cluster_outlier_df_lof = pd.concat((cluster_outlier_df_lof, outlier_df_lof),
                                               keys=("Well", "Label", "Cluster"), ignore_index=True)

    print("Saving outliers list")
    cluster_outlier_df.to_csv("outlier_plots/clusters/labels/%s_%s_if.csv" % (fm, log), index=False)
    cluster_outlier_df_lof.to_csv("outlier_plots/clusters/labels/%s_%s_lof.csv" % (fm, log), index=False)

    print("Plotting Outliers Histogram")

    cmap = cm.get_cmap("jet")
    colors = list()
    for i in range(0, n_clusters):
        colors.append(cmap(i / float(n_clusters)))

    plt.hist(outlier_lists, label=cluster_labels, color=colors, histtype="step", stacked=True)
    plt.hist(outlier_lists_lof, label=cluster_labels_lof, color=colors, histtype="step", stacked=True)
    plt.legend(fontsize="xx-small", markerscale=.3)
    plt.ylabel("Count")
    n_outliers = len(cluster_outlier_df[cluster_outlier_df["Label"] == 1].index.values)
    n_inliers = len(cluster_outlier_df[cluster_outlier_df["Label"] == 0].index.values)
    n_outliers_lof = len(cluster_outlier_df_lof[cluster_outlier_df_lof["Label"] == 1].index.values)
    n_inliers_lof = len(cluster_outlier_df_lof[cluster_outlier_df_lof["Label"] == 0].index.values)
    plt.xticks([0, 1], ["Inliers (n= %i, %i)" % (n_inliers, n_inliers_lof),
                        "Outliers (n = %i, %i)" % (n_outliers, n_outliers_lof)])
    plt.title("Outlier Detection")
    plt.savefig("outlier_plots/clusters/hist_%s_%s_%s.pdf" % (fm, log, interval))
    plt.clf()

    print("Generating Outliers XY plot")
    plot_outliers_xy(cluster_outlier_df, pick_df, interpreter2, fm, log, interval, lof=False)
    plot_outliers_xy(cluster_outlier_df_lof, pick_df, interpreter2, fm, log, interval, lof=True)

    print("Merging outputs")
    raw_pick_df = merge_spreadsheets(raw_pick_df, cluster_outlier_df, log, lof=False, interval=interval,
                                     interpreter1=interpreter1, interpreter2=interpreter2)
    raw_pick_df = merge_spreadsheets(raw_pick_df, cluster_outlier_df_lof, log, lof=True, interval=interval,
                                     interpreter1=interpreter1, interpreter2=interpreter2)

    return raw_pick_df