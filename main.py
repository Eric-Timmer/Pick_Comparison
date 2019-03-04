import pandas as pd
from utilities import split_into_clusters
from process import process
import glob
import os

def main(file_list, interpreter1_csv_files):
    list_of_fms = list()
    curves = ["GR", "SP", "DT", "RHOB", "RESD"]
    for i in interpreter1_csv_files:
        try:
            log, fm = os.path.basename(i).split("_")
        except ValueError:  # throws errors because las_interval_extractor.py is still spitting out results
            continue
        fm = fm.split(".")[0]
        list_of_fms.append(fm)

    list_of_fms = list(set(list_of_fms))

    for f in list_of_fms:
        raw_pick_df = None
        for i in file_list:
            base = os.path.basename(i).split(".")[0]
            if f.lower() == base.split("_")[0].lower():
                raw_pick_df = pd.read_excel(i)
                break
        if raw_pick_df is None:
            print("OOOPS, pick df not found")
            continue
        n_clusters = int(len(raw_pick_df["UWI"].values) / 1500.)
        print("Separating data into clusters n = %i" % n_clusters)
        pick_df = split_into_clusters(f, raw_pick_df, n_clusters)
        n_clusters = len(set(pick_df["Clusters"].values))

        for c in curves:
            inter = 10
            raw_pick_df = process(f, c, pick_df, raw_pick_df, "AGS", "AER", n_clusters, inter, testing=False)

        raw_pick_df.to_excel("outlier_plots/clusters/final/%s.xlsx" % f)



if __name__ == "__main__":
    file_list = [""]
    interpreter1_csv_files = [""]
    main(file_list, interpreter1_csv_files)
