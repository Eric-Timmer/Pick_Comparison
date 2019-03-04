import pandas as pd
import numpy as np

def load(fm, log, interpreter, interval, testing="top"):
    amalgam = pd.read_csv("extracted_las_data/%s/%s_%s.csv" % (interpreter, log, fm), index_col=0)
    cols = list()
    for i in amalgam.columns.values:
        if "." in i:
            continue
        else:
            cols.append(i)
    amalgam = amalgam[cols]

    if interval == "full":
        amalgam = standardize(amalgam)

        return amalgam
    elif testing == "top":
        interval *= 10
        down_max = int(interval)
        up_max = 0
        if up_max < 0 or down_max > 500:
            print("ERROR, INTERVAL SELECTED IS TOO LARGE")
            exit()

        amalgam = amalgam.iloc[up_max:down_max]
        amalgam = standardize(amalgam)

        return amalgam
    elif testing == "base":
        interval *= 10
        down_max = 500
        up_max = int(500 - interval)
        if up_max < 0 or down_max > 500:
            print("ERROR, INTERVAL SELECTED IS TOO LARGE")
            exit()
        amalgam = amalgam.loc[up_max:down_max]
        amalgam = standardize(amalgam)

        return amalgam

    else:
        midpoint = 500 / 2.
        interval *= 10
        down_max = int(midpoint + interval / 2.)
        up_max = int(midpoint - interval / 2.)
        if up_max < 0 or down_max > 500:
            print("ERROR, INTERVAL SELECTED IS TOO LARGE")
            exit()
        amalgam = amalgam.loc[up_max:down_max]
        amalgam = standardize(amalgam)
        return amalgam

def standardize(df, by_column=True, min_max=True):
    if by_column is True:
        temp = pd.DataFrame()
        for col in df.columns.values:
            data = df[col]
            stand = (data - np.nanmean(data)) / np.nanstd(data)
            if min_max is True:
                stand = (stand - np.nanmin(stand)) / (np.nanmax(stand) - np.nanmin(stand))
            temp[col] = stand
        df = temp
    else:
        df = (df - df.mean()) / df.std()
    return df