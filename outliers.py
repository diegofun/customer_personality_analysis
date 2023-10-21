import numpy as np
import pandas as pd


def detect_outliers(d, data):
    out = []
    for i in d:
        Q3, Q1 = np.percentile(data[i], [75, 25])
        IQR = Q3 - Q1

        ul = Q3 + 1.5 * IQR
        ll = Q1 - 1.5 * IQR

        # outliers dataframe is the same as 'data', when values meet conditions, including values and indexes
        outliers = data[i][(data[i] > ul) | (data[i] < ll)]
        out.append(outliers)
        print(f'*** {i} outlier points***', '\n', outliers, '\n')

    return out
