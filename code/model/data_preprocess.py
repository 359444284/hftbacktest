import time

import numpy as np
import pandas as pd
from numba import njit, prange


# @njit(parallel=True)
@njit()
def filter(probs, buy_thro, sell_thro):
    for i in prange(len(probs)):
        if probs[i] > buy_thro[i]:
            probs[i] = 2
        elif probs[i] < sell_thro[i]:
            probs[i] = 1
        else:
            probs[i] = 0


if __name__ == '__main__':
    min_data_size = 2000

    data = pd.read_csv('ML_model_output.csv', header=0, index_col=0, names=['timestamp', 'signal'])
    probs = data['signal'].to_numpy()
    date = data['timestamp'].to_numpy()
    quantile = 0.1

    positive_filter = probs > 0
    negative_filter = probs < 0
    positive_prob = np.where(positive_filter, probs, np.nan)
    negative_prob = np.where(negative_filter, probs, np.nan)
    buy_throds = pd.DataFrame(positive_prob).rolling(min_data_size, min_periods=1).quantile(
        quantile).values
    sell_throds = pd.DataFrame(negative_prob).rolling(min_data_size, min_periods=1).quantile(
        1 - quantile).values

    # s = time.perf_counter()
    filter(probs, buy_throds, sell_throds)
    # e = time.perf_counter()
    # print(e-s)
    data['signal'] = probs
    data = data.iloc[min_data_size:]
    print(data.head(5))
    print(max(data['timestamp'].diff(1).fillna(500)))
    data.to_csv('model_output_modi.csv', index=False)

