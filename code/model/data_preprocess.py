import os
import time

import numpy as np
import pandas as pd
from numba import njit, prange


# @njit(parallel=True)
@njit()
def filter(probs, buy_thro, sell_thro):
    for i in prange(len(probs)):
        if probs[i] > buy_thro[i]:
            probs[i] = 1
        elif probs[i] < sell_thro[i]:
            probs[i] = -1
        else:
            probs[i] = 0

def data_preprocess(file_name, min_data_size=2000, quantile=0.1):
    min_data_size = min_data_size

    # data = pd.read_csv('of_ML_model_output.csv', header=0, index_col=0, names=['timestamp', 'signal', 'mp', 'bp1', 'ap1', 'bv1', 'av1'])
    data = pd.read_csv(file_name, header=0, index_col=0,
                       names=['timestamp', 'signal', 'mp', 'bp1', 'ap1', 'bv1', 'av1'])
    # data = pd.read_csv('model_output.csv', header=0, index_col=0, names=['timestamp', 'signal', 'mp', 'bp1', 'ap1', 'bv1', 'av1'])
    probs = data['signal'].to_numpy()

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
    data.to_csv('/home/biden/PycharmProjects/hftbacktest/code/model/model_output_modi.csv', index=False)


if __name__ == '__main__':
    data_preprocess()

