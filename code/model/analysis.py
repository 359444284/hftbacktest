import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import njit, prange


of_data = pd.read_csv('of_model_output_modi.csv', header=0, index_col=0,
                   names=['timestamp', 'signal1', 'mp', 'bp1', 'ap1', 'bv1', 'av1'])

row_data = pd.read_csv('model_output_modi.csv', header=0, index_col=0,
                       names=['timestamp', 'signal0', 'mp', 'bp1', 'ap1', 'bv1', 'av1'])
print(len(of_data), len(row_data))

combine = pd.concat([of_data, row_data], axis=1)
print(combine.head(1))
interval = 1000
begin = 10000
picture = combine[['signal1', 'signal0']].iloc[begin:begin + interval]
picture[picture['signal1'] == picture['signal0']] = 0
picture.plot(alpha=0.5)

plt.show()

print(np.corrcoef(combine['signal0'].values, combine['signal1'].values))