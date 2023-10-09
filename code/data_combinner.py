import numpy as np
import pandas as pd
import os
from hftbacktest import merge_on_local_timestamp, validate_data

data_dict = '../hftbacktest/data/modified_data/'
data_files = os.listdir(data_dict)
data_files = [file for file in data_files if len(file)<=20]
data_files.sort()
data_files = data_files[2:]
print(data_files)

out_dict = '../hftbacktest/data/customer_data/'
os.makedirs(out_dict, exist_ok=True)

predicted_info = pd.read_csv('model/model_output_modi.csv', header=0)
predicted_info = predicted_info.astype(int)
predicted_info['timestamp'] = predicted_info['timestamp'] * 1000  # Convert timestamp column to datetime if it's not already
# Group the DataFrame by daily intervals

count = 0
for file_name in data_files:
    feed_data = np.load(data_dict + file_name)['data']
    begin_data, end_data = feed_data[0][2], feed_data[-1][2]
    print(begin_data, end_data)
    tmp_signal = predicted_info[(predicted_info['timestamp'] >= begin_data) & (predicted_info['timestamp'] <= end_data)]
    tmp = np.full((len(tmp_signal), 6), np.nan, np.float64)
    tmp[:, 0] = 110
    tmp[:, 1] = -1
    tmp[:, 2] = tmp_signal['timestamp'].astype(int)
    tmp[:, 3] = 0
    tmp[:, 4] = tmp_signal['signal']
    tmp[:, 5] = 0
    # 1686614438934000
    print(len(tmp_signal))
    print(tmp[0], tmp[-1])
    print(feed_data[0], feed_data[-1])
    merged = merge_on_local_timestamp(feed_data, tmp)

    validate_data(merged)

    np.savez(out_dict + file_name, data=merged)
