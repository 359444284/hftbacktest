import gzip

import numpy as np

from hftbacktest import validate_data
from hftbacktest.data.utils import binancefutures, create_last_snapshot
import os
data_dict = './row_data/'
out_dict = './modified_data/'
data_files = sorted(os.listdir(data_dict))[1:]

# for i in range(0, len(data_files)):
#     input_name = data_files[i]
#     output_name = input_name.split('.')[0]
#     binancefutures.convert(data_dict + input_name, output_filename=out_dict + output_name)
#
data = create_last_snapshot(out_dict + 'btcusdt_20230611.npz', tick_size=0.01, lot_size=0.001)
np.savez(out_dict + 'btcusdt_20230611_eod.npz', data=data)

# Build 20230405 End of Day snapshot.
# Due to the file size limitation, btcusdt_20230405.npz does not contain data for the entire day.
create_last_snapshot(
    out_dict + 'btcusdt_20230612.npz',
    tick_size=0.1,
    lot_size=0.001,
    initial_snapshot=out_dict + 'btcusdt_20230611_eod.npz',
    output_snapshot_filename=out_dict + 'btcusdt_20230612_eod'
)

create_last_snapshot(
    out_dict + 'btcusdt_20230613.npz',
    tick_size=0.1,
    lot_size=0.001,
    initial_snapshot=out_dict + 'btcusdt_20230612_eod.npz',
    output_snapshot_filename=out_dict + 'btcusdt_20230613_eod'
)
