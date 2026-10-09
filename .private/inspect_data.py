from pathlib import Path
import h5py
from scipy.io import loadmat
p = Path('/mnt/d/meta-robotics/Code/FingerForce_Decode/data/processed_0528/session0_1/filtered_Middle_Flex.mat')
if h5py.is_hdf5(p):
    with h5py.File(p) as f:
        for key in ('emg_clean', 'good_channels', 'fs', 'fsamp', 'sampling_rate', 'filter_params'):
            if key in f:
                value = f[key]
                print(key, type(value).__name__, getattr(value, 'shape', None))
                if isinstance(value, h5py.Dataset) and value.size < 300:
                    print(value[...].reshape(-1).tolist())
else:
    data = loadmat(p)
    for key in ('emg_clean', 'good_channels', 'fs', 'fsamp', 'sampling_rate', 'filter_params'):
        if key in data:
            print(key, data[key].shape)
            if data[key].size < 300:
                print(data[key])
