from pathlib import Path
import h5py
root = Path('/mnt/d/meta-robotics/Code/FingerForce_Decode/data')
paths = [root / 'processed_0528/session0_1/filtered_Middle_Flex.mat']
paths += list(root.glob('processed_*/session0_1/*Middle*Trapezoid*.mat'))
paths += list(root.glob('processed_*/session0_1/*Middle*Flex*.mat'))[:20]
for p in dict.fromkeys(paths):
    if h5py.is_hdf5(p):
        with h5py.File(p) as f:
            print(str(p), {key: getattr(f[key], 'shape', 'group') for key in f if not key.startswith('#')})
