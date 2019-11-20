from os import path

import h5py
import numpy as np
from tqdm import tqdm

N_SPEC = 9026365

LAMOST_DIR = path.join("data", "lamost")
LAMOST_HDF5 = path.join(LAMOST_DIR, "lamost_dr5_v3.hdf5")


if __name__ == "__main__":
    f = h5py.File(LAMOST_HDF5, "r")

    n_waves = np.zeros(N_SPEC, dtype=np.int)

    spec_gen = (spec for planid in f.values() for spec in planid.values())
    for idx, spec in enumerate(tqdm(spec_gen, total=N_SPEC)):
        n_waves[idx] = spec["wavelength"].shape[0]

    print(np.unique(n_waves, return_counts=True))
    print(n_waves.mean())
