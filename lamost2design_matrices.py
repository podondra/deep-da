from os import path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from qso import lamost
from qso.utils import WAVEMIN, WAVEMAX, N_WAVELENGTHS


EPS = 0.00005

LAMOST_DIR = path.join("data", "lamost")
LAMOST_HDF5 = path.join(LAMOST_DIR, "lamost_dr5_v3.hdf5")
LAMOST_COVERAGE = path.join(LAMOST_DIR, "lamost_dr5_v3_coverage.csv")
DESIGN_MATRICES_HDF5 = path.join("data", "design_matrices.hdf5")


if __name__ == "__main__":
    lamost_dr5_v3 = h5py.File(LAMOST_HDF5, "r")
    f = h5py.File(DESIGN_MATRICES_HDF5, "r+")
    catalog = pd.read_csv(
        LAMOST_COVERAGE,
        index_col=["planid", "lmjd", "spid", "fiberid"],
        dtype={"wavemax": "f4", "wavemin": "f4"}
    )

    grp = f.require_group("lamost")
    X = grp.require_dataset(
            "X",
            shape=(catalog.shape[0], N_WAVELENGTHS),
            dtype=np.float32
            )
    filename = grp.require_dataset(
            "filename",
            shape=(catalog.shape[0], ),
            dtype="S30"
            )

    for i, index in enumerate(tqdm(catalog.index)):
        name = lamost.get_filename(*index)
        filename[i] = np.string_(name)
        key = path.join("{}".format(index[0]), name)
        spec = lamost_dr5_v3[key]
        loglam = np.log10(spec["wavelength"][:])
        X[i] = spec["flux"][(WAVEMIN - EPS <= loglam) & (loglam <= WAVEMAX + EPS)]
