from os import path

import h5py
import numpy as np
from tqdm import tqdm

from qso import sdss
from qso.utils import WAVEMIN, WAVEMAX, N_WAVELENGTHS


SDSS_DIR = path.join("data", "sdss")
SDSS_HDF5 = path.join(SDSS_DIR, "sdss_dr14.hdf5")
SDSS_SELECTED_CATALOG = path.join(SDSS_DIR, "sdss_dr14_selected.csv")
DESIGN_MATRICES_HDF5 = path.join("data", "design_matrices.hdf5")


if __name__ == "__main__":
    sdss_dr14 = h5py.File(SDSS_HDF5, "r")
    f = h5py.File(DESIGN_MATRICES_HDF5, "w-")

    catalog = sdss.read_selected_catalog(SDSS_SELECTED_CATALOG)
    n_spec = catalog.shape[0]

    grp = f.create_group("sdss")
    X = grp.require_dataset(
            "X", shape=(n_spec, N_WAVELENGTHS), dtype=np.float32
            )
    y = grp.require_dataset(
            "y", shape=(n_spec, ), dtype=np.bool
            )
    filename = grp.require_dataset(
            "filename", shape=(n_spec, ), dtype="S26"
            )

    progress_bar = tqdm(catalog.iterrows(), total=catalog.shape[0])
    for i, (index, series) in enumerate(progress_bar):
        name = sdss.get_filename(*index)
        filename[i] = np.string_(name)
        key = path.join("{:04d}".format(index[0]), name)
        spec = sdss_dr14[key]
        loglam = spec["loglam"][:]
        X[i] = spec["flux"][(WAVEMIN <= loglam) & (loglam <= WAVEMAX)]
        y[i] = series["qso"]
