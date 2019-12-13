from os import path

from astropy.io import fits
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from qso import lamost


LAMOST_DIR = path.join("data", "lamost")
LAMOST_CATALOG = path.join(LAMOST_DIR, "dr5_v3.fits")
LAMOST_HDF5 = path.join(LAMOST_DIR, "lamost_dr5_v3.hdf5")


if __name__ == "__main__":
    catalog_hdul = fits.open(LAMOST_CATALOG)
    catalog = catalog_hdul[1].data

    multiindex = pd.MultiIndex.from_arrays(
        [
            catalog["planid"],
            catalog["lmjd"].astype("i4"),
            catalog["spid"],
            catalog["fiberid"],
        ],
        names=["planid", "lmjd", "spid", "fiberid"]
    )

    wavemin = np.zeros(multiindex.shape, dtype="f4")
    wavemax = np.zeros(multiindex.shape, dtype="f4")

    f = h5py.File(LAMOST_HDF5, "r")

    for i, index in enumerate(tqdm(multiindex)):
        wave = f[path.join(index[0], lamost.get_filename(*index), "wavelength")]
        wavemin[i] = wave[0]
        wavemax[i] = wave[-1]

    catalog_df = pd.DataFrame(
        data={
            "wavemax": wavemax,
            "wavemin": wavemin
            },
        index=multiindex
    )

    catalog_df.to_csv(path.join(LAMOST_DIR, "lamost_dr5_v3_coverage.csv"))
