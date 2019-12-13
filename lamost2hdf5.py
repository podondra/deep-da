from os import path

from astropy.io import fits
import h5py
import numpy as np
from tqdm import tqdm

from qso import lamost


LAMOST_DIR = path.join("data", "lamost")
LAMOST_DR5 = path.join(LAMOST_DIR, "lamost_dr5_v3")
LAMOST_CATALOG = path.join(LAMOST_DIR, "dr5_v3.csv")
LAMOST_HDF5 = path.join(LAMOST_DIR, "lamost_dr5_v3.hdf5")

DT = np.float32


if __name__ == "__main__":
    catalog = lamost.read_general_catalog(LAMOST_CATALOG)
    with h5py.File(LAMOST_HDF5, "w-") as f:
        for _, spec in tqdm(catalog.iterrows(), total=len(catalog)):
            filename = lamost.get_spec_filename(spec)
            dr_path = lamost.get_spec_filepath(spec, filename)
            filepath = path.join(LAMOST_DR5, dr_path)
            with fits.open(filepath) as hdul:
                data = hdul[0].data
                planid_group = f.require_group(spec["planid"])
                g = planid_group.create_group(filename)
                g.create_dataset("flux", data=data[0], dtype=DT)
                #g.create_dataset("inverse", data=data[1], dtype=DT)
                g.create_dataset("wavelength", data=data[2], dtype=DT)
                #g.create_dataset("andmask", data=data[3], dtype=DT)
                #g.create_dataset("ormask", data=data[4], dtype=DT)
