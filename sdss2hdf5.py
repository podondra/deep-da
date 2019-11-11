from os import path

from astropy.io import fits
import h5py
import numpy as np
from tqdm import tqdm

from qso import sdss


SDSS_DIR = path.join("data", "sdss")
SDSS_DR14 = path.join(SDSS_DIR, "sdss_dr14")
SDSS_CATALOG = path.join(SDSS_DIR, "specObj-dr14.fits")
SDSS_HDF5 = path.join(SDSS_DIR, "sdss_dr14.hdf5")

FLOAT = np.float32
INT = np.int32


if __name__ == "__main__":
    with h5py.File(SDSS_HDF5, "w-") as f, \
            fits.open(SDSS_CATALOG) as catalog_hdul:
        for spec in tqdm(catalog_hdul[1].data):
            filename = sdss.get_spec_filename(spec)
            dr_path = sdss.get_spec_filepath(spec, filename)
            filepath = path.join(SDSS_DR14, dr_path)
            with fits.open(filepath) as hdul:
                data = hdul[1].data
                group = f.create_group(filename)
                group.create_dataset("flux", data=data["flux"], dtype=FLOAT)
                group.create_dataset("loglam", data=data["loglam"], dtype=FLOAT)
                group.create_dataset("and_mask", data=data["and_mask"], dtype=INT)
                group.create_dataset("or_mask", data=data["or_mask"], dtype=INT)
                group.create_dataset("ivar", data=data["ivar"], dtype=FLOAT)
                group.create_dataset("wdisp", data=data["wdisp"], dtype=FLOAT)
                group.create_dataset("sky", data=data["sky"], dtype=FLOAT)
                group.create_dataset("model", data=data["model"], dtype=FLOAT)
