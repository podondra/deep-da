import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from qso.utils import N_WAVELENGTHS


if __name__ == "__main__":
    mat = h5py.File("data/design_matrices.hdf5", "r")
    f = h5py.File("data/lamost/lamost_dataset.hdf5", "x")

    lamost = mat["lamost"]
    X_all = lamost["X"]
    id_all = lamost["filename"][:]

    idx_all = np.arange(id_all.size)

    id_tr, id_te, idx_tr, idx_te = train_test_split(id_all, idx_all, test_size=0.1)
    id_tr, id_va, idx_tr, idx_va = train_test_split(id_tr, idx_tr, test_size=0.2)

    for name, data in [("id_tr", id_tr), ("id_va", id_va), ("id_te", id_te)]:
        f.create_dataset(name, data=data)

    X_tr = f.create_dataset("X_tr", shape=(id_tr.size, N_WAVELENGTHS), dtype=X_all.dtype)
    X_va = f.create_dataset("X_va", shape=(id_va.size, N_WAVELENGTHS), dtype=X_all.dtype)
    X_te = f.create_dataset("X_te", shape=(id_te.size, N_WAVELENGTHS), dtype=X_all.dtype)

    for i, idx in enumerate(tqdm(idx_te)):
        X_te[i] = minmax_scale(X_all[idx], feature_range=(-1, 1))

    for i, idx in enumerate(tqdm(idx_va)):
        X_va[i] = minmax_scale(X_all[idx], feature_range=(-1, 1))

    for i, idx in enumerate(tqdm(idx_tr)):
        X_tr[i] = minmax_scale(X_all[idx], feature_range=(-1, 1))

    mat.close()
    f.close()
