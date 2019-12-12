import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm


TEST_SIZE = 100000
VALIDATION_SIZE = 50000


def fill_design_matrix(idx, X, X_new):
    for j, i in enumerate(tqdm(idx)):
        flux = X[i]
        flux = minmax_scale(flux, feature_range=(-1, 1))
        X_new[j] = flux


if __name__ == "__main__":
    # load data
    f = h5py.File("data/design_matrices.hdf5", "r+")
    sdss = f["sdss"]
    X = sdss["X"]
    y = sdss["y"][:]
    filename = sdss["filename"][:]

    # create index
    idx = np.arange(len(X))

    idx_train, idx_test, \
            y_train, y_test, \
            filename_train, filename_test = train_test_split(
                    idx, y, filename,
                    test_size=TEST_SIZE,
                    shuffle=True,
                    stratify=y,
                    random_state=92
                    )

    idx_train, idx_validation, \
            y_train, y_validation, \
            filename_train, filename_validation = train_test_split(
                    idx_train, y_train, filename_train,
                    test_size=VALIDATION_SIZE,
                    shuffle=True,
                    stratify=y_train,
                    random_state=61
                    )

    train_size = len(idx_train)
    n_features = X.shape[1]

    # filenames split
    sdss.create_dataset("filename_train",      shape=(train_size, ),      dtype=filename.dtype, data=filename_train)
    sdss.create_dataset("filename_validation", shape=(VALIDATION_SIZE, ), dtype=filename.dtype, data=filename_validation)
    sdss.create_dataset("filename_test",       shape=(TEST_SIZE, ),       dtype=filename.dtype, data=filename_test)

    # labels split 
    sdss.create_dataset("y_train",      shape=(train_size, ),      dtype=y.dtype, data=y_train)
    sdss.create_dataset("y_validation", shape=(VALIDATION_SIZE, ), dtype=y.dtype, data=y_validation)
    sdss.create_dataset("y_test",       shape=(TEST_SIZE, ),       dtype=y.dtype, data=y_test)

    # data split
    X_train      = sdss.create_dataset("X_train",      shape=(train_size, n_features),      dtype=X.dtype)
    X_validation = sdss.create_dataset("X_validation", shape=(VALIDATION_SIZE, n_features), dtype=X.dtype)
    X_test       = sdss.create_dataset("X_test",       shape=(TEST_SIZE, n_features),       dtype=X.dtype)

    # fill data
    fill_design_matrix(idx_validation, X, X_validation)
    fill_design_matrix(idx_test,       X, X_test)
    fill_design_matrix(idx_train,      X, X_train)

    f.close()
