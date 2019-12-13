from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    def __init__(self, X, y, load=True):
        self.X = X
        if load:
            self.X = X[...]
        self.y = y[:]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx].reshape(1, -1), self.y[idx].astype("f4")
