from itertools import count

import h5py
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qso.dataset import HDF5Dataset, SDSS_DATASET
from qso.lenet import LeNet_5


def train(model, device, tr_loader, optimizer, writer, iterator):
    model.train()
    for data, target in tr_loader:
        data, target = data.to(device), target.view(-1, 1).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss/training", loss, next(iterator))


def validate(model, device, va_loader, writer, epoch):
    model.eval()
    va_loss = 0
    with torch.no_grad():
        for data, target in va_loader:
            data, target = data.to(device), target.view(-1, 1).to(device)
            output = model(data)
            va_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()
    va_loss /= len(va_loader.dataset)
    writer.add_scalar("loss/validation", va_loss, epoch)


def init_weights(m):
    if type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    #sdss_grp = h5py.File("data/design_matrices.hdf5", 'r', libver='latest', swmr=True)["sdss"]
    sdss = h5py.File(SDSS_DATASET, 'r')

    tr_set = HDF5Dataset(sdss["X_tr"], sdss["y_tr"])
    va_set = HDF5Dataset(sdss["X_va"], sdss["y_va"])
    tr_loader = DataLoader(tr_set, batch_size=64, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=1000)

    writer = SummaryWriter("runs/lenet_50")
    device = torch.device("cuda")
    lenet = LeNet_5().to(device)
    lenet.apply(init_weights)
    optimizer = optim.Adam(lenet.parameters())

    epochs = range(1, 51)
    iterator = count(1)
    for epoch in epochs:
        train(lenet, device, tr_loader, optimizer, writer, iterator)
        validate(lenet, device, va_loader, writer, epoch)
        torch.save(lenet.state_dict(), "lenet.pt")
