import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qso.utils import init_weights, load_ds, predict


class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
        self.pool1 = nn.MaxPool1d(16, 16)
        self.conv2 = nn.Conv1d(32, 48, 5, padding=2)
        self.pool2 = nn.MaxPool1d(16, 16)
        self.fc1 = nn.Linear(48 * 14, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


ds_file = "data/dataset.hdf5"
bs = 64
bs_eval = 2048

src_ds = load_ds(ds_file, "sdss")
src_dl = DataLoader(src_ds, batch_size=bs, shuffle=True)
src_dl_eval = DataLoader(src_ds, batch_size=bs_eval)
src_dl_va = DataLoader(load_ds(ds_file, "sdss", va=True), batch_size=bs_eval)
trg_dl_va = DataLoader(load_ds(ds_file, "lamost", va=True), batch_size=bs_eval)

dev = torch.device("cuda")
model = LeNet_5().to(dev)
model.apply(init_weights)
criterion = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters())

writer = SummaryWriter("runs/lenet")

epochs = 20
for epoch in range(1, epochs + 1):
    for xb, yb in src_dl:
        xb, yb = xb.to(dev), yb.to(dev)
        loss = criterion(model(xb), yb.unsqueeze(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        src_trues, src_preds = predict(model, src_dl_eval, dev)
        src_trues_va, src_preds_va = predict(model, src_dl_va, dev)
        trg_trues_va, trg_preds_va = predict(model, trg_dl_va, dev)

    writer.add_scalars("loss", {"training": criterion(src_preds, src_trues),
                                "validation": criterion(src_preds_va, src_trues_va)}, epoch)
    writer.add_scalars("f1", {"source": f1_score(src_trues_va.bool(), src_preds_va > 0),
                              "target": f1_score(trg_trues_va.bool(), trg_preds_va > 0)}, epoch)

torch.save(model.state_dict(), "models/lenet-5.pt")
