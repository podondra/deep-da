from functools import partial

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qso.utils import init_weights, load_ds, predict


def loss_clf_coral(model, src_dl, trg_dl, dev):
    assert src_dl.batch_size == trg_dl.batch_size
    bs = src_dl.batch_size
    y_true = torch.zeros(src_dl.dataset.tensors[0].size(0))
    y_pred = torch.zeros_like(y_true)
    src_fs = torch.zeros(src_dl.dataset.tensors[0].size(0), 100)
    trg_fs = torch.zeros_like(src_fs)
    for i, ((src_xb, src_yb), (trg_xb, _)) in enumerate(zip(src_dl, trg_dl)):
        src_xb, src_yb, trg_xb = src_xb.to(dev), src_yb.to(dev), trg_xb.to(dev)
        start = i * bs
        end = start + bs
        src_fb = model.feature_extractor(src_xb)
        src_fs[start:end] = src_fb
        trg_fs[start:end] = model.feature_extractor(trg_xb)
        y_true[start:end] = src_yb
        y_pred[start:end] = model.predictor(src_fb).squeeze()
    loss_clf = criterion_clf(y_pred, y_true)
    loss_coral = coral(src_fs, trg_fs, None)
    return loss_clf, loss_coral


def cov(data, dev):
    n = data.size(0)
    cov = torch.ones((1, n)).to(dev) @ data
    return (data.t() @ data - (cov.t() @ cov) / n) / (n - 1)


def coral(source_data, target_data, dev):
    d = source_data.size(1)
    source_cov = cov(source_data, dev)
    target_cov = cov(target_data, dev)    
    return (source_cov - target_cov).pow(2).sum() / (4 * d * d)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
        self.pool1 = nn.MaxPool1d(16, 16)
        self.conv2 = nn.Conv1d(32, 48, 5, padding=2)
        self.pool2 = nn.MaxPool1d(16, 16)
        self.fc1 = nn.Linear(48 * 14, 100)
        self.fc2 = nn.Linear(100, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(100, 1)
    
    def forward(self, x):
        x = self.fc3(x)
        return x


class DeepCORAL(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.predictor = Predictor()
        
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.predictor(x)


ds_file = "data/dataset.hdf5"
bs = 128
bs_eval = 2048

src_dl = DataLoader(load_ds(ds_file, "sdss"), batch_size=bs, shuffle=True)
src_dl_va = DataLoader(load_ds(ds_file, "sdss", va=True), batch_size=bs_eval)

trg_dl = DataLoader(load_ds(ds_file, "lamost"), batch_size=bs, shuffle=True)
trg_dl_va = DataLoader(load_ds(ds_file, "lamost", va=True), batch_size=bs_eval)

lam =  0.0005
dev = torch.device("cuda")
model = DeepCORAL().to(dev)
model.apply(init_weights)
criterion_clf = nn.BCEWithLogitsLoss()
criterion_coral = partial(coral, dev=dev)
opt = optim.Adam(model.parameters())
writer = SummaryWriter("runs/deep-no_coral")

epochs = 20 
for epoch in range(1, epochs + 1):
    for (src_xb, src_yb), (trg_xb, _) in zip(src_dl, trg_dl):
        src_xb, src_yb = src_xb.to(dev), src_yb.to(dev)
        trg_xb = trg_xb.to(dev)

        src_fb = model.feature_extractor(src_xb)
        trg_fb = model.feature_extractor(trg_xb)
        pred = model.predictor(src_fb)
        loss_clf = criterion_clf(pred, src_yb.unsqueeze(-1))
        loss_coral = criterion_coral(src_fb, trg_fb)

        loss = loss_clf + lam * loss_coral

        loss.backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        loss_clf, loss_coral = loss_clf_coral(model, src_dl, trg_dl, dev)
        src_trues_va, src_preds_va = predict(model, src_dl_va, dev)
        trg_trues_va, trg_preds_va = predict(model, trg_dl_va, dev)

    # log statistics
    writer.add_scalars("loss", {"training": loss_clf,
                                "coral": loss_coral,
                                "validation": criterion_clf(src_preds_va, src_trues_va)}, epoch)
    writer.add_scalars("f1", {"source": f1_score(src_trues_va.bool(), src_preds_va > 0),
                              "target": f1_score(trg_trues_va.bool(), trg_preds_va > 0)}, epoch)

torch.save(model.state_dict(), "models/deep-coral.pt")
