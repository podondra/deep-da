from itertools import count

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qso.utils import init_weights, load_ds, predict
from qso.models import DANN


ds_file = "data/dataset.hdf5"
bs = 64
bs_eval = 2048

src_ds = load_ds(ds_file, "sdss")
src_dl = DataLoader(src_ds, batch_size=bs, shuffle=True)
src_dl_eval = DataLoader(src_ds, batch_size=bs_eval)
src_dl_va = DataLoader(load_ds(ds_file, "sdss", va=True), batch_size=bs_eval)
trg_dl = DataLoader(load_ds(ds_file, "lamost"), batch_size=bs, shuffle=True)
trg_dl_va = DataLoader(load_ds(ds_file, "lamost", va=True), batch_size=bs_eval)

dev = torch.device("cuda")
model = DANN().to(dev)
dom_predictor = lambda x: model.domain_classifier(model.feature_extractor(x), None)
model.apply(init_weights)
dom_criterion = nn.BCEWithLogitsLoss()
pred_criterion = nn.BCEWithLogitsLoss()
lr0 = 0.01
opt = optim.SGD(model.parameters(), lr=lr0, momentum=0.9)

writer = SummaryWriter("runs/dann")

alpha = 10
beta = 0.75
gamma = 10
epochs = 20
iters = epochs * min(len(src_dl), len(trg_dl))
iterator = count(1)
for epoch in range(1, epochs + 1):
    for i, (src_xb, src_yb), (trg_xb, _) in zip(iterator, src_dl, trg_dl):        
        src_xb, src_yb = src_xb.to(dev), src_yb.to(dev)
        trg_xb = trg_xb.to(dev)
        xb = torch.cat([src_xb, trg_xb])
        dom_yb = torch.cat([torch.zeros(src_xb.size(0)), torch.ones(trg_xb.size(0))]).to(dev)

        # update hyperparameters
        p = (i - 1) / (iters - 1)
        lam = (2 / (1 + np.exp(-gamma * p))) - 1
        lr = lr0 / ((1 + alpha * p) ** beta)
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        fb = model.feature_extractor(xb)
        dom_pred = model.domain_classifier(fb, lam)
        pred = model.label_predictor(fb[:src_xb.size(0)])

        dom_loss = dom_criterion(dom_pred, dom_yb.unsqueeze(-1))
        pred_loss = pred_criterion(pred, src_yb.unsqueeze(-1))
        loss = pred_loss + dom_loss

        loss.backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        src_trues, src_preds = predict(model, src_dl_eval, dev)
        src_trues_va, src_preds_va = predict(model, src_dl_va, dev)
        trg_trues_va, trg_preds_va = predict(model, trg_dl_va, dev)
        _, src_doms = predict(dom_predictor, src_dl_va, dev)
        _, trg_doms = predict(dom_predictor, trg_dl_va, dev)

    # log statistics
    writer.add_scalars("loss", {"training": pred_criterion(src_preds, src_trues),
                                "validation": pred_criterion(src_preds_va, src_trues_va)}, epoch)
    writer.add_scalars("f1", {"source": f1_score(src_trues_va.bool(), src_preds_va > 0),
                              "target": f1_score(trg_trues_va.bool(), trg_preds_va > 0)}, epoch)
    doms_true = torch.cat([torch.zeros(src_doms.size(0), dtype=torch.bool),
                           torch.ones(trg_doms.size(0), dtype=torch.bool)])
    doms_pred = torch.cat([src_doms > 0, trg_doms > 0])
    writer.add_scalars("accuracy", {"both": accuracy_score(doms_true, doms_pred),
                                    "source": (src_doms < 0).float().mean().item(),
                                    "target": (trg_doms > 0).float().mean().item()}, epoch)

torch.save(model.state_dict(), "models/dann.pt")
