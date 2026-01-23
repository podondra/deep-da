import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qso.utils import init_weights, load_ds, predict, WAVES


def reconstruct(model, dl, dev):
    bs = dl.batch_size
    trues = torch.zeros(dl.dataset.tensors[0].size())
    preds = torch.zeros_like(trues)
    for i, (xb, _) in enumerate(dl):
        start = i * bs
        end = start + bs
        trues[start:end] = xb
        preds[start:end] = model(xb.to(dev))
    return trues, preds


def plot_rnd_rec(X_true, X_pred):
    rnd_idx = np.random.randint(X_true.size(0))
    fig, ax = plt.subplots()
    ax.plot(X_true[rnd_idx].squeeze())
    ax.plot(X_pred[rnd_idx].squeeze())
    return fig


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
        self.pool1 = nn.MaxPool1d(16, 16)
        self.conv2 = nn.Conv1d(32, 48, 5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv1d(48, 32, 5, padding=2)
        self.upsample1 = nn.Upsample(size=3659)
        self.conv4 = nn.Conv1d(32, 1, 5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv3(x))
        x = self.upsample1(x)
        x = F.hardtanh(self.conv4(x))
        return x


class LabelPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool2 = nn.MaxPool1d(16, 16)
        self.fc1 = nn.Linear(48 * 14, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DRCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.label_predictor = LabelPredictor()

    def forward(self, x):
        x = self.encoder(x)
        x = self.label_predictor(x)
        return x


ds_file = "data/dataset.hdf5"
bs = 128
bs_eval = 2048

src_ds = load_ds(ds_file, "sdss")
src_dl = DataLoader(src_ds, batch_size=bs, shuffle=True)
src_dl_eval = DataLoader(src_ds, batch_size=bs_eval)
src_dl_va = DataLoader(load_ds(ds_file, "sdss", va=True), batch_size=bs_eval)
trg_dl = DataLoader(load_ds(ds_file, "lamost"), batch_size=bs, shuffle=True)
trg_dl_va = DataLoader(load_ds(ds_file, "lamost", va=True), batch_size=bs_eval)

dev = torch.device("cuda")
model = DRCN().to(dev)
model.apply(init_weights)
convae = lambda x: model.decoder(model.encoder(x))
criterion_rec = nn.MSELoss()
criterion_clf = nn.BCEWithLogitsLoss()
opt_rec = optim.Adam([{"params": model.encoder.parameters()}, {"params": model.decoder.parameters()}])
opt_clf = optim.Adam([{"params": model.encoder.parameters()}, {"params": model.label_predictor.parameters()}])

writer = SummaryWriter("runs/drcn")

# typically, the optimal value was in the range [0.4, 0.7]
lam = 0.5

epochs = 20
for epoch in range(1, epochs + 1):
    for xb, _ in trg_dl:
        xb = xb.to(dev)
        loss = (1 - lam) * criterion_rec(convae(xb), xb)
        loss.backward()
        opt_rec.step()
        opt_rec.zero_grad()

    with torch.no_grad():
        true_recs_tr, pred_recs_tr = reconstruct(convae, trg_dl, dev)
        true_recs_va, pred_recs_va = reconstruct(convae, trg_dl_va, dev)

    writer.add_figure("training", plot_rnd_rec(true_recs_tr, pred_recs_tr), epoch)
    writer.add_figure("validation", plot_rnd_rec(true_recs_va, pred_recs_va), epoch)

    writer.add_scalars(
        "reconstruction loss",
        {"training": criterion_rec(pred_recs_tr, true_recs_tr),
         "validation": criterion_rec(pred_recs_va, true_recs_va)},
        epoch
    )

    for xb, yb in src_dl:
        xb, yb = xb.to(dev), yb.to(dev)
        loss = lam * criterion_clf(model(xb), yb.unsqueeze(-1))
        loss.backward()
        opt_clf.step()
        opt_clf.zero_grad()

    with torch.no_grad():
        src_trues, src_preds = predict(model, src_dl_eval, dev)
        src_trues_va, src_preds_va = predict(model, src_dl_va, dev)
        trg_trues_va, trg_preds_va = predict(model, trg_dl_va, dev)

    # log statistics
    writer.add_scalars("loss", {"training": criterion_clf(src_preds, src_trues),
                                "validation": criterion_clf(src_preds_va, src_trues_va)}, epoch)
    writer.add_scalars("f1", {"source": f1_score(src_trues_va.bool(), src_preds_va > 0),
                              "target": f1_score(trg_trues_va.bool(), trg_preds_va > 0)}, epoch)

torch.save(model.state_dict(), "drcn.pt")
