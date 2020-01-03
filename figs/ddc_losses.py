from pathlib import Path

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


rc("font", family="serif")
rc("text", usetex=True)


DATA_DIR = Path("data")


if __name__ == "__main__":
    epochs = list(range(1, 21))

    tr_loss = pd.read_csv(DATA_DIR / "ddc_loss_training.csv")["Value"].values
    va_loss = pd.read_csv(DATA_DIR / "ddc_loss_validation.csv")["Value"].values

    ax = plt.axes(xlabel="Epoch", ylabel="Binary cross entropy loss")
    ax.plot(epochs, tr_loss, 'o-', label="Training loss")
    ax.plot(epochs, va_loss, 'o-', label="Validation loss")
    ax.set_xticks(epochs)
    ax.legend()

    plt.savefig("ddc_losses.pdf")

    mmd = np.sqrt(pd.read_csv(DATA_DIR / "ddc_loss_mmd.csv")["Value"].values)
    no_mmd = pd.read_csv(DATA_DIR / "ddc_no-mmd-loss_mmd.csv")["Value"].values

    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(epochs, no_mmd, 'o-', label="No MMD loss")
    axs[1].plot(epochs, mmd, 'o-', label="MMD loss")
    axs[1].set_xticks(epochs)
    for ax in axs:
        ax.legend()
        ax.set_ylabel("MMD")
    axs[1].set_xlabel("Epoch")

    plt.savefig("ddc_mmds.pdf")
