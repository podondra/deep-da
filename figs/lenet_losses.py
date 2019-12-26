from pathlib import Path

from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


rc("font", family="serif")
rc("text", usetex=True)


DATA_DIR = Path("data") 


if __name__ == "__main__":
    overfit_va_file = DATA_DIR / "overfit_validation-loss.csv"
    overfit_tr_file = DATA_DIR / "overfit_training-loss.csv"
    va_file = DATA_DIR / "lenet_validation-loss.csv"
    tr_file = DATA_DIR / "lenet_training-loss.csv"

    overfit_va_df = pd.read_csv(overfit_va_file)
    epochs = overfit_va_df["Step"].values
    overfit_va_loss = overfit_va_df["Value"].values
    overfit_tr_loss = pd.read_csv(overfit_tr_file)["Value"].values

    va_loss = pd.read_csv(va_file)["Value"].values
    tr_loss = pd.read_csv(tr_file)["Value"].values

    ax = plt.axes(xlabel="Epoch", ylabel="Binary cross entropy loss")
    ax.plot(epochs, overfit_tr_loss, '.', label="Training loss")
    ax.plot(epochs, overfit_va_loss, '.', label="Validation loss")
    ax.plot(epochs, tr_loss, '.', label="Training loss with dropout")
    ax.plot(epochs, va_loss, '.', label="Validation loss with dropout")
    ax.set_xticks(epochs)
    ax.legend()

    plt.savefig("lenet_losses.pdf")
