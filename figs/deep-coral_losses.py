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

    tr_loss = pd.read_csv(DATA_DIR / "deep-coral_loss_training.csv")["Value"].values
    va_loss = pd.read_csv(DATA_DIR / "deep-coral_loss_validation.csv")["Value"].values
    coral_loss = pd.read_csv(DATA_DIR / "deep-coral_loss_coral.csv")["Value"].values

    ax = plt.axes(xlabel="Epoch", ylabel="Binary cross entropy or CORAL loss")
    ax.plot(epochs, tr_loss, 'o-', label="Training loss")
    ax.plot(epochs, va_loss, 'o-', label="Validation loss")
    ax.plot(epochs, coral_loss, 'o-', label="CORAL loss")
    ax.set_xticks(epochs)
    ax.legend()

    plt.savefig("deep-coral_losses.pdf")
