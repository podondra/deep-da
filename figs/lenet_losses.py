from pathlib import Path

from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


rc("font", family="serif")
rc("text", usetex=True)


DATA_DIR = Path("data") 


if __name__ == "__main__":
    va_file = DATA_DIR / "lenet_loss_validation.csv"
    tr_file = DATA_DIR / "lenet_loss_training.csv"

    epochs = list(range(1, 21))
    va_loss = pd.read_csv(va_file)["Value"].values
    tr_loss = pd.read_csv(tr_file)["Value"].values

    ax = plt.axes(xlabel="Epoch", ylabel="Binary cross entropy loss")
    ax.plot(epochs, tr_loss, 'o-', label="Training loss")
    ax.plot(epochs, va_loss, 'o-', label="Validation loss")
    ax.set_xticks(epochs)
    ax.legend()

    plt.savefig("lenet_losses.pdf")
