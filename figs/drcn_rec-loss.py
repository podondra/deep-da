from pathlib import Path

from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


rc("font", family="serif")
rc("text", usetex=True)


DATA_DIR = Path("data") 


if __name__ == "__main__":
    epochs = list(range(1, 21))
    loss = pd.read_csv(DATA_DIR / "drcn_reconstruction_validation_loss.csv")["Value"].values

    ax = plt.axes(xlabel="Epoch", ylabel="Mean squared error loss")
    ax.plot(epochs, loss, 'o-', label="Validation reconstruction loss")
    ax.set_xticks(epochs)
    ax.legend()

    plt.savefig("drcn_rec-loss.pdf")