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

    coral_loss = pd.read_csv(DATA_DIR / "deep-coral_no-coral-loss_coral.csv")["Value"].values

    ax = plt.axes(xlabel="Epoch", ylabel="Correlation alignment")
    ax.plot(epochs, coral_loss, 'o-')
    ax.set_xticks(epochs)

    plt.savefig("deep-coral_coral.pdf")
