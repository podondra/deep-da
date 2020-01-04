from pathlib import Path

from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


rc("font", family="serif")
rc("text", usetex=True)


DATA_DIR = Path("data") 


if __name__ == "__main__":
    gammas = [0.1, 0.3, 1, 3, 10]
    files = [
            "dann_loss_training_01.csv",
            "dann_loss_training_03.csv",
            "dann_loss_training_1.csv",
            "dann_loss_training_3.csv",
            "dann_loss_training_10.csv"
            ]

    epochs = list(range(1, 21))
    ax = plt.axes(xlabel="Epoch", ylabel="Binary cross entropy loss")

    for gamma, csv in zip(gammas, files):
        loss = pd.read_csv(str(DATA_DIR / csv))["Value"].values
        ax.plot(epochs, loss, 'o-', label="\(\gamma = {}\)".format(gamma))

    ax.set_xticks(epochs)
    ax.legend()

    plt.savefig("dann_losses.pdf")
