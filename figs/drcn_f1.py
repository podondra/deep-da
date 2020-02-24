from pathlib import Path

from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


rc("font", family="serif")
rc("text", usetex=True)


DATA_DIR = Path("data")


if __name__ == "__main__":
    epochs = list(range(1, 21))
    lenet_f1 = pd.read_csv(DATA_DIR / "lenet_f1_target.csv")["Value"].values
    drcn_f1 = pd.read_csv(DATA_DIR / "drcn_f1_target.csv")["Value"].values

    ax = plt.axes(xlabel="Epoch", ylabel="\(F_1\) score")
    ax.plot(epochs, lenet_f1, 'o-', label="target \(F_1\) score of our LeNet-5")
    ax.plot(epochs, drcn_f1, 'o-', label="target \(F_1\) score of DRCN")
    ax.set_xticks(epochs)
    ax.legend()

    plt.savefig("drcn_f1.pdf")
