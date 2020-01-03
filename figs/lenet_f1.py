from pathlib import Path

from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


rc("font", family="serif")
rc("text", usetex=True)


DATA_DIR = Path("data")


if __name__ == "__main__":
    src_file = DATA_DIR / "lenet_f1_source.csv"
    trg_file = DATA_DIR / "lenet_f1_target.csv"

    epochs = list(range(1, 21))
    src_f1 = pd.read_csv(src_file)["Value"].values
    trg_f1 = pd.read_csv(trg_file)["Value"].values

    ax = plt.axes(xlabel="Epoch", ylabel="\(F_1\) score")
    ax.plot(epochs, src_f1, 'o-', label="source \(F_1\) score")
    ax.plot(epochs, trg_f1, 'o-', label="target \(F_1\) score")
    ax.set_xticks(epochs)
    ax.legend()

    plt.savefig("lenet_f1.pdf")
