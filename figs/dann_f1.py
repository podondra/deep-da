from pathlib import Path

from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd


rc("font", family="serif")
rc("text", usetex=True)


DATA_DIR = Path("data")


if __name__ == "__main__":
    gammas = [0.1, 0.3, 1, 3, 10]
    src_f1 = [0.9388, 0.931, 0.9095, 0.7069, 0.8473]
    trg_f1 = [0.2181, 0.1965, 0.1621, 0.0338, 0.07935]

    ax = plt.axes(xlabel="\(\gamma\)", ylabel="\(F_1\) score", xscale="log")
    ax.plot(gammas, src_f1, 'o', label="source \(F_1\) score")
    ax.plot(gammas, trg_f1, 'o', label="target \(F_1\) score")
    ax.set_xticks(gammas)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()

    plt.savefig("dann_f1.pdf")
