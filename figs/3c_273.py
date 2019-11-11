from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np


rc("font", family="serif")
rc("text", usetex=True)

hdul = fits.open("data/spec-0519-52283-0021.fits")
data = hdul[1].data

xlabel = "Wavelength (\AA{})"
ylabel = "$f_{\lambda}$ (10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA{}$^{-1}$)"

ax = plt.axes(xlabel=xlabel, ylabel=ylabel)
ax.plot(np.power(10, data["loglam"]), data["flux"])
plt.savefig("3c_273.pdf")
