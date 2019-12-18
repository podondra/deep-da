import numpy as np


WAVEMIN, WAVEMAX = 3.5843, 3.9501
N_WAVELENGTHS = 3659
WAVES = np.logspace(WAVEMIN, WAVEMAX, N_WAVELENGTHS)


def init_weights(m):
    if type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)
