#### Refractive index calculation and plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, W0612

import numpy as np

def com_refractive(density_energy_real,density_energy_imag):
    index = np.sqrt((np.sqrt(np.square(density_energy_real)+np.square(density_energy_imag))+density_energy_real)/2)
    return index
