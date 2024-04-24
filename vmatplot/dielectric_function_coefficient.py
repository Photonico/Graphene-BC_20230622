#### Dielectric function coefficient calculation and plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, W0612

import numpy as np

from vmatplot.absorption_coefficient import cal_absorption_coefficient

# Speed of light in vacuum in meters per second
c_mps = 299792458
# Speed of light in vacuum nanometers per second
c_nm = c_mps * 1e9  # 1e9 is equivalent to 10^9

def cal_dielectric_function_coefficient(frequency,density_energy_real,density_energy_imag):
    absorption = cal_absorption_coefficient(frequency,density_energy_real,density_energy_imag)
    coe = np.sqrt(np.sqrt(2)*frequency*absorption)/c_nm
    return coe
