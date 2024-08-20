#### linear_optical_properties
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, W0612

import numpy as np

from vmatplot.dielectric_function import dielectric_systems_list

## Constants
c_ms = 2.99792458e8     # Speed of light in vacuum in meters per second
c_nm = c_ms * 1e9       # Speed of light in vacuum nanometers per second
hbar = 4.135667662e-15

## Theoretical formulas
# 1 absorption coefficient
def comp_absorption_coefficient(frequency,density_energy_real,density_energy_imag):
    coe = (np.sqrt(2)*frequency/c_nm)*(np.sqrt(np.sqrt(np.square(density_energy_real)+np.square(density_energy_imag))-density_energy_real))
    return coe

# 2 refractive index
def com_refractive(density_energy_real,density_energy_imag):
    index = np.sqrt((np.sqrt(np.square(density_energy_real)+np.square(density_energy_imag))+density_energy_real)/2)
    return index

# 3 extinction coefficient
def com_extinction(density_energy_real,density_energy_imag):
    coe = np.sqrt((np.sqrt(np.square(density_energy_real)+np.square(density_energy_imag))-density_energy_imag)/2)
    return coe

# 4 reflectivity
def com_reflectivity(density_energy_real,density_energy_imag):
    n = com_refractive(density_energy_real,density_energy_imag)
    k = com_extinction(density_energy_real,density_energy_imag)
    R = (np.square(n-1)+np.square(k))/(np.square(n+1)+np.square(k))
    return R

# 5 energy loss spectrum
def com_energy_loss_spectrum(density_energy_real,density_energy_imag):
    spectrum = (density_energy_imag)/(np.square(density_energy_real)+np.square(density_energy_imag))
    return spectrum

## Plotting
# systems list
def LOP_create_matters(*args):
    # data = dielectric_function_list(systems)
    # data[0] = current curve label
    # data[1] = dielectric function data
    # data[2] = color family
    # data[3] = linestyle
    # data[4] = alpha
    # data[5] = linewidth
    return dielectric_systems_list(*args)

# plot absorption coefficient

def plot_linear_optics(suptitle, formula=None, systems=None, components=None, comp_aliases=None,
                       layout=None, unit=None, boundary_1=(None, None), boundary_2=(None, None),
                       figure_size=(None,None)):

    return 0
