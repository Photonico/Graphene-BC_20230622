#### linear_optical_properties
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, W0612

import numpy as np

from vmatplot.dielectric_function import dielectric_systems_list

## References

# <https://vaspkit.com/tutorials.html#linear-optical-properties>

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
def comp_refractive_index(density_energy_real,density_energy_imag):
    index = np.sqrt((np.sqrt(np.square(density_energy_real)+np.square(density_energy_imag))+density_energy_real)/2)
    return index

# 3 extinction coefficient
def comp_extinction_coefficient(density_energy_real,density_energy_imag):
    coe = np.sqrt((np.sqrt(np.square(density_energy_real)+np.square(density_energy_imag))-density_energy_imag)/2)
    return coe

# 4 reflectivity
def comp_reflectivity(density_energy_real,density_energy_imag):
    n = comp_refractive_index(density_energy_real,density_energy_imag)
    k = comp_extinction_coefficient(density_energy_real,density_energy_imag)
    R = (np.square(n-1)+np.square(k))/(np.square(n+1)+np.square(k))
    return R

# 5 energy loss spectrum
def comp_energy_loss_spectrum(density_energy_real,density_energy_imag):
    spectrum = (density_energy_imag)/(np.square(density_energy_real)+np.square(density_energy_imag))
    return spectrum

## Plotting
# systems list
def lop_systems(*args):
    return dielectric_systems_list(*args)

def identify_linear_optical_functions(incoming=None):
    help_info = "Please use one of the following terminologies as a string-type variable:\n" + \
                "\t absorption coefficient, refractive index, extinction coefficient, reflectivity, energy-loss\n"
    linear_flag, linear_title, compfunc_name, plotfunc_name = None, None, None, None
    if incoming.lower() in ["absorption coefficient","absorption"]:
        linear_flag = "absorption"
        linear_title = "Absorption coefficient"
        compfunc_name = "comp_absorption_coefficient"
        plotfunc_name = "plot_absorption_coefficient"
    elif incoming.lower() in ["refractive index","refractive"]:
        linear_flag = "refractive"
        linear_title = "Refractive"
        compfunc_name = "comp_refractive_index"
        plotfunc_name = "plot_refractive_index"
    elif incoming.lower() in ["extinction coefficient", "extinction"]:
        linear_flag = "extinction"
        linear_title = "Extinction coefficient"
        compfunc_name = "comp_extinction_coefficient"
        plotfunc_name = "plot_extinction_coefficient"
    elif incoming.lower() in ["reflectivity"]:
        linear_flag = "reflectivity"
        linear_title = "Reflectivity"
        compfunc_name = "comp_reflectivity"
        plotfunc_name = "plot_reflectivity"
    elif incoming.lower() in ["energy-loss function", "energy-loss spectrum", "energy-loss"]:
        linear_flag = "energy-loss"
        linear_title = "Energy-loss spectrum"
        compfunc_name = "comp_energy_loss_spectrum"
        plotfunc_name = "plot_energy_loss_spectrum"
    else:
        print(help_info)
        return None
    return {"flag":linear_flag, "title":linear_title, "calculation function":compfunc_name, "plotting function": plotfunc_name}

def lop_plotting_help(linear_chars):
    func_label = identify_linear_optical_functions(linear_chars)
    if func_label is not None:
        help_info = f"Usage: {func_label['plotting function']} \n" +\
                    f"Demonstrate {(func_label['title']).lower()} by each component \n" +\
                    "\t suptitle: the suptitle; \n" +\
                    "\t systems: dielectric function data list; \n" +\
                    "\t components: planes ('xx'<default>, 'yy', 'zz', 'xy', 'yx', 'yz', 'zy', 'zx', 'xz'); \n" +\
                    "\t layout: subfigures layout (horizontal<default>, vertical); \n" +\
                    "\t unit: x-axis unit (eV<default>, nm); \n" +\
                    "\t boundary: a-axis range <optional>; \n" +\
                    "\t figure_size: figure size <optional>. \n"
    else: help_info = None
    return help_info
