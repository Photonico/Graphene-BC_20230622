#### linear optical properties
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, W0612

import numpy as np
import matplotlib.pyplot as plt

from vmatplot.dielectric_function import dielectric_systems_list
from vmatplot.commons import process_boundaries_scaling, extract_part
from vmatplot.output import canvas_setting, color_sampling
from vmatplot.algorithms import energy_to_wavelength, energy_to_frequency

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
    linear_title, linear_flag, compfunc_name, plotfunc_name = None, None, None, None
    if incoming.lower() in ["absorption coefficient","absorption"]:
        linear_title = "Absorption coefficient"
        linear_flag = "absorption"
        compfunc_name = "comp_absorption_coefficient"
        plotfunc_name = "plot_absorption_coefficient"
    elif incoming.lower() in ["refractive index","refractive"]:
        linear_title = "Refractive"
        linear_flag = "refractive"
        compfunc_name = "comp_refractive_index"
        plotfunc_name = "plot_refractive_index"
    elif incoming.lower() in ["extinction coefficient", "extinction"]:
        linear_title = "Extinction coefficient"
        linear_flag = "extinction"
        compfunc_name = "comp_extinction_coefficient"
        plotfunc_name = "plot_extinction_coefficient"
    elif incoming.lower() in ["reflectivity"]:
        linear_title = "Reflectivity"
        linear_flag = "reflectivity"
        compfunc_name = "comp_reflectivity"
        plotfunc_name = "plot_reflectivity"
    elif incoming.lower() in ["energy-loss function", "energy-loss spectrum", "energy-loss"]:
        linear_title = "Energy-loss spectrum"
        linear_flag = "energy-loss"
        compfunc_name = "comp_energy_loss_spectrum"
        plotfunc_name = "plot_energy_loss_spectrum"
    else:
        print(help_info)
        return None
    return {"title":linear_title, "flag":linear_flag, "calculation function":compfunc_name, "plotting function": plotfunc_name}

# current linear optical propertie
def current_lop(lop_flag, *args):
    formula_flag = identify_linear_optical_functions(lop_flag)["flag"]
    if formula_flag == "absorption":
        return comp_absorption_coefficient(*args)
    elif formula_flag == "refractive":
        return comp_refractive_index(*args)
    elif formula_flag == "extinction":
        return comp_extinction_coefficient(*args)
    elif formula_flag == "reflectivity":
        return comp_reflectivity(*args)
    elif formula_flag == "energy-loss":
        return comp_energy_loss_spectrum(*args)

def determine_formula_flag(plotting_function_name):
    if plotting_function_name == "plot_absorption_coefficient":
        formula_flag = "absorption"
    elif plotting_function_name == "plot_refractive_index":
        formula_flag = "refractive"
    elif plotting_function_name == "plot_extinction_coefficient":
        formula_flag = "extinction"
    elif plotting_function_name == "plot_reflectivity":
        formula_flag = "reflectivity"
    elif plotting_function_name == "plot_energy_loss_spectrum":
        formula_flag = "energy-loss"
    return formula_flag

def lop_plotting_help(linear_chars):
    func_label = identify_linear_optical_functions(linear_chars)
    if func_label is not None:
        help_info = f"Usage: {func_label['plotting function']} \n" +\
                    f"Demonstrate {(func_label['title']).lower()} by each component \n" +\
                    "\t suptitle: the suptitle; \n" +\
                    "\t systems_list: dielectric function data list; \n" +\
                    "\t components: select components in a list ({'xx'<default>, 'yy', 'zz', 'xy', 'yx', 'yz', 'zy', 'zx', 'xz'}); \n" +\
                    "\t layout: subfigures layout (horizontal<default>, vertical); \n" +\
                    "\t unit: x-axis unit (eV<default>, nm); \n" +\
                    "\t boundary: a-axis range <optional>; \n" +\
                    "\t figure_size: figure size <optional>. \n"
    else: help_info = None
    return help_info

def plot_lop_monocomp(suptitle, systems_list=None, current_property=None, components="xx", layout="horizontal",
                      unit="eV", boundary=(None,None), figure_size=(None,None)):
    ## Help information
    current_help = lop_plotting_help("absorption")
    if suptitle.lower() =="help":
        print(current_help)

    print("test")

def plot_linear_optical_property(suptitle, systems_list=None, current_property=None, components="xx", layout="horizontal",
                                 unit="eV", boundary=(None,None), figure_size=(None,None)):
    ## Help information
    current_help = lop_plotting_help("absorption")
    if suptitle.lower() =="help":
        print(current_help)

    ## scale flag and databoundaries
    scale_flag, source_start, source_end, scaled_start, scaled_end = process_boundaries_scaling(boundary)

    ## multi components flag
    if isinstance(components, str) or isinstance(components, dict):
        return plot_lop_monocomp(suptitle, systems_list, current_property, components,layout,unit, boundary, figure_size)
    elif isinstance(components, list) and len(components) == 1:
        return plot_lop_monocomp(suptitle, systems_list, current_property, components,layout, unit, boundary, figure_size)

    component_labels, comp_aliases = [], []
    ## components
    for comp in components:
        if isinstance(comp, dict):
            for key, value in comp.items():
                component_labels.append(f"{key}-component")
                comp_aliases.append(value)
        else:
            component_labels.append(f"{comp}-component")
            comp_aliases.append(f"{comp}-component")

    ## figure settings
    layout_flag = "horizontal" if layout.lower() not in ["vertical", "ver","v"] else "vertical"
    if scale_flag is True:
        if layout_flag == "horizontal":
            fig_setting = canvas_setting(8*len(components), 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(2, len(components), figsize=fig_setting[0], dpi=fig_setting[1])
            axes_element = [axs[i, j] for j in range(len(components)) for i in range(2)] if len(components) != 1 else [axs[0], axs[1]]
        else:
            fig_setting = canvas_setting(16, 6*len(components)) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(len(components), 2, figsize=fig_setting[0], dpi=fig_setting[1])
            axes_element = [axs[i, j] for i in range(len(components)) for j in range(2)] if len(components) != 1 else [axs[0], axs[1]]
    else:
        if layout_flag == "horizontal":
            fig_setting = canvas_setting(8*len(components), 6) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(1, len(components), figsize=fig_setting[0], dpi=fig_setting[1])
            axes_element = [axs[i] for i in range(len(components))]
        else:
            fig_setting = canvas_setting(8, 6*len(components)) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(len(components), 1, figsize=fig_setting[0], dpi=fig_setting[1])
            axes_element = [axs[i] for i in range(len(components))]

    ## identify x-axis unit
    var_label = "wavelength" if unit and unit.lower() == "nm" else "energy"
    xaxis_str = "Photon wavelength (nm)" if var_label == "wavelength" else "Photon energy (eV)"

    ## systems information
    dataset = lop_systems(systems_list)
    component_labels = [comp.lower() + "-component" for comp in components] if not comp_aliases else comp_aliases

    ## formula flag
    formula_flag = identify_linear_optical_functions(current_property)["flag"]

    ## suptitle
    suptitle_prefix = identify_linear_optical_functions(formula_flag)["title"]
    fig.suptitle(f"{suptitle_prefix} {suptitle}\n", fontsize=fig_setting[3][0])

    ## testing area
    # print(identify_linear_optical_functions(formula_flag)["title"])

    ## data plotting
    # scale_flag is True (scaling opened)
    if scale_flag is True:
        for supplot_index in range(2*len(components)):
            ax = axes_element[supplot_index]
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

            # initialization
            wavelength_starts, wavelength_ends, energy_starts, energy_ends = [], [], [], []
            if supplot_index%2 == 0:
                x_start = source_start
                x_end = source_end
            else:
                x_start = scaled_start
                x_end = scaled_end

            # current component index and label
            component_index = supplot_index // 2
            if isinstance(components[component_index], dict):
                current_component = list(components[component_index].keys())[0]
            else:
                current_component = components[component_index].lower()
            data_key_real = f"density_{current_component}_real"
            data_key_imag = f"density_{current_component}_imag"

            # curve plotting
            for _, data in enumerate(dataset):
                energy_full = data[1]["density_energy_real"]
                density_real_full, density_imag_full = data[1][data_key_real], data[1][data_key_imag]
                frequency_real_full = energy_to_frequency(energy_full)

                # calculate curve
                if formula_flag == "absorption":
                    curve_full = current_lop(formula_flag, frequency_real_full, density_real_full, density_imag_full)
                else:
                    curve_full = current_lop(formula_flag, density_real_full, density_imag_full)

                if var_label == "energy":
                    demo_var, curve_energy = extract_part(energy_full, curve_full, x_start, x_end)
                    ax.plot(demo_var, curve_energy, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{data[0]}")
                    energy_starts.append(min(demo_var))
                    energy_ends.append(max(demo_var))
                elif var_label == "wavelength":
                    wavelength_full = energy_to_wavelength(energy_full)
                    wavelength_var, curve_wavelength = extract_part(wavelength_full, curve_full, x_start, x_end)
                    ax.plot(wavelength_var, curve_wavelength, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{data[0]}")
                    wavelength_starts.append(min(wavelength_var))
                    wavelength_ends.append(np.max(np.array(wavelength_var)[np.isfinite(wavelength_var)]))

            # subtitles and axis label (self-assertive): subtitles
            if layout_flag == "vertical" and supplot_index in range(2):
                ax.set_title(["Original view", "Rescaled view"][supplot_index])
            elif layout_flag == "horizontal" and supplot_index%2 == 0:
                ax.set_title(component_labels[component_index])

            # ylabel
            if layout_flag == "vertical" and supplot_index%2 == 0:
                ax.set_ylabel(f"{suptitle_prefix} for {component_labels[component_index]}")
            elif layout_flag == "horizontal" and supplot_index in range(2):
                ax.set_ylabel([f"{suptitle_prefix}", f"{suptitle_prefix} (Rescaled)"][supplot_index])
            # xlabel
            if layout_flag == "vertical" and supplot_index >= 2*len(components)-2:
                ax.set_xlabel(xaxis_str)
            elif layout_flag == "horizontal" and supplot_index%2 == 1:
                ax.set_xlabel(xaxis_str)

            ax.legend(loc="best")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

    # scale_flag is False (scaling closed)
    else:
        for supplot_index in range(len(components)):
            ax = axes_element[supplot_index]
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

            # initialization
            wavelength_starts, wavelength_ends, energy_starts, energy_ends = [], [], [], []
            x_start = source_start
            x_end = source_end

            # current component index and label
            component_index = supplot_index
            if isinstance(components[component_index], dict):
                current_component = list(components[component_index].keys())[0]
            else:
                current_component = components[component_index].lower()
            data_key_real = f"density_{current_component}_real"
            data_key_imag = f"density_{current_component}_imag"

            # curve plotting
            for _, data in enumerate(dataset):
                energy_full = data[1]["density_energy_real"]
                density_real_full, density_imag_full = data[1][data_key_real], data[1][data_key_imag]
                frequency_real_full = energy_to_frequency(energy_full)

                # calculate curve
                if formula_flag == "absorption":
                    curve_full = current_lop(formula_flag, frequency_real_full, density_real_full, density_imag_full)
                else:
                    curve_full = current_lop(density_real_full, density_imag_full)

                if var_label == "energy":
                    demo_var, curve_energy = extract_part(energy_full, curve_full, x_start, x_end)
                    ax.plot(demo_var, curve_energy, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{data[0]}")
                    energy_starts.append(min(demo_var))
                    energy_ends.append(max(demo_var))
                elif var_label == "wavelength":
                    wavelength_full = energy_to_wavelength(energy_full)
                    wavelength_var, curve_wavelength = extract_part(wavelength_full, curve_full, x_start, x_end)
                    ax.plot(wavelength_var, curve_wavelength, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{data[0]}")
                    wavelength_starts.append(min(wavelength_var))
                    wavelength_ends.append(np.max(np.array(wavelength_var)[np.isfinite(wavelength_var)]))

            # subtitles and axis label (self-assertive): subtitles
            if layout_flag == "vertical":
                ax.set_ylabel(f"{suptitle_prefix} for {component_labels[component_index]}")
                if layout_flag == "vertical" and supplot_index == len(components)-1:
                    ax.set_xlabel(xaxis_str)
            else:
                ax.set_title(component_labels[component_index])
                ax.set_xlabel(xaxis_str)
                if supplot_index == 0:
                    ax.set_ylabel(f"{suptitle_prefix}")

            ax.legend(loc="best")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

    plt.tight_layout()
