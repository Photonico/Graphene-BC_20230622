#### linear optical properties
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, W0612

import numpy as np
import matplotlib.pyplot as plt

from vmatplot.dielectric_function import dielectric_systems_list
from vmatplot.commons import process_boundary, extract_part
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

def identify_linear_optical_functions(incoming = None):
    help_info = "Please use one of the following terminologies as a string-type variable:\n" + \
                "\t absorption coefficient, refractive index, extinction coefficient, reflectivity, energy-loss\n"
    linear_title, linear_flag, compfunc_name, plotfunc_name = None, None, None, None
    if incoming.lower() in ["absorption coefficient","absorption"]:
        linear_title = "Absorption coefficient"
        linear_flag = "absorption"
        compfunc_name = "comp_absorption_coefficient"
    elif incoming.lower() in ["refractive index","refractive"]:
        linear_title = "Refractive index"
        linear_flag = "refractive"
        compfunc_name = "comp_refractive_index"
    elif incoming.lower() in ["extinction coefficient", "extinction"]:
        linear_title = "Extinction coefficient"
        linear_flag = "extinction"
        compfunc_name = "comp_extinction_coefficient"
    elif incoming.lower() in ["reflectivity"]:
        linear_title = "Reflectivity"
        linear_flag = "reflectivity"
        compfunc_name = "comp_reflectivity"
    elif incoming.lower() in ["energy-loss function", "energy-loss spectrum", "energy-loss"]:
        linear_title = "Energy-loss spectrum"
        linear_flag = "energy-loss"
        compfunc_name = "comp_energy_loss_spectrum"
    else:
        print(help_info)
        return None
    return {"title":linear_title, "flag":linear_flag, "calculation function":compfunc_name}

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

def lop_plotting_help():
    help_info = "Usage: plot_linear_optical_property \n" +\
                "Demonstrate linear optical properties by each component \n" +\
                "\t suptitle: the suptitle; \n" +\
                "\t systems_list: dielectric function data list; \n" +\
                "\t components: select components in a list ({'xx'<default>, 'yy', 'zz', 'xy', 'yx', 'yz', 'zy', 'zx', 'xz'}); \n" +\
                "\t expansion: select one variable to expansion (rescale<auto>, properties, systems); \n" +\
                "\t layout: subfigures layout (horizontal<default>, vertical); \n" +\
                "\t unit: x-axis unit (eV<default>, nm); \n" +\
                "\t boundary: a-axis range <optional>; \n" +\
                "\t figure_size: figure size <optional>. \n"
    return help_info

### rebuild

def plot_linear_optical_property(suptitle, systems=None, properties=None, components="xx", layout="horizontal", unit="eV", boundary=(None, None), figure_size=(None, None)):

    ## Support information
    if suptitle.lower() in ["help", "support"]:
        help_info = lop_plotting_help()
        print(help_info)
        return

    ## properties labels and determination
    multi_prop_flag = None
    if isinstance(properties, str):
        multi_prop_flag = False
        formula_flag = identify_linear_optical_functions(properties)["flag"]
        formula_title = identify_linear_optical_functions(properties)["title"]
    elif isinstance(properties, list):
        formula_flags, formula_titles =[],[]
        for formula_input in properties:
            formula_flags.append(identify_linear_optical_functions(formula_input)["flag"])
            formula_titles.append(identify_linear_optical_functions(formula_input)["title"])
        if len(properties) == 1:
            multi_prop_flag = False
            formula_flag = formula_flags[0]
            formula_title = formula_titles[0]
        elif len(properties) > 1:
            multi_prop_flag = True
    if multi_prop_flag is True:
        print("We currently do not support multiple linear optical properties in one figure")
        return None

    ## multi component determination
    multi_comp_flag = None
    comp_labels, comp_aliases = [], []
    if isinstance(components, str):
        multi_comp_flag = False
        comp_labels.append(components)
        comp_aliases.append(f"{components}-component")
    elif isinstance(components, dict):
        if len(components) == 1:
            multi_comp_flag = False
            comp_labels.append(list(components.keys())[0])
            comp_aliases.append(list(components.values())[0])
        elif len(components) > 1:
            multi_comp_flag = True
            comp_labels.append(list(components.keys()))
            comp_aliases.append(list(components.values()))
    elif isinstance(components, list):
        if len(components) == 1:
            multi_comp_flag = False
        elif len(components) > 1:
            multi_comp_flag = True
        for comp_unit in components:
            if isinstance(comp_unit, dict):
                comp_labels.append(list(comp_unit.keys())[0])
                comp_aliases.append(list(comp_unit.values())[0])
            elif isinstance(comp_unit, str):
                comp_labels.append(comp_unit)
                comp_aliases.append(f"{comp_unit}-component")
    if multi_comp_flag is False:
        comp_label = comp_labels[0]
        comp_aliase = comp_aliases[0]

    ## figure settings
    layout_flag = "horizontal" if layout.lower() not in ["vertical", "ver","v"] else "vertical"
    if multi_comp_flag is False:
        ## figure settings
        fig_setting = canvas_setting() if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
        plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
        params = fig_setting[2]
        plt.rcParams.update(params)

    elif multi_comp_flag is True:
        ## figure settings
        if len(components) == 2:
            if layout_flag == "horizontal":
                fig_setting = canvas_setting(16, 6) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(1, 2, figsize=fig_setting[0], dpi=fig_setting[1])
                axs = axs.reshape(1, 2)
                axes_element = [axs[0, i] for i in range(2)]
            else:
                fig_setting = canvas_setting(8, 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1])
                axs = axs.reshape(2, 1)
                axes_element = [axs[i, 0] for i in range(2)]
        elif len(components) in [3, 5, 7]:
            if layout_flag == "horizontal":
                fig_setting = canvas_setting(8*len(components), 6) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(1, len(components), figsize=fig_setting[0], dpi=fig_setting[1])
                axes_element = [axs[i] for i in range(len(components))]
            else:
                fig_setting = canvas_setting(10, 6*len(components)) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(len(components), 1, figsize=fig_setting[0], dpi=fig_setting[1])
                axes_element = [axs[i] for i in range(len(components))]
        elif len(components) in [4, 6, 8]:
            if layout_flag == "horizontal":
                fig_setting = canvas_setting(8*len(components)/2, 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(2, int(len(components)/2), figsize=fig_setting[0], dpi=fig_setting[1])
                axes_element = [axs[i, j] for i in range(2) for j in range(int(len(components)/2))]
            else:
                fig_setting = canvas_setting(16, 6*len(components)/2+1) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(int(len(components)/2), 2, figsize=fig_setting[0], dpi=fig_setting[1])
                axes_element = [axs[i, j] for j in range(2) for i in range(int(len(components)/2))]
        elif len(components) == 9:
            if layout_flag == "horizontal":
                fig_setting = canvas_setting(24, 18) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(3, 3, figsize=fig_setting[0], dpi=fig_setting[1])
                axes_element = [axs[i, j] for i in range(3) for j in range(3)]
            else:
                fig_setting = canvas_setting(24, 18) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(3, 3, figsize=fig_setting[0], dpi=fig_setting[1])
                axes_element = [axs[i, j] for j in range(3) for i in range(3)]

    ## boundaries processing
    photon_start, photon_end = process_boundary(boundary)

    ## identify x-axis unit
    var_label = "wavelength" if unit and unit.lower() == "nm" else "energy"
    xaxis_str = "Photon wavelength (nm)" if var_label == "wavelength" else "Photon energy (eV)"

    ## systems information
    dataset = dielectric_systems_list(systems)

    ## suptitle
    if multi_comp_flag is False:
        plt.title(f"{formula_title} for {comp_aliase} {suptitle}", fontsize=fig_setting[3][0])
    elif multi_comp_flag is True:
        fig.suptitle(f"{formula_title} {suptitle}", fontsize=fig_setting[3][0])

    ## data plotting
    if multi_comp_flag is False:
        plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # component key
        current_component = comp_label.lower()
        data_key_real = f"density_{current_component}_real"
        data_key_imag = f"density_{current_component}_imag"

        # curve plotting
        for _, data in enumerate(dataset):
            energy_real, density_energy_real = extract_part(data[1]["density_energy_real"], data[1][data_key_real], photon_start, photon_end)
            energy_imag, density_energy_imag = extract_part(data[1]["density_energy_imag"], data[1][data_key_imag], photon_start, photon_end)
            frequency_real = energy_to_frequency(energy_real)
            wavelength_real = energy_to_wavelength(energy_real)
            if formula_flag == "absorption":
                variables = current_lop(formula_flag,frequency_real,density_energy_real,density_energy_imag)
            else:
                variables = current_lop(formula_flag,density_energy_real,density_energy_imag)
            if var_label == "energy":
                plt.plot(energy_real, variables, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{data[0]}")
            elif var_label == "wavelength":
                plt.plot(wavelength_real, variables, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{data[0]}")

        # axis labels
        plt.ylabel(f"{formula_title}")
        plt.xlabel(xaxis_str)

        # legends and scientific notation
        plt.legend(loc="best")
        plt.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

        plt.tight_layout()

    elif multi_comp_flag is True:
        for subplot_index in range(len(components)):
            ax = axes_element[subplot_index]
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

            # components index and subtitles
            component_index = subplot_index
            ax.set_title(comp_aliases[component_index])

            # current component key and label
            current_component = comp_labels[component_index].lower()
            data_key_real = f"density_{current_component}_real"
            data_key_imag = f"density_{current_component}_imag"

            # curve plotting
            for _, data in enumerate(dataset):
                energy_real, density_energy_real = extract_part(data[1]["density_energy_real"], data[1][data_key_real], photon_start, photon_end)
                energy_imag, density_energy_imag = extract_part(data[1]["density_energy_imag"], data[1][data_key_imag], photon_start, photon_end)
                frequency_real = energy_to_frequency(energy_real)
                wavelength_real = energy_to_wavelength(energy_real)
                if formula_flag == "absorption":
                    variables = current_lop(formula_flag,frequency_real,density_energy_real,density_energy_imag)
                else:
                    variables = current_lop(formula_flag,density_energy_real,density_energy_imag)
                if var_label == "energy":
                    ax.plot(energy_real, variables, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{data[0]}")
                elif var_label == "wavelength":
                    ax.plot(wavelength_real, variables, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{data[0]}")

            # axis labels
            if layout_flag == "horizontal" and len(components) == 2:
                ax.set_xlabel(xaxis_str)
                if subplot_index == 0:
                    ax.set_ylabel(f"{formula_title}")
            elif layout_flag == "vertical" and len(components) == 2:
                ax.set_ylabel(f"{formula_title}")
                if subplot_index == len(components)-1:
                    ax.set_xlabel(xaxis_str)
            elif layout_flag == "horizontal" and len(components) == 9:
                if subplot_index >= len(components)-3:
                    ax.set_xlabel(xaxis_str)
                if subplot_index%3==0:
                    ax.set_ylabel(f"{formula_title}")
            elif layout_flag == "vertical" and len(components) == 9:
                if subplot_index%3==2:
                    ax.set_xlabel(xaxis_str)
                if subplot_index in [0,1,2]:
                    ax.set_ylabel(f"{formula_title}")
            elif layout_flag == "horizontal" and len(components)%2==0:
                if subplot_index >= len(components)-(len(components)/2):
                    ax.set_xlabel(xaxis_str)
                if subplot_index%(len(components)/2)==0:
                    ax.set_ylabel(f"{formula_title}")
            elif layout_flag == "vertical" and len(components)%2==0:
                if (subplot_index+1) % len(components)==0:
                    ax.set_xlabel(xaxis_str)
                if subplot_index < len(components):
                    ax.set_ylabel(f"{formula_title}")
            elif layout_flag == "horizontal" and len(components) in [3,5,7]:
                ax.set_xlabel(xaxis_str)
                if subplot_index == 0:
                    ax.set_ylabel(f"{formula_title}")
            elif layout_flag == "vertical" and len(components) in [3,5,7]:
                if subplot_index == len(components)-1:
                    ax.set_xlabel(xaxis_str)
                ax.set_ylabel(f"{formula_title}")

            # legends and scientific notation
            ax.legend(loc="best")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

            plt.tight_layout()
