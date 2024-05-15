#### Reflectivity calculation and plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, W0612

import numpy as np
import matplotlib.pyplot as plt

from vmatplot.output import canvas_setting, color_sampling
from vmatplot.algorithms import process_boundary, extract_part, energy_to_wavelength, energy_to_frequency
from vmatplot.dielectric_function_plotting import create_matters_dielectric_function
from vmatplot.refractive_index import com_refractive
from vmatplot.extinction_coefficient import com_extinction

def com_reflectivity(density_energy_real,density_energy_imag):
    n = com_refractive(density_energy_real,density_energy_imag)
    k = com_extinction(density_energy_real,density_energy_imag)
    R = (np.square(n-1)+np.square(k))/(np.square(n+1)+np.square(k))
    return R

def create_matters_reflectivity(*args):
    # data = create_matters_dielectric_function(dielectric_list)
    # data[0] = current curve label
    # data[1] = dielectric data
    # data[2] = color family
    # data[3] = alpha
    # data[4] = linewidth
    return create_matters_dielectric_function(*args)

def plot_reflectivity_XZ_row(title, matters_list=None, unit=None, inplane_boundary=(None, None), outplane_boundary=(None, None)):
    help_info = "Usage: plot_reflectivity_XZ" + \
                "The independent value includes \n" +\
                "\t title, \n" +\
                "\t dielectric function data list, \n" +\
                "\t x-axis unit, \n" +\
                "\t Inplane photon wavelenght range (Optional), \n" +\
                "\t Outplane photon wavelenght range (Optional). \n"
    if title in ["help", "Help"]:
        print(help_info)
    # Figure settings
    fig_setting = canvas_setting(16, 6)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # Colors calling
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b"]

    # Materials information
    dataset = create_matters_reflectivity(matters_list)
    subtitles = ["In-plane", "Out-of-plane"]

    # Title
    # Suptitle
    current_title = title
    fig.suptitle(f"Reflectivity {current_title}", fontsize=fig_setting[3][0], y=1.00)

    # Boundary
    inplane_start, inplane_end = process_boundary(inplane_boundary)
    outplane_start, outplane_end = process_boundary(outplane_boundary)

    # Data plotting
    for supplot_index in range(2):
        ax = axes_element[supplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        ax.set_title(subtitles[supplot_index])

        for _, data in enumerate(dataset):
            # Labels
            current_label = data[0]
            # Inplane
            if supplot_index == 0:
                inplane_energy_full = data[1]["density_energy_real"]
                inplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                inplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])

                inplane_variables_full = com_reflectivity(data[1]["density_xx_real"],data[1]["density_xx_imag"])

                if unit in ["nm", "NM"]:
                    inplane_wavelength, inplane_absorption = extract_part(inplane_wavelength_full,inplane_variables_full,inplane_start,inplane_end)
                    ax.plot(inplane_wavelength,inplane_absorption,color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"{current_label}")
                else:
                    inplane_energy, inplane_absorption = extract_part(inplane_energy_full,inplane_variables_full,inplane_start,inplane_end)
                    ax.plot(inplane_energy,inplane_absorption,color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"{current_label}")

            # Outplane
            elif supplot_index == 1:
                outplane_energy_full = data[1]["density_energy_real"]
                outplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                outplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])

                outplane_variables_full = com_reflectivity(data[1]["density_zz_real"],data[1]["density_zz_imag"])

                if unit in ["nm", "NM"]:
                    outplane_wavelength, outplane_absorption = extract_part(outplane_wavelength_full,outplane_variables_full,outplane_start,outplane_end)
                    ax.plot(outplane_wavelength,outplane_absorption,color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"{current_label}")
                else:
                    outplane_energy, outplane_absorption = extract_part(outplane_energy_full,outplane_variables_full,outplane_start,outplane_end)
                    ax.plot(outplane_energy,outplane_absorption,color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"{current_label}")

        # axis label
        if supplot_index == 0:
            ax.set_ylabel(r"Reflectivity")
        if unit in ["nm", "NM"]:
            ax.set_xlabel(r"Photon wavelength (nm)")
        else:
            ax.set_xlabel(r"Photon energy (eV)")

        ax.legend(loc="upper right")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2,2), useOffset=False, useMathText=True)

        # Subplots label
        orderlab_shift = 0.05
        if supplot_index == 0:
            x_loc = 1-orderlab_shift
            y_loc = 0+orderlab_shift
        if supplot_index == 1:
            x_loc = 0+orderlab_shift
            y_loc = 0+orderlab_shift
        ax.annotate(f"({order_labels[supplot_index]})",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="center", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

def plot_reflectivity_XZ_col(title, matters_list=None, unit=None, inplane_boundary=(None, None), outplane_boundary=(None, None)):
    help_info = "Usage: plot_reflectivity_XZ" + \
                "The independent value includes \n" +\
                "\t title, \n" +\
                "\t dielectric function data list, \n" +\
                "\t x-axis unit, \n" +\
                "\t Inplane photon wavelenght range (Optional), \n" +\
                "\t Outplane photon wavelenght range (Optional). \n"
    if title in ["help", "Help"]:
        print(help_info)
    # Figure settings
    fig_setting = canvas_setting(8, 11)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # Colors calling
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b"]

    # Materials information
    dataset = create_matters_reflectivity(matters_list)
    subtitles = ["In-plane", "Out-of-plane"]

    # Title
    # Suptitle
    current_title = title
    fig.suptitle(f"Reflectivity {current_title}", fontsize=fig_setting[3][0], y=1.00)

    # Boundary
    inplane_start, inplane_end = process_boundary(inplane_boundary)
    outplane_start, outplane_end = process_boundary(outplane_boundary)

    # Data plotting
    for supplot_index in range(2):
        ax = axes_element[supplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        ax.set_title(subtitles[supplot_index])

        for _, data in enumerate(dataset):
            # Labels
            current_label = data[0]
            # Inplane
            if supplot_index == 0:
                inplane_energy_full = data[1]["density_energy_real"]
                inplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                inplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])

                inplane_variables_full = com_reflectivity(data[1]["density_xx_real"],data[1]["density_xx_imag"])

                if unit in ["nm", "NM"]:
                    inplane_wavelength, inplane_absorption = extract_part(inplane_wavelength_full,inplane_variables_full,inplane_start,inplane_end)
                    ax.plot(inplane_wavelength,inplane_absorption,color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"{current_label}")
                else:
                    inplane_energy, inplane_absorption = extract_part(inplane_energy_full,inplane_variables_full,inplane_start,inplane_end)
                    ax.plot(inplane_energy,inplane_absorption,color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"{current_label}")

            # Outplane
            elif supplot_index == 1:
                outplane_energy_full = data[1]["density_energy_real"]
                outplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                outplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])

                outplane_variables_full = com_reflectivity(data[1]["density_zz_real"],data[1]["density_zz_imag"])

                if unit in ["nm", "NM"]:
                    outplane_wavelength, outplane_absorption = extract_part(outplane_wavelength_full,outplane_variables_full,outplane_start,outplane_end)
                    ax.plot(outplane_wavelength,outplane_absorption,color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"{current_label}")
                else:
                    outplane_energy, outplane_absorption = extract_part(outplane_energy_full,outplane_variables_full,outplane_start,outplane_end)
                    ax.plot(outplane_energy,outplane_absorption,color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"{current_label}")

        # axis label
        if supplot_index == 1:
            if unit in ["nm", "NM"]:
                ax.set_xlabel(r"Photon wavelength (nm)")
            else:
                ax.set_xlabel(r"Photon energy (eV)")
        ax.set_ylabel(r"Reflectivity")
        ax.legend(loc="upper right")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2,2), useOffset=False, useMathText=True)

        # Subplots label
        orderlab_shift = 0.05
        x_loc = 0+orderlab_shift*0.75
        y_loc = 1-orderlab_shift
        ax.annotate(f"({order_labels[supplot_index]})",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="center", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

def plot_reflectivity_XZ_zoom

def plot_reflectivity_XZ(*args):
    if len(args) <= 5:
        # return plot_reflectivity_XZ_col(*args)
        return plot_reflectivity_XZ_row(*args)
    elif len(args) > 5:
        return plot_reflectivity_XZ_zoom(*args)
