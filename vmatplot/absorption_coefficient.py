#### Absorption coefficient calculation and plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, W0612

import numpy as np
import matplotlib.pyplot as plt

from vmatplot.output import canvas_setting, color_sampling
from vmatplot.algorithms import energy_to_wavelength, energy_to_frequency
from vmatplot.commons import process_boundary, extract_part
from vmatplot.dielectric_function_plotting import create_matters_dielectric_function

# Speed of light in vacuum in meters per second
c_ms = 299792458
# Speed of light in vacuum nanometers per second
c_nm = c_ms * 1e9  # 1e9 is equivalent to 10^9

def comp_absorption_coefficient(frequency,density_energy_real,density_energy_imag):
    coe = (np.sqrt(2)*frequency/c_nm)*(np.sqrt(np.sqrt(np.square(density_energy_real)+np.square(density_energy_imag))-density_energy_real))
    return coe

def comp_absorption_coefficient_custom(frequency,density_energy_real,density_energy_imag):
    coe = np.sqrt(2)*frequency*(np.sqrt(np.square(density_energy_real)+np.square(density_energy_imag))-density_energy_real)
    return coe

def create_matters_absorption(*args):
    # data = create_matters_dielectric_function(dielectric_list)
    # data[0] = current curve label
    # data[1] = dielectric data
    # data[2] = color family
    # data[3] = linestyle
    # data[4] = alpha
    # data[5] = linewidth
    return create_matters_dielectric_function(*args)

def plot_absorption_XXZZ_row(title, absorption_list=None, unit=None, abs_type=None, inplane_boundary=(None, None), outplane_boundary=(None, None)):
    help_info = "Usage: plot_absorption_XXZZ \n" + \
                "The independent value includes \n" +\
                "\t title, \n" +\
                "\t dielectric function data list, \n" +\
                "\t x-axis unit, \n" +\
                "\t formula type, \n" +\
                "\t Inplane photon wavelenght range (Optional), \n" +\
                "\t Outplane photon wavelenght range (Optional). \n"
    if title in ["help", "Help"]:
        print(help_info)
        return

    # Figure settings
    fig_setting = canvas_setting(16, 6)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # Colors calling
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b"]

    # Materials information
    dataset = create_matters_absorption(absorption_list)
    subtitles = ["In-plane", "Out-of-plane"]

    # Title
    # Suptitle
    current_title = title
    fig.suptitle(f"Absorption coefficient {current_title}", fontsize=fig_setting[3][0], y=1.00)

    # Boundary
    if outplane_boundary == (None,None):
        outplane_boundary = inplane_boundary
    inplane_start, inplane_end = process_boundary(inplane_boundary)
    outplane_start, outplane_end = process_boundary(outplane_boundary)

    # Data plotting
    for subplot_index in range(2):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        ax.set_title(subtitles[subplot_index])

        for _, data in enumerate(dataset):
            # Labels
            current_label = data[0]
            # Inplane
            if subplot_index == 0:
                inplane_energy_full = data[1]["density_energy_real"]
                inplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                inplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])
                if abs_type in ["CUSTOM", "Custom", "custom"]:
                    inplane_absorption_full = comp_absorption_coefficient_custom(inplane_frequency_full,data[1]["density_xx_real"],data[1]["density_xx_imag"])
                else:
                    inplane_absorption_full = comp_absorption_coefficient(inplane_frequency_full,data[1]["density_xx_real"],data[1]["density_xx_imag"])

                if unit in ["nm", "NM"]:
                    inplane_wavelength, inplane_absorption = extract_part(inplane_wavelength_full,inplane_absorption_full,inplane_start,inplane_end)
                    ax.plot(inplane_wavelength,inplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")
                else:
                    inplane_energy, inplane_absorption = extract_part(inplane_energy_full,inplane_absorption_full,inplane_start,inplane_end)
                    ax.plot(inplane_energy,inplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")

            # Outplane
            elif subplot_index == 1:
                outplane_energy_full = data[1]["density_energy_real"]
                outplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                outplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])
                if abs_type in ["CUSTOM", "Custom", "custom"]:
                    outplane_absorption_full = comp_absorption_coefficient_custom(outplane_frequency_full,data[1]["density_zz_real"],data[1]["density_zz_imag"])
                else:
                    outplane_absorption_full = comp_absorption_coefficient(outplane_frequency_full,data[1]["density_zz_real"],data[1]["density_zz_imag"])

                if unit in ["nm", "NM"]:
                    outplane_wavelength, outplane_absorption = extract_part(outplane_wavelength_full,outplane_absorption_full,outplane_start,outplane_end)
                    ax.plot(outplane_wavelength,outplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")
                else:
                    outplane_energy, outplane_absorption = extract_part(outplane_energy_full,outplane_absorption_full,outplane_start,outplane_end)
                    ax.plot(outplane_energy,outplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")

        # axis label
        if subplot_index == 0:
            ax.set_ylabel(r"Absorption coefficient")
        if unit in ["nm", "NM"]:
            ax.set_xlabel(r"Photon wavelength (nm)")
        else:
            ax.set_xlabel(r"Photon energy (eV)")

        ax.legend(loc="best")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

        # # Subplots label
        # orderlab_shift = 0.05
        # # if subplot_index == 0:
        # #     x_loc = 1-orderlab_shift
        # #     y_loc = 0+orderlab_shift
        # # if subplot_index == 1:
        # #     x_loc = 0+orderlab_shift
        # #     y_loc = 0+orderlab_shift
        # x_loc = 0+orderlab_shift*0.75
        # y_loc = 1-orderlab_shift
        # ax.annotate(f"({order_labels[subplot_index]})",
        #             xy=(x_loc,y_loc),
        #             xycoords="axes fraction",
        #             fontsize=1.0 * 16,
        #             ha="center", va="center",
        #             bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

def plot_absorption_XXZZ_col(title, absorption_list=None, unit=None, abs_type=None, inplane_boundary=(None, None), outplane_boundary=(None, None)):
    help_info = "Usage: plot_absorption_XXZZ \n" + \
                "The independent value includes \n" +\
                "\t title, \n" +\
                "\t dielectric function data list, \n" +\
                "\t x-axis unit, \n" +\
                "\t formula type, \n" +\
                "\t Inplane photon wavelenght range (Optional), \n" +\
                "\t Outplane photon wavelenght range (Optional). \n"
    if title in ["help", "Help"]:
        print(help_info)
        return

    # Figure settings
    fig_setting = canvas_setting(8, 11)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # Colors calling
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b"]

    # Materials information
    dataset = create_matters_absorption(absorption_list)
    subtitles = ["In-plane", "Out-of-plane"]

    # Title
    # Suptitle
    current_title = title
    # fig.suptitle(f"Absorption coefficient {current_title}", fontsize=fig_setting[3][0], y=0.96)
    fig.suptitle(f"Absorption coefficient {current_title}", fontsize=fig_setting[3][0], y=1.00)

    # Boundary
    if outplane_boundary == (None,None):
        outplane_boundary = inplane_boundary
    inplane_start, inplane_end = process_boundary(inplane_boundary)
    outplane_start, outplane_end = process_boundary(outplane_boundary)

    # Data plotting
    for subplot_index in range(2):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        ax.set_title(subtitles[subplot_index])

        for _, data in enumerate(dataset):
            # Labels
            current_label = data[0]
            # Inplane
            if subplot_index == 0:
                inplane_energy_full = data[1]["density_energy_real"]
                inplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                inplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])
                if abs_type in ["CUSTOM", "Custom", "custom"]:
                    inplane_absorption_full = comp_absorption_coefficient_custom(inplane_frequency_full,data[1]["density_xx_real"],data[1]["density_xx_imag"])
                else:
                    inplane_absorption_full = comp_absorption_coefficient(inplane_frequency_full,data[1]["density_xx_real"],data[1]["density_xx_imag"])

                if unit in ["nm", "NM"]:
                    inplane_wavelength, inplane_absorption = extract_part(inplane_wavelength_full,inplane_absorption_full,inplane_start,inplane_end)
                    ax.plot(inplane_wavelength,inplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")
                else:
                    inplane_energy, inplane_absorption = extract_part(inplane_energy_full,inplane_absorption_full,inplane_start,inplane_end)
                    ax.plot(inplane_energy,inplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")

            # Outplane
            elif subplot_index == 1:
                outplane_energy_full = data[1]["density_energy_real"]
                outplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                outplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])
                if abs_type in ["CUSTOM", "Custom", "custom"]:
                    outplane_absorption_full = comp_absorption_coefficient_custom(outplane_frequency_full,data[1]["density_zz_real"],data[1]["density_zz_imag"])
                else:
                    outplane_absorption_full = comp_absorption_coefficient(outplane_frequency_full,data[1]["density_zz_real"],data[1]["density_zz_imag"])

                if unit in ["nm", "NM"]:
                    outplane_wavelength, outplane_absorption = extract_part(outplane_wavelength_full,outplane_absorption_full,outplane_start,outplane_end)
                    ax.plot(outplane_wavelength,outplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")
                else:
                    outplane_energy, outplane_absorption = extract_part(outplane_energy_full,outplane_absorption_full,outplane_start,outplane_end)
                    ax.plot(outplane_energy,outplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")

        # axis label
        if subplot_index == 1:
            if unit in ["nm", "NM"]:
                ax.set_xlabel(r"Photon wavelength (nm)")
            else:
                ax.set_xlabel(r"Photon energy (eV)")
        ax.set_ylabel(r"Absorption coefficient")
        ax.legend(loc="best")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

        # # Subplots label
        # orderlab_shift = 0.05
        # # x_loc = 1-orderlab_shift*0.75
        # x_loc = 0+orderlab_shift*0.75
        # y_loc = 1-orderlab_shift
        # ax.annotate(f"({order_labels[subplot_index]})",
        #             xy=(x_loc,y_loc),
        #             xycoords="axes fraction",
        #             fontsize=1.0 * 16,
        #             ha="center", va="center",
        #             bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

def plot_absorption_XXZZ_zoom(title, matters_list=None, unit=None, abs_type=None,
                              inplane_boundary_1=(None, None), outplane_boundary_1=(None, None),
                              inplane_boundary_2=(None, None), outplane_boundary_2=(None, None)):
    help_info = "Usage: plot_absorption_XXZZ \n" + \
                "The independent value includes \n" +\
                "\t title, \n" +\
                "\t dielectric function data list, \n" +\
                "\t x-axis unit, \n" +\
                "\t formula type, \n" +\
                "\t Inplane photon wavelenght range 1 (Optional), \n" +\
                "\t Outplane photon wavelenght range 1 (Optional). \n" +\
                "\t Inplane photon wavelenght range 2 (Optional), \n" +\
                "\t Outplane photon wavelenght range 2 (Optional). \n"
    if title in ["help", "Help"]:
        print(help_info)
        return

    # General information
    prop = "Absorption coefficient"
    comp_function = create_matters_absorption

    # Figure settings
    fig_setting = canvas_setting(16, 12)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

    # Colors calling
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b","c","d"]

    # Materials information
    dataset = comp_function(matters_list)
    subtitles = ["In-plane", "Out-of-plane", "In-plane (zoomed)", "Out-of-plane (zoomed)"]

    # Data plotting
    for subplot_index in range(4):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        ax.set_title(subtitles[subplot_index])

    # Title
    # Suptitle
    current_title = title
    fig.suptitle(f"{prop} {current_title}", fontsize=fig_setting[3][0], y=1.00)

    # Boundary
    if outplane_boundary_1 == (None,None):
        outplane_boundary_1, inplane_boundary_2, outplane_boundary_2 = inplane_boundary_1
    elif inplane_boundary_2 == (None,None):
        inplane_boundary_2 = inplane_boundary_1; outplane_boundary_2 = outplane_boundary_1
    elif outplane_boundary_2 == (None,None):
        inplane_range_1 = process_boundary(inplane_boundary_1)[-1]-process_boundary(inplane_boundary_1)[0]
        inplane_range_2 = process_boundary(inplane_boundary_2)[-1]-process_boundary(inplane_boundary_2)[0]
        rate = inplane_range_2/inplane_range_1
        outplane_boundary_2 = tuple(x * rate for x in outplane_boundary_1)

    inplane_start_1, inplane_end_1   = process_boundary(inplane_boundary_1)
    outplane_start_1, outplane_end_1 = process_boundary(outplane_boundary_1)
    inplane_start_2, inplane_end_2   = process_boundary(inplane_boundary_2)
    outplane_start_2, outplane_end_2 = process_boundary(outplane_boundary_2)

    # Data plotting
    for subplot_index in range(4):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        ax.set_title(subtitles[subplot_index])

        for _, data in enumerate(dataset):
            # Labels
            current_label = data[0]

            # Inplane
            if subplot_index == 0:
                inplane_energy_full = data[1]["density_energy_real"]
                inplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                inplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])
                if abs_type in ["CUSTOM", "Custom", "custom"]:
                    inplane_absorption_full = comp_absorption_coefficient_custom(inplane_frequency_full,data[1]["density_xx_real"],data[1]["density_xx_imag"])
                else:
                    inplane_absorption_full = comp_absorption_coefficient(inplane_frequency_full,data[1]["density_xx_real"],data[1]["density_xx_imag"])

                if unit in ["nm", "NM"]:
                    inplane_wavelength, inplane_absorption = extract_part(inplane_wavelength_full,inplane_absorption_full,inplane_start_1,inplane_end_1)
                    ax.plot(inplane_wavelength,inplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")
                else:
                    inplane_energy, inplane_absorption = extract_part(inplane_energy_full,inplane_absorption_full,inplane_start_1,inplane_end_1)
                    ax.plot(inplane_energy,inplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")

            # Outplane
            elif subplot_index == 1:
                outplane_energy_full = data[1]["density_energy_real"]
                outplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                outplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])
                if abs_type in ["CUSTOM", "Custom", "custom"]:
                    outplane_absorption_full = comp_absorption_coefficient_custom(outplane_frequency_full,data[1]["density_zz_real"],data[1]["density_zz_imag"])
                else:
                    outplane_absorption_full = comp_absorption_coefficient(outplane_frequency_full,data[1]["density_zz_real"],data[1]["density_zz_imag"])

                if unit in ["nm", "NM"]:
                    outplane_wavelength, outplane_absorption = extract_part(outplane_wavelength_full,outplane_absorption_full,outplane_start_1,outplane_end_1)
                    ax.plot(outplane_wavelength,outplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")
                else:
                    outplane_energy, outplane_absorption = extract_part(outplane_energy_full,outplane_absorption_full,outplane_start_1,outplane_end_1)
                    ax.plot(outplane_energy,outplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")

            # Inplane (zoomed)
            elif subplot_index == 2:
                inplane_energy_full = data[1]["density_energy_real"]
                inplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                inplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])
                if abs_type in ["CUSTOM", "Custom", "custom"]:
                    inplane_absorption_full = comp_absorption_coefficient_custom(inplane_frequency_full,data[1]["density_xx_real"],data[1]["density_xx_imag"])
                else:
                    inplane_absorption_full = comp_absorption_coefficient(inplane_frequency_full,data[1]["density_xx_real"],data[1]["density_xx_imag"])

                if unit in ["nm", "NM"]:
                    inplane_wavelength, inplane_absorption = extract_part(inplane_wavelength_full,inplane_absorption_full,inplane_start_2,inplane_end_2)
                    ax.plot(inplane_wavelength,inplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")
                else:
                    inplane_energy, inplane_absorption = extract_part(inplane_energy_full,inplane_absorption_full,inplane_start_2,inplane_end_2)
                    ax.plot(inplane_energy,inplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")

            # Outplane (zoomed)
            elif subplot_index == 3:
                outplane_energy_full = data[1]["density_energy_real"]
                outplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                outplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])
                if abs_type in ["CUSTOM", "Custom", "custom"]:
                    outplane_absorption_full = comp_absorption_coefficient_custom(outplane_frequency_full,data[1]["density_zz_real"],data[1]["density_zz_imag"])
                else:
                    outplane_absorption_full = comp_absorption_coefficient(outplane_frequency_full,data[1]["density_zz_real"],data[1]["density_zz_imag"])

                if unit in ["nm", "NM"]:
                    outplane_wavelength, outplane_absorption = extract_part(outplane_wavelength_full,outplane_absorption_full,outplane_start_2,outplane_end_2)
                    ax.plot(outplane_wavelength,outplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")
                else:
                    outplane_energy, outplane_absorption = extract_part(outplane_energy_full,outplane_absorption_full,outplane_start_2,outplane_end_2)
                    ax.plot(outplane_energy,outplane_absorption,color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"{current_label}")

        # axis label
        if subplot_index in [0,2]:
            ax.set_ylabel(f"{prop}")
        if subplot_index in [2,3]:
            if unit in ["nm", "NM"]:
                ax.set_xlabel(r"Photon wavelength (nm)")
            else:
                ax.set_xlabel(r"Photon energy (eV)")
        ax.legend(loc="best")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

        # # Subplots label
        # orderlab_shift = 0.05
        # # if subplot_index == 0:
        # #     x_loc = 1-orderlab_shift*0.75
        # #     y_loc = 0+orderlab_shift
        # # elif subplot_index == 1:
        # #     x_loc = 0+orderlab_shift*0.75
        # #     y_loc = 0+orderlab_shift
        # # elif subplot_index == 2:
        # #     x_loc = 1-orderlab_shift*0.75
        # #     y_loc = 1-orderlab_shift
        # # elif subplot_index == 3:
        # #     x_loc = 0+orderlab_shift*0.75
        # #     y_loc = 1-orderlab_shift
        # x_loc = 0+orderlab_shift*0.75
        # y_loc = 1-orderlab_shift

        # ax.annotate(f"({order_labels[subplot_index]})",
        #                 xy=(x_loc,y_loc),
        #                 xycoords="axes fraction",
        #                 fontsize=1.0 * 16,
        #                 ha="center", va="center",
        #                 bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

def plot_absorption_XXZZ(*args):
    if len(args) <= 6:
        # return plot_absorption_XXZZ_col(*args)
        return plot_absorption_XXZZ_row(*args)
    elif len(args) > 6:
        return plot_absorption_XXZZ_zoom(*args)
