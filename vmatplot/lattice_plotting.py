#### Free energy versus lattice constant plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter

from vmatplot.algorithms import fit_eos
from vmatplot.commons import extract_part
from vmatplot.output import canvas_setting, color_sampling
from vmatplot.lattice import read_free_energy_lattice_count
from vmatplot.lattice import specify_free_energy_lattice

# from vmatplot.lattice_plotting import

# testing: BC₃

def create_matters_lattice(lattice_list):
    # data = create_matters_lattice(lattice_list)
    # data[0] = current curve label
    # data[1] = lattice and free energy data for scatter
        # data[1][0] = scatter lattice values
        # data[1][1] = scatter free energy
    # data[2] = source lattice and free energy data
        # data[1][0] = source lattice values
        # data[1][1] = source free energy
    # data[3] = specific lattice and free energy
        # data[3][0] = specific lattice
        # data[3][1] = specific free energy
    # data[4] = color
    data = []
    for lattice_dir in lattice_list:
        if len(lattice_dir) == 3:
            label, data_path, spec_path = lattice_dir
            color = "blue"
            nsamp = "all"
            lattice_data = read_free_energy_lattice_count(data_path, nsamp)
            lattice_tot  = read_free_energy_lattice_count(data_path)
            data.append([label, lattice_data, lattice_tot, spec_path, color])
        elif len(lattice_dir) == 4:
            label, data_path, spec_path, color = lattice_dir
            nsamp = "all"
            lattice_data = read_free_energy_lattice_count(data_path, nsamp)
            lattice_tot  = read_free_energy_lattice_count(data_path)
            data.append([label, lattice_data, lattice_tot, spec_path, color])
        elif len(lattice_dir) == 5:
            label, data_path, spec_path, color, nsamp = lattice_dir
            lattice_data = read_free_energy_lattice_count(data_path, nsamp)
            lattice_tot  = read_free_energy_lattice_count(data_path)
            data.append([label, lattice_data, lattice_tot, spec_path, color])
    return data

def plot_free_energy_lattice_single(suptitle, lattice_list, lattice_range = (None, None)):
    # figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    plt.title(f"Tot energy versus lattice {suptitle}")

    # plot data
    lattice_info_set = create_matters_lattice(lattice_list)

    for _, lattice_info in enumerate(lattice_info_set):
        # current label
        current_label = lattice_info[0]
        # fitted curve
        selection_fitted = extract_part(lattice_info[2][0],lattice_info[2][1],lattice_range[0],lattice_range[1])
        fitted_lattice, fitted_free_energy = fit_eos(selection_fitted[0], selection_fitted[1])
        colors = color_sampling(lattice_info[4])
        if lattice_info[1] is not None:
            samples_scatter=extract_part(lattice_info[1][0],lattice_info[1][1],lattice_range[0],lattice_range[1])
            plt.plot(fitted_lattice, fitted_free_energy, color=colors[1], zorder=1)
            plt.scatter(samples_scatter[0], samples_scatter[1], s=48, fc="#FFFFFF", ec=colors[1], label=f"Source data {current_label}", zorder=1)
        else:
            plt.plot(fitted_lattice, fitted_free_energy, color=colors[1], label=f"Fitted curve {current_label}", zorder=1)
        # demonstrate the minimum free energy and the corresponding lattice
        selection_source=extract_part(lattice_info[2][0],lattice_info[2][1],lattice_range[0],lattice_range[1])
        energy_min_index = np.argmin(selection_source[1])       # Find the index of the minimum energy
        lattice_min = selection_source[0][energy_min_index]     # Retrieve the corresponding lattice value
        free_energy_min = selection_source[1][energy_min_index] # Retrieve the minimum free energy value
        plt.scatter(lattice_min, free_energy_min, s=48, fc=colors[2], ec=colors[2], label=f"Source lowest point {current_label}", zorder=1)
        # specific data
        if lattice_info[3] not in [None, ""]:
            selected_lattice, selected_energy = specify_free_energy_lattice(lattice_info[3])
            plt.scatter(selected_lattice,  selected_energy, s=24, ec=colors[0], fc=colors[0], label=f"Selected data {current_label}", zorder=2)
        # axis
        plt.xlabel(r"Lattice constant (Å)")
        plt.ylabel(r"Energy (eV)")
        plt.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

    plt.legend(loc=fig_setting[4])
    plt.tight_layout()

def plot_free_energy_lattice_double(suptitle, lattice_list_1, lattice_list_2, subtitle_1, subtitle_2,
                                    lattice_range_1 = (None, None), lattice_range_2 = (None, None)):
    # figure Settings
    fig_setting = canvas_setting(16, 6)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # Colors calling
    annotate_color = color_sampling("Grey")
    # order_labels = ["a","b"]

    # Subfigures information
    subtitles = [subtitle_1, subtitle_2]
    lattice_info_set_1 = create_matters_lattice(lattice_list_1)
    lattice_info_set_2 = create_matters_lattice(lattice_list_2)
    order_labels = subtitles

    # Title
    plt.suptitle(f"Total energy versus lattice {suptitle}", fontsize=fig_setting[3][0], y=1.00)
    lattice_ranges = [lattice_range_1, lattice_range_2]

    # plot data
    for subplot_index in range(2):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        # ax.set_title(subtitles[subplot_index])

        if subplot_index == 0:
            current_set = lattice_info_set_1
        else:
            current_set = lattice_info_set_2
        for _, lattice_info in enumerate(current_set):
            # current label
            current_label = lattice_info[0]
            # fitted curve
            selection_fitted = extract_part(lattice_info[2][0],lattice_info[2][1],lattice_ranges[subplot_index][0],lattice_ranges[subplot_index][1])
            fitted_lattice, fitted_free_energy = fit_eos(selection_fitted[0], selection_fitted[1])
            colors = color_sampling(lattice_info[4])
            if lattice_info[1] is not None:
                samples_scatter=extract_part(lattice_info[1][0],lattice_info[1][1],lattice_ranges[subplot_index][0],lattice_ranges[subplot_index][1])
                ax.plot(fitted_lattice, fitted_free_energy, label=f"Fitted curve {current_label}", color=colors[1], zorder=1)
                ax.scatter(samples_scatter[0], samples_scatter[1], s=48, fc="#FFFFFF", ec=colors[1], label=f"Source data {current_label}", zorder=1)
            else:
                ax.plot(fitted_lattice, fitted_free_energy, color=colors[1], label=f"Fitted curve {current_label}", zorder=1)
            # demonstrate the minimum free energy and the corresponding lattice
            selection_source=extract_part(lattice_info[2][0],lattice_info[2][1],lattice_ranges[subplot_index][0],lattice_ranges[subplot_index][1])
            energy_min_index = np.argmin(selection_source[1])       # Find the index of the minimum energy
            lattice_min = selection_source[0][energy_min_index]     # Retrieve the corresponding lattice value
            free_energy_min = selection_source[1][energy_min_index] # Retrieve the minimum free energy value
            # ax.scatter(lattice_min, free_energy_min, s=48, fc=colors[2], ec=colors[2], label=f"Source lowest point {current_label}", zorder=1)
            ax.scatter(lattice_min, free_energy_min, s=48, fc=colors[2], ec=colors[2], zorder=1)
            # specific data
            if lattice_info[3] not in [None, ""]:
                selected_lattice, selected_energy = specify_free_energy_lattice(lattice_info[3])
                # ax.scatter(selected_lattice,  selected_energy, s=24, ec=colors[0], fc=colors[0], label=f"Selected data {current_label}", zorder=2)
                ax.scatter(selected_lattice,  selected_energy, s=24, ec=colors[0], fc=colors[0], zorder=2, label=f"Selected data {current_label}")

        # axis label
        ax.set_xlabel(r"Lattice constant (Å)")

        if subplot_index == 0:
            ax.set_ylabel(r"Energy (eV)")

        # Legend
        ax.legend(loc="upper right")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

        # Subplots label
        orderlab_shift = 0.05
        x_loc = 0+orderlab_shift*0.75
        y_loc = 1-orderlab_shift
        ax.annotate(f"{order_labels[subplot_index]}",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="left", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

def plot_free_energy_lattice_triple(suptitle, lattice_list_1, lattice_list_2, lattice_list_3, subtitle_1, subtitle_2, subtitle_3,
                                    lattice_range_1 = (None, None), lattice_range_2 = (None, None), lattice_range_3 = (None, None)):
    # figure Settings
    fig_setting = canvas_setting(24, 6)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 3, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1], axs[2]]

    # Colors calling
    annotate_color = color_sampling("Grey")
    # order_labels = ["a","b","c"]

    # Subfigures information
    subtitles = [subtitle_1, subtitle_2, subtitle_3]
    lattice_info_set_1 = create_matters_lattice(lattice_list_1)
    lattice_info_set_2 = create_matters_lattice(lattice_list_2)
    lattice_info_set_3 = create_matters_lattice(lattice_list_3)
    order_labels = subtitles

    # Title
    plt.suptitle(f"Total energy versus lattice {suptitle}", fontsize=fig_setting[3][0], y=1.00)
    lattice_ranges = [lattice_range_1, lattice_range_2, lattice_range_3]

    # plot data
    for subplot_index in range(3):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        # ax.set_title(subtitles[subplot_index])

        if subplot_index == 0:
            current_set = lattice_info_set_1
        elif subplot_index == 1:
            current_set = lattice_info_set_2
        else:
            current_set = lattice_info_set_3
        for _, lattice_info in enumerate(current_set):
            # current label
            current_label = lattice_info[0]
            # fitted curve
            selection_fitted = extract_part(lattice_info[2][0],lattice_info[2][1],lattice_ranges[subplot_index][0],lattice_ranges[subplot_index][1])
            fitted_lattice, fitted_free_energy = fit_eos(selection_fitted[0], selection_fitted[1])
            colors = color_sampling(lattice_info[4])
            if lattice_info[1] is not None:
                samples_scatter=extract_part(lattice_info[1][0],lattice_info[1][1],lattice_ranges[subplot_index][0],lattice_ranges[subplot_index][1])
                ax.plot(fitted_lattice, fitted_free_energy, label=f"Fitted curve {current_label}", color=colors[1], zorder=1)
                ax.scatter(samples_scatter[0], samples_scatter[1], s=48, fc="#FFFFFF", ec=colors[1], label=f"Source data {current_label}", zorder=1)
                # ax.plot(fitted_lattice, fitted_free_energy, color=colors[1], zorder=1)
                # ax.scatter(samples_scatter[0], samples_scatter[1], s=48, fc="#FFFFFF", ec=colors[1], label=f"{current_label}", zorder=1)
            else:
                ax.plot(fitted_lattice, fitted_free_energy, color=colors[1], label=f"Fitted curve {current_label}", zorder=1)
            # demonstrate the minimum free energy and the corresponding lattice
            selection_source=extract_part(lattice_info[2][0],lattice_info[2][1],lattice_ranges[subplot_index][0],lattice_ranges[subplot_index][1])
            energy_min_index = np.argmin(selection_source[1])       # Find the index of the minimum energy
            lattice_min = selection_source[0][energy_min_index]     # Retrieve the corresponding lattice value
            free_energy_min = selection_source[1][energy_min_index] # Retrieve the minimum free energy value
            # ax.scatter(lattice_min, free_energy_min, s=48, fc=colors[2], ec=colors[2], label=f"Source lowest point {current_label}", zorder=1)
            ax.scatter(lattice_min, free_energy_min, s=48, fc=colors[2], ec=colors[2], zorder=1)
            # specific data
            if lattice_info[3] not in [None, ""]:
                selected_lattice, selected_energy = specify_free_energy_lattice(lattice_info[3])
                # ax.scatter(selected_lattice,  selected_energy, s=24, ec=colors[0], fc=colors[0], label=f"Selected data {current_label}", zorder=2)
                ax.scatter(selected_lattice,  selected_energy, s=24, ec=colors[0], fc=colors[0], label=f"Selected data {current_label}", zorder=2)

        # axis label
        ax.set_xlabel(r"Lattice constant (Å)")
        if subplot_index == 0:
            ax.set_ylabel(r"Energy (eV)")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

        # Legend
        ax.legend(loc="upper right")

        # Subplots label
        orderlab_shift = 0.05
        x_loc = 0+orderlab_shift*0.75
        y_loc = 1-orderlab_shift
        ax.annotate(f"{order_labels[subplot_index]}",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="left", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

def plot_free_energy_lattice_quadruple(suptitle, lattice_list_1, lattice_list_2, lattice_list_3, lattice_list_4,
                                       subtitle_1, subtitle_2, subtitle_3, subtitle_4,
                                       lattice_range_1=(None,None), lattice_range_2=(None,None),
                                       lattice_range_3=(None,None), lattice_range_4=(None,None)):
    # figure Settings
    fig_setting = canvas_setting(16, 12)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

    # Colors calling
    annotate_color = color_sampling("Grey")
    # order_labels = ["a","b","c","d"]

    # Subfigures information
    subtitles = [subtitle_1, subtitle_2, subtitle_3, subtitle_4]
    lattice_info_set_1 = create_matters_lattice(lattice_list_1)
    lattice_info_set_2 = create_matters_lattice(lattice_list_2)
    lattice_info_set_3 = create_matters_lattice(lattice_list_3)
    lattice_info_set_4 = create_matters_lattice(lattice_list_4)
    order_labels = subtitles

    # Title
    plt.suptitle(f"Total energy versus lattice {suptitle}", fontsize=fig_setting[3][0], y=1.00)
    lattice_ranges = [lattice_range_1, lattice_range_2, lattice_range_3, lattice_range_4]

    # plot data
    for subplot_index in range(4):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        # ax.set_title(subtitles[subplot_index])

        if subplot_index == 0:
            current_set = lattice_info_set_1
        elif subplot_index == 1:
            current_set = lattice_info_set_2
        elif subplot_index == 2:
            current_set = lattice_info_set_3
        else:
            current_set = lattice_info_set_4
        for _, lattice_info in enumerate(current_set):
            # current label
            current_label = lattice_info[0]
            # fitted curve
            selection_fitted = extract_part(lattice_info[2][0],lattice_info[2][1],lattice_ranges[subplot_index][0],lattice_ranges[subplot_index][1])
            fitted_lattice, fitted_free_energy = fit_eos(selection_fitted[0], selection_fitted[1])
            colors = color_sampling(lattice_info[4])
            if lattice_info[1] is not None:
                samples_scatter=extract_part(lattice_info[1][0],lattice_info[1][1],lattice_ranges[subplot_index][0],lattice_ranges[subplot_index][1])
                ax.plot(fitted_lattice, fitted_free_energy, label=f"Fitted curve {current_label}", color=colors[1], zorder=1)
                ax.scatter(samples_scatter[0], samples_scatter[1], s=48, fc="#FFFFFF", ec=colors[1], label=f"Source data {current_label}", zorder=1)
            else:
                ax.plot(fitted_lattice, fitted_free_energy, color=colors[1], label=f"Fitted curve {current_label}", zorder=1)
            # demonstrate the minimum free energy and the corresponding lattice
            selection_source=extract_part(lattice_info[2][0],lattice_info[2][1],lattice_ranges[subplot_index][0],lattice_ranges[subplot_index][1])
            energy_min_index = np.argmin(selection_source[1])       # Find the index of the minimum energy
            lattice_min = selection_source[0][energy_min_index]     # Retrieve the corresponding lattice value
            free_energy_min = selection_source[1][energy_min_index] # Retrieve the minimum free energy value
            # ax.scatter(lattice_min, free_energy_min, s=48, fc=colors[2], ec=colors[2], label=f"Source lowest point {current_label}", zorder=1)
            ax.scatter(lattice_min, free_energy_min, s=48, fc=colors[2], ec=colors[2], zorder=1)
            # specific data
            if lattice_info[3] not in [None, ""]:
                selected_lattice, selected_energy = specify_free_energy_lattice(lattice_info[3])
                # ax.scatter(selected_lattice,  selected_energy, s=24, ec=colors[0], fc=colors[0], label=f"Selected data {current_label}", zorder=2)
                ax.scatter(selected_lattice,  selected_energy, s=24, ec=colors[0], fc=colors[0], label=f"Selected data {current_label}", zorder=2)

        # axis label
        if subplot_index in [0,2]:
            ax.set_ylabel(r"Energy (eV)")
        if subplot_index in [2,3]:
            ax.set_xlabel(r"Lattice constant (Å)")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

        # Legend
        ax.legend(loc="upper right")

        # Subplots label
        orderlab_shift = 0.05
        x_loc = 0+orderlab_shift*0.75
        y_loc = 1-orderlab_shift
        ax.annotate(f"{order_labels[subplot_index]}",
                    xy=(x_loc, y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="left", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

def plot_free_energy_lattice(subfigures_amount, *args):
    help_info = "Usage: plot_free_energy_lattice \n" + \
                "Use summarize_free_energy_lattice_directory to extract the free energy versus lattice into free_energy_lattice.dat firstly.\n"
    if subfigures_amount == 1:
        return plot_free_energy_lattice_single(*args)
    elif subfigures_amount == 2:
        return plot_free_energy_lattice_double(*args)
    elif subfigures_amount == 3:
        return plot_free_energy_lattice_triple(*args)
    elif subfigures_amount == 4:
        return plot_free_energy_lattice_quadruple(*args)
    # help information
    else:
        print(help_info)
