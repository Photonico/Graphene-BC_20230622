#### Lattice plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

# Matplotlib colormap reference: https://matplotlib.org/stable/gallery/color/colormap_reference.html

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import griddata
from Store.output import canvas_setting, color_sampling
from Store.algorithms import polynomially_fit_curve
from Store.lattice import read_lattice_free_energy_data, read_lattice_free_energy_count, specify_lattice_free_energy
from Store.lattice_biolayer import read_biolayer_lattice_data, extract_minimum_biolayer_lattice

def plot_lattice_free_energy_solo(matter, sample_count, source_data, color_family, selected_data=None):
    # Settings input
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Figure title
    plt.title(f"Free energy versus lattice for {matter}"); plt.xlabel(r"Lattice constant (Å)"); plt.ylabel(r"Energy (eV)")

    # Color calling
    colors = color_sampling(color_family)

    # Data input
    lattice_source, free_energy_source = read_lattice_free_energy_data(source_data)
    lattice_sample, free_energy_sample = read_lattice_free_energy_count(source_data, sample_count)

    fitted_lattice, fitted_free_energy = polynomially_fit_curve(lattice_source, free_energy_source, 3, 4000)
    if selected_data is not None:
        selected_lattice, select_free_energy = specify_lattice_free_energy(selected_data)

    # Minimum free energy and the corresponding lattice
    min_energy_index = free_energy_source.index(min(free_energy_source))
    min_lattice = lattice_source[min_energy_index]
    min_free_energy = min(free_energy_source)

    # Plotting
    plt.plot(fitted_lattice, fitted_free_energy, c=colors[1], label="Fitted data", zorder=1)
    plt.scatter(lattice_sample, free_energy_sample, s=48, fc="#FFF", ec=colors[1], label="Source data", zorder=1)
    plt.scatter(min_lattice, min_free_energy, s=48, fc=colors[2], ec=colors[2], label="Source lowest point", zorder=1)
    if selected_data is not None:
        plt.scatter(selected_lattice,  select_free_energy, s=24, fc=colors[0], ec=colors[0], label="Selected data", zorder=2)

    plt.legend(loc=fig_setting[3])
    plt.show()

def plot_lattice_free_energy_duo(title, sample_count, matter1, source_data1, color_family1, matter2, source_data2, color_family2, selected_data1=None, selected_data2=None):
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    plt.title(f"Free energy versus lattice for {title}"); plt.xlabel(r"Lattice constant (Å)"); plt.ylabel(r"Energy (eV)")

    # Color calling
    colors1 = color_sampling(color_family1)
    colors2 = color_sampling(color_family2)

    # Data input
    lattice_source1, free_energy_source1 = read_lattice_free_energy_data(source_data1)
    lattice_source2, free_energy_source2 = read_lattice_free_energy_data(source_data2)

    fitted_lattice1, fitted_free_energy1 = polynomially_fit_curve(lattice_source1, free_energy_source1, 3, 4000)
    fitted_lattice2, fitted_free_energy2 = polynomially_fit_curve(lattice_source2, free_energy_source2, 3, 4000)

    lattice_sample1, free_energy_sample1 = read_lattice_free_energy_count(source_data1, sample_count)
    lattice_sample2, free_energy_sample2 = read_lattice_free_energy_count(source_data2, sample_count)

    if selected_data1 is not None:
        selected_lattice1, select_free_energy1 = specify_lattice_free_energy(selected_data1)
    if selected_data2 is not None:
        selected_lattice2, select_free_energy2 = specify_lattice_free_energy(selected_data2)

    # Minimum free energy and the corresponding lattice
    min_energy_index1 = free_energy_source1.index(min(free_energy_source1))
    min_lattice1 = lattice_source1[min_energy_index1]
    min_free_energy1 = min(free_energy_source1)

    min_energy_index2 = free_energy_source2.index(min(free_energy_source2))
    min_lattice2 = lattice_source1[min_energy_index2]
    min_free_energy2 = min(free_energy_source2)

    # Plotting 1
    plt.plot(fitted_lattice1, fitted_free_energy1, c=colors1[1], label=f"Fitted data of {matter1}", zorder=1)
    plt.scatter(lattice_sample1, free_energy_sample1, s=48, fc="#FFF", ec=colors1[1], label=f"Source data of {matter1}", zorder=1)
    plt.scatter(min_lattice1, min_free_energy1, s=48, ec=colors1[2], fc=colors1[2], label=f"Source lowest point of {matter1}", zorder=1)
    if selected_data1 is not None:
        plt.scatter(selected_lattice1,  select_free_energy1, s=24, ec=colors1[0], fc=colors1[0], label=f"Selected data of {matter1}", zorder=2)

    # Plotting 2
    plt.plot(fitted_lattice2, fitted_free_energy2, c=colors2[1], label=f"Fitted data of {matter2}", zorder=1)
    plt.scatter(lattice_sample2, free_energy_sample2, s=48, fc="#FFF", ec=colors2[1], label=f"Source data of {matter2}", zorder=1)
    plt.scatter(min_lattice2, min_free_energy2, s=48, ec=colors2[2], fc=colors2[2], label=f"Source lowest point of {matter2}", zorder=1)
    if selected_data2 is not None:
        plt.scatter(selected_lattice2,  select_free_energy2, s=24, ec=colors2[0], fc=colors2[0], label=f"Selected data of {matter2}", zorder=2)

    plt.legend(loc=fig_setting[3])
    plt.show()

def plot_lattice_free_energy_tri(title, sample_count,
                                 matter1, source_data1, color_family1,
                                 matter2, source_data2, color_family2,
                                 matter3, source_data3, color_family3,
                                 selected_data1=None, selected_data2=None, selected_data3=None):

    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    plt.title(f"Free energy versus lattice for {title}"); plt.xlabel(r"Lattice constant (Å)"); plt.ylabel(r"Energy (eV)")

    # Color calling
    colors1 = color_sampling(color_family1)
    colors2 = color_sampling(color_family2)
    colors3 = color_sampling(color_family3)

    # Data input
    lattice_source1, free_energy_source1 = read_lattice_free_energy_data(source_data1)
    lattice_source2, free_energy_source2 = read_lattice_free_energy_data(source_data2)
    lattice_source3, free_energy_source3 = read_lattice_free_energy_data(source_data3)

    fitted_lattice1, fitted_free_energy1 = polynomially_fit_curve(lattice_source1, free_energy_source1, 3, 4000)
    fitted_lattice2, fitted_free_energy2 = polynomially_fit_curve(lattice_source2, free_energy_source2, 3, 4000)
    fitted_lattice3, fitted_free_energy3 = polynomially_fit_curve(lattice_source3, free_energy_source3, 3, 4000)

    lattice_sample1, free_energy_sample1 = read_lattice_free_energy_count(source_data1, sample_count)
    lattice_sample2, free_energy_sample2 = read_lattice_free_energy_count(source_data2, sample_count)
    lattice_sample3, free_energy_sample3 = read_lattice_free_energy_count(source_data3, sample_count)

    if selected_data1 is not None:
        selected_lattice1, select_free_energy1 = specify_lattice_free_energy(selected_data1)
    if selected_data2 is not None:
        selected_lattice2, select_free_energy2 = specify_lattice_free_energy(selected_data2)
    if selected_data3 is not None:
        selected_lattice3, select_free_energy3 = specify_lattice_free_energy(selected_data3)

    # Minimum free energy and the corresponding lattice
    min_energy_index1 = free_energy_source1.index(min(free_energy_source1))
    min_lattice1 = lattice_source1[min_energy_index1]
    min_free_energy1 = min(free_energy_source1)

    min_energy_index2 = free_energy_source2.index(min(free_energy_source2))
    min_lattice2 = lattice_source1[min_energy_index2]
    min_free_energy2 = min(free_energy_source2)

    min_energy_index3 = free_energy_source3.index(min(free_energy_source3))
    min_lattice3 = lattice_source1[min_energy_index3]
    min_free_energy3 = min(free_energy_source3)

    # Plotting 1
    plt.plot(fitted_lattice1, fitted_free_energy1, c=colors1[1], label=f"Fitted data of {matter1}", zorder=1)
    plt.scatter(lattice_sample1, free_energy_sample1, s=48, fc="#FFF", ec=colors1[1], label=f"Source data of {matter1}", zorder=1)
    plt.scatter(min_lattice1, min_free_energy1, s=48, ec=colors1[2], fc=colors1[2], label=f"Source lowest point of {matter1}", zorder=1)
    if selected_data1 is not None:
        plt.scatter(selected_lattice1,  select_free_energy1, s=24, ec=colors1[0], fc=colors1[0], label=f"Selected data of {matter1}", zorder=2)

    # Plotting 2
    plt.plot(fitted_lattice2, fitted_free_energy2, c=colors2[1], label=f"Fitted data of {matter2}", zorder=1)
    plt.scatter(lattice_sample2, free_energy_sample2, s=48, fc="#FFF", ec=colors2[1], label=f"Source data of {matter2}", zorder=1)
    plt.scatter(min_lattice2, min_free_energy2, s=48, ec=colors2[2], fc=colors2[2], label=f"Source lowest point of {matter2}", zorder=1)
    if selected_data2 is not None:
        plt.scatter(selected_lattice2,  select_free_energy2, s=24, ec=colors2[0], fc=colors2[0], label=f"Selected data of {matter2}", zorder=2)

    # Plotting 3
    plt.plot(fitted_lattice3, fitted_free_energy3, c=colors3[1], label=f"Fitted data of {matter3}", zorder=1)
    plt.scatter(lattice_sample3, free_energy_sample3, s=48, fc="#FFF", ec=colors3[1], label=f"Source data of {matter3}", zorder=1)
    plt.scatter(min_lattice3, min_free_energy3, s=48, ec=colors3[2], fc=colors3[2], label=f"Source lowest point of {matter3}", zorder=1)
    if selected_data3 is not None:
        plt.scatter(selected_lattice3,  select_free_energy3, s=24, ec=colors3[0], fc=colors3[0], label=f"Selected data of {matter3}", zorder=2)

    plt.legend(loc=fig_setting[3])
    plt.show()

def plot_lattice_free_energy_quad(title, sample_count,
                                 matter1, source_data1, color_family1,
                                 matter2, source_data2, color_family2,
                                 matter3, source_data3, color_family3,
                                 matter4, source_data4, color_family4,
                                 selected_data1=None, selected_data2=None, selected_data3=None, selected_data4=None):

    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    plt.title(f"Free energy versus lattice for {title}"); plt.xlabel(r"Lattice constant (Å)"); plt.ylabel(r"Energy (eV)")

    # Color calling
    colors1 = color_sampling(color_family1)
    colors2 = color_sampling(color_family2)
    colors3 = color_sampling(color_family3)
    colors4 = color_sampling(color_family4)

    # Data input
    lattice_source1, free_energy_source1 = read_lattice_free_energy_data(source_data1)
    lattice_source2, free_energy_source2 = read_lattice_free_energy_data(source_data2)
    lattice_source3, free_energy_source3 = read_lattice_free_energy_data(source_data3)
    lattice_source4, free_energy_source4 = read_lattice_free_energy_data(source_data4)

    fitted_lattice1, fitted_free_energy1 = polynomially_fit_curve(lattice_source1, free_energy_source1, 3, 4000)
    fitted_lattice2, fitted_free_energy2 = polynomially_fit_curve(lattice_source2, free_energy_source2, 3, 4000)
    fitted_lattice3, fitted_free_energy3 = polynomially_fit_curve(lattice_source3, free_energy_source3, 3, 4000)
    fitted_lattice4, fitted_free_energy4 = polynomially_fit_curve(lattice_source4, free_energy_source4, 3, 4000)

    lattice_sample1, free_energy_sample1 = read_lattice_free_energy_count(source_data1, sample_count)
    lattice_sample2, free_energy_sample2 = read_lattice_free_energy_count(source_data2, sample_count)
    lattice_sample3, free_energy_sample3 = read_lattice_free_energy_count(source_data3, sample_count)
    lattice_sample4, free_energy_sample4 = read_lattice_free_energy_count(source_data4, sample_count)

    if selected_data1 is not None:
        selected_lattice1, select_free_energy1 = specify_lattice_free_energy(selected_data1)
    if selected_data2 is not None:
        selected_lattice2, select_free_energy2 = specify_lattice_free_energy(selected_data2)
    if selected_data3 is not None:
        selected_lattice3, select_free_energy3 = specify_lattice_free_energy(selected_data3)
    if selected_data4 is not None:
        selected_lattice4, select_free_energy4 = specify_lattice_free_energy(selected_data4)

    # Minimum free energy and the corresponding lattice
    min_energy_index1 = free_energy_source1.index(min(free_energy_source1))
    min_lattice1 = lattice_source1[min_energy_index1]
    min_free_energy1 = min(free_energy_source1)

    min_energy_index2 = free_energy_source2.index(min(free_energy_source2))
    min_lattice2 = lattice_source1[min_energy_index2]
    min_free_energy2 = min(free_energy_source2)

    min_energy_index3 = free_energy_source3.index(min(free_energy_source3))
    min_lattice3 = lattice_source1[min_energy_index3]
    min_free_energy3 = min(free_energy_source3)

    min_energy_index4 = free_energy_source4.index(min(free_energy_source4))
    min_lattice4 = lattice_source1[min_energy_index4]
    min_free_energy4 = min(free_energy_source4)

    # Plotting 1
    # plt.plot(fitted_lattice1, fitted_free_energy1, c=colors1[1], label=f"Fitted data of {matter1}", zorder=1)
    plt.plot(fitted_lattice1, fitted_free_energy1, c=colors1[1], zorder=1)
    plt.scatter(lattice_sample1, free_energy_sample1, s=48, fc="#FFF", ec=colors1[1], label=f"Source data of {matter1}", zorder=1)
    plt.scatter(min_lattice1, min_free_energy1, s=48, ec=colors1[2], fc=colors1[2], label=f"Source lowest point of {matter1}", zorder=1)
    if selected_data1 is not None:
        plt.scatter(selected_lattice1,  select_free_energy1, s=24, ec=colors1[0], fc=colors1[0], label=f"Selected data of {matter1}", zorder=2)

    # Plotting 2
    # plt.plot(fitted_lattice2, fitted_free_energy2, c=colors2[1], label=f"Fitted data of {matter2}", zorder=1)
    plt.plot(fitted_lattice2, fitted_free_energy2, c=colors2[1], zorder=1)
    plt.scatter(lattice_sample2, free_energy_sample2, s=48, fc="#FFF", ec=colors2[1], label=f"Source data of {matter2}", zorder=1)
    plt.scatter(min_lattice2, min_free_energy2, s=48, ec=colors2[2], fc=colors2[2], label=f"Source lowest point of {matter2}", zorder=1)
    if selected_data2 is not None:
        plt.scatter(selected_lattice2,  select_free_energy2, s=24, ec=colors2[0], fc=colors2[0], label=f"Selected data of {matter2}", zorder=2)

    # Plotting 3
    # plt.plot(fitted_lattice3, fitted_free_energy3, c=colors3[1], label=f"Fitted data of {matter3}", zorder=1)
    plt.plot(fitted_lattice3, fitted_free_energy3, c=colors3[1], zorder=1)
    plt.scatter(lattice_sample3, free_energy_sample3, s=48, fc="#FFF", ec=colors3[1], label=f"Source data of {matter3}", zorder=1)
    plt.scatter(min_lattice3, min_free_energy3, s=48, ec=colors3[2], fc=colors3[2], label=f"Source lowest point of {matter3}", zorder=1)
    if selected_data3 is not None:
        plt.scatter(selected_lattice3,  select_free_energy3, s=24, ec=colors3[0], fc=colors3[0], label=f"Selected data of {matter3}", zorder=2)

    # Plotting 4
    # plt.plot(fitted_lattice4, fitted_free_energy4, c=colors4[1], label=f"Fitted data of {matter4}", zorder=1)
    plt.plot(fitted_lattice4, fitted_free_energy4, c=colors4[1], zorder=1)
    plt.scatter(lattice_sample4, free_energy_sample4, s=48, fc="#FFF", ec=colors4[1], label=f"Source data of {matter4}", zorder=1)
    plt.scatter(min_lattice4, min_free_energy4, s=48, ec=colors4[2], fc=colors4[2], label=f"Source lowest point of {matter4}", zorder=1)
    if selected_data4 is not None:
        plt.scatter(selected_lattice4,  select_free_energy4, s=24, ec=colors4[0], fc=colors4[0], label=f"Selected data of {matter4}", zorder=2)

    plt.legend(loc=fig_setting[3])
    plt.show()

lattice_free_energy_ploting_help_info = "help information"

def plot_lattice_free_energy(matter_count, *args):
    if matter_count == "help":
        print(lattice_free_energy_ploting_help_info)
    if matter_count == 1:
        if len(args) == 4:
            return plot_lattice_free_energy_solo(args[0], args[1], args[2], args[3])
        elif len(args) == 5:
            return plot_lattice_free_energy_solo(args[0], args[1], args[2], args[3], args[4])
    if matter_count == 2:
        if len(args) == 8:
            return plot_lattice_free_energy_duo(args[0], args[1], args[2], args[3],
                                                args[4], args[5], args[6], args[7])
        elif len(args) == 10:
            return plot_lattice_free_energy_duo(args[0], args[1], args[2], args[3],
                                                args[4], args[5], args[6], args[7],
                                                args[8], args[9])
    if matter_count == 3:
        if len(args) == 11:
            return plot_lattice_free_energy_tri(args[0], args[1], args[2], args[3],
                                                args[4], args[5], args[6], args[7],
                                                args[8], args[9], args[10])
        elif len(args) == 14:
            return plot_lattice_free_energy_tri(args[0], args[1], args[2], args[3],
                                                args[4], args[5], args[6], args[7],
                                                args[8], args[9], args[10],args[11],
                                                args[12], args[13])
    if matter_count == 4:
        if len(args) == 14:
            return plot_lattice_free_energy_quad(args[0], args[1], args[2], args[3],
                                                 args[4], args[5], args[6], args[7],
                                                 args[8], args[9], args[10],args[11],
                                                 args[12], args[13])
        elif len(args) == 18:
            return plot_lattice_free_energy_quad(args[0], args[1], args[2], args[3],
                                                 args[4], args[5], args[6], args[7],
                                                 args[8], args[9], args[10],args[11],
                                                 args[12], args[13],args[14], args[15],
                                                 args[16], args[17])

def plot_biolayer_lattice(matter, source_data, colormap, point_color, additional_data=None):
    # Data input
    lattice_source, distance_source, free_energy_source = read_biolayer_lattice_data(source_data)
    lattice_source = np.array(lattice_source)
    distance_source = np.array(distance_source)
    free_energy_source = np.array(free_energy_source)

    # Extrema of source data
    energy_min = np.min(free_energy_source)
    energy_max = np.max(free_energy_source)
    energy_range = energy_max - energy_min
    energy_demo = energy_min + energy_range * 0.125
    lattice_min = lattice_source [np.argmin(free_energy_source)]
    distance_min = distance_source [np.argmin(free_energy_source)]

    # Data grid
    lattice_fine = np.linspace(lattice_source.min(), lattice_source.max(), 1024)
    distance_fine = np.linspace(distance_source.min(), distance_source.max(), 1024)
    lattice_grid_fine, distance_grid_fine = np.meshgrid(lattice_fine, distance_fine)

    # Interpolate using the "cubic" method
    free_energy_grid_fine = griddata((lattice_source, distance_source), free_energy_source, (lattice_grid_fine, distance_grid_fine), method="linear")

    # Fitted data
    Fitted_data = extract_minimum_biolayer_lattice(source_data)
    lattice_fitted_min = Fitted_data[0]
    distance_fitted_min = Fitted_data[1]
    # free_energy_fitted_min = Fitted_data[-1]

    # Additional data
    if additional_data is not None:
        additional_lattice = additional_data[0]
        additional_distance = additional_data[1]
        # additional_energy = additional_data[-1]

    # Settings input
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Color calling
    colors = color_sampling(point_color)

    # Figure title
    plt.title(f"Free energy versus lattice and distance for {matter}")
    plt.xlabel(r"Lattice constant (Å)"); plt.ylabel(r"Interlayer spacing (Å)")

    cp = plt.pcolormesh(lattice_grid_fine, distance_grid_fine, free_energy_grid_fine, shading="auto", cmap=colormap, alpha = 0.75, vmax = energy_demo, zorder=1)
    plt.colorbar(cp)

    # Extreme of source data
    plt.scatter(lattice_min, distance_min, s=48, c=colors[2], label="Extrema of source data", zorder=2)
    # Extreme of fitted data
    plt.scatter(lattice_fitted_min, distance_fitted_min, s=48, lw=1.5, facecolors="none", ec=colors[2], label="Extrema of fitted data", zorder=2)
    # Additional point
    if additional_data is not None:
        plt.scatter(additional_lattice, additional_distance, s=36, c=colors[3], label="Extrema of selected data", zorder=2)

    plt.legend(loc=fig_setting[3])
