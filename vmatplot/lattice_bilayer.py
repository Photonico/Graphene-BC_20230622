#### Free energy extracting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from matplotlib.ticker import ScalarFormatter
from vmatplot.output import canvas_setting, color_sampling
from vmatplot.lattice import check_vasprun
from vmatplot.algorithms import polynomially_fit_surface

def specify_bilayer_lattice(directory):
    if directory == "help":
        print("Please use this function on the directory of the specific work folder and.")
        return []

    xml_path = os.path.join(directory, "vasprun.xml")
    contcar_path = os.path.join(directory, "CONTCAR")

    if os.path.isfile(xml_path) and os.path.isfile(contcar_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract free energy
            free_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)

            # Extract lattice constant
            # Assuming the lattice constant is the length of the 'a' vector from the final structure
            basis_vectors = root.findall(".//calculation/structure/crystal/varray[@name='basis']")[-1]
            a_vector = basis_vectors[0].text.split()
            lattice = (float(a_vector[0])**2 + float(a_vector[1])**2 + float(a_vector[2])**2)**0.5

            coordinates = []
            with open (contcar_path, "r", encoding="utf-8") as contcar_file:
                lines = contcar_file.readlines()
                distance_bound = float(lines[4].split()[-1])
                for index, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in ["cartesian", "cart", "direct", "fractional"]):
                        coord_start_index = index + 1
                        break
                else: raise ValueError("Direct or Cartesian keyword not found in CONTCAR file.")
                total_atoms = sum([int(count) for count in lines[coord_start_index - 2].split()])
                coord_end_index = coord_start_index + total_atoms

                coordinates = lines[coord_start_index: coord_end_index]; top_layer = []; bottom_layer = []
                z_values = [float(coord.split()[-1]) for coord in coordinates]; z_average = np.mean(z_values)
                for coord in coordinates:
                    _, _, z = map(float, coord.split())
                    if z < z_average:
                        bottom_layer.append(coord)
                    else:
                        top_layer.append(coord)
                z_values_bottom = [float(coord.split()[-1]) for coord in bottom_layer]; z_average_bottom = np.mean(z_values_bottom)
                z_values_top = [float(coord.split()[-1]) for coord in top_layer]; z_average_top = np.mean(z_values_top)
                if any(keyword in line.lower() for keyword in ["cartesian", "cart"]):
                    distance = z_average_top - z_average_bottom
                if any(keyword in line.lower() for keyword in ["direct", "fractional"]):
                    distance = (z_average_top - z_average_bottom) * distance_bound
            return lattice, distance, free_energy

        except ET.ParseError as e:
            print("Error parsing vasprun.xml:", e)
            return None
    else:
        print("vasprun.xml or CONTCAR is not found in the current directory")
        return None

def summarize_bilayer_lattice(directory=".", lattice_start = None, lattice_end = None, distance_start = None, distance_end = None):
    result_file = "lattice_distance.dat"
    result_file_path = os.path.join(directory, result_file)

    if directory == "help":
        print("Please use this function on the parent directory of the project's main folder.")
        return []

    # Use check_vasprun to get folders with complete vasprun.xml
    dirs_to_walk = check_vasprun(directory)
    results = []

    for work_dir in dirs_to_walk:
        xml_path = os.path.join(work_dir, "vasprun.xml")
        contcar_path = os.path.join(work_dir, "CONTCAR")

        if os.path.isfile(xml_path) and os.path.isfile(contcar_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Extract free energy
                free_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)

                # Extract lattice constant
                # Assuming the lattice constant is the length of the 'a' vector from the final structure
                basis_vectors = root.findall(".//calculation/structure/crystal/varray[@name='basis']")[-1]
                a_vector = basis_vectors[0].text.split()
                a_length = (float(a_vector[0])**2 + float(a_vector[1])**2 + float(a_vector[2])**2)**0.5
                lattice_constant = a_length

                coordinates = []
                with open (contcar_path, "r", encoding="utf-8") as contcar_file:
                    lines = contcar_file.readlines()
                    distance_bound = float(lines[4].split()[-1])
                    for index, line in enumerate(lines):
                        if any(keyword in line.lower() for keyword in ["cartesian", "cart", "direct", "fractional"]):
                            coord_start_index = index + 1
                            break
                    else: raise ValueError("Direct or Cartesian keyword not found in CONTCAR file.")
                    total_atoms = sum([int(count) for count in lines[coord_start_index - 2].split()])
                    coord_end_index = coord_start_index + total_atoms

                    coordinates = lines[coord_start_index: coord_end_index]; top_layer = []; bottom_layer = []
                    z_values = [float(coord.split()[-1]) for coord in coordinates]; z_average = np.mean(z_values)
                    for coord in coordinates:
                        _, _, z = map(float, coord.split())
                        if z < z_average:
                            bottom_layer.append(coord)
                        else:
                            top_layer.append(coord)
                    z_values_bottom = [float(coord.split()[-1]) for coord in bottom_layer]; z_average_bottom = np.mean(z_values_bottom)
                    z_values_top = [float(coord.split()[-1]) for coord in top_layer]; z_average_top = np.mean(z_values_top)
                    if any(keyword in line.lower() for keyword in ["cartesian", "cart"]):
                        distance = z_average_top - z_average_bottom
                    if any(keyword in line.lower() for keyword in ["direct", "fractional"]):
                        distance = (z_average_top - z_average_bottom) * distance_bound

                # Check if lattice constant is within the specified range
                TOLERANCE = 1e-6
                lattice_within_start = lattice_start is None or lattice_constant >= lattice_start - TOLERANCE
                lattice_within_end = lattice_end is None or lattice_constant <= lattice_end + TOLERANCE
                distance_within_start = distance_start is None or distance >= distance_start - TOLERANCE
                distance_within_end = distance_end is None or distance <= distance_end + TOLERANCE

                if lattice_within_start and lattice_within_end and distance_within_start and distance_within_end:
                    results.append((lattice_constant, distance, free_energy))

            except (ET.ParseError, ValueError, IndexError) as e:
                print(f"Error processing {work_dir}: {e}")

        else:
            print(f"vasprun.xml or CONTCAR is not found in {work_dir}.")

    # Sort the results by lattice_constant (the first element of the tuple)
    results.sort(key=lambda x: x[0])

    # Now write the sorted results to the file
    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            f.write("Lattice\t Distance\t Free energy\n")
            for lattice_constant, distance, free_energy in results:
                f.write(f"{lattice_constant}\t{distance}\t{free_energy}\n")

    except IOError as e:
        print("Error writing to file:", e)

def read_bilayer_lattice_data(lattice_path, density = 1):
    help_info = "Usage: read_bilayer_lattice_data(lattice_path, density)\n" + \
                "lattice_path: Path to the data file containing lattice and free energy values.\n" + \
                "density: Sampling density. For instance, use 1 for 100% sampling (default), 0.1 for 10%, and 0.5 for 50%."

    # Check if the user asked for help
    if lattice_path == "help":
        print(help_info)
        return

    # Initialize the lists for lattice constant and free energy
    lattice, distance, free_energy = [], [], []

    # Calculate the interval based on density
    interval = int(1 / density) if density != 0 else 1
    with open(lattice_path, "r", encoding="utf-8") as dat_file:
        lines = dat_file.readlines()[1:]
        for idx, line in enumerate(lines):
            # Only process lines based on the specified interval
            if idx % interval == 0:
                split_line = line.strip().split()
                lattice_value = float(split_line[0])
                lattice.append(lattice_value)
                distance_value = float(split_line[1])
                distance.append(distance_value)
                free_energy_value = float(split_line[-1])
                free_energy.append(free_energy_value)
    return lattice, distance, free_energy

def extract_minimum_bilayer_lattice(source_data):
    lattice_source, distance_source, free_energy_source = read_bilayer_lattice_data(source_data)
    lattice_fitted, distance_fitted, free_energy_fitted = polynomially_fit_surface(lattice_source, distance_source, free_energy_source, 3, 4000)

    # Find the index of the minimum value in the fittede data
    min_index = np.argmin(free_energy_fitted)

    # Use this index to get the minimum value of fitted_energy and its corresponding fitted_lattice value
    free_energy_fitted_min = free_energy_fitted[min_index]
    lattice_fitted_min = lattice_fitted[min_index]
    distance_fitted_min = distance_fitted[min_index]

    return lattice_fitted_min, distance_fitted_min, free_energy_fitted_min

def extract_maximum_bilayer_lattice(source_data):
    lattice_source, distance_source, free_energy_source = read_bilayer_lattice_data(source_data)
    lattice_fitted, distance_fitted, free_energy_fitted = polynomially_fit_surface(lattice_source, distance_source, free_energy_source, 3, 4000)

    # Find the index of the maximum value in the fittede data
    max_index = np.argmax(free_energy_fitted)

    # Use this index to get the maximum value of fitted_energy and its corresponding fitted_lattice value
    free_energy_fitted_max = free_energy_fitted[max_index]
    lattice_fitted_max = lattice_fitted[max_index]
    distance_fitted_max = distance_fitted[max_index]

    return lattice_fitted_max, distance_fitted_max, free_energy_fitted_max

def extract_extreme_bilayer_lattice(extract_type, source_data):
    if extract_type in ("Minium", "minium", "MIN", "Min", "min"):
        return extract_minimum_bilayer_lattice(source_data)
    if extract_type in ("Maxium", "maxium", "MAX", "Max", "max"):
        return extract_maximum_bilayer_lattice(source_data)

def plot_bilayer_lattice_single(suptitle, source_data, colormap, point_color, additional_work=None, legend_loc="upper right"):
    # Data input
    lattice_source, distance_source, free_energy_source = read_bilayer_lattice_data(source_data)
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
    Fitted_data = extract_minimum_bilayer_lattice(source_data)
    lattice_fitted_min = Fitted_data[0]
    distance_fitted_min = Fitted_data[1]
    # free_energy_fitted_min = Fitted_data[-1]

    # Additional data
    if additional_work is not None:
        additional_data = specify_bilayer_lattice(additional_work)
        additional_lattice = additional_data[0]
        additional_distance = additional_data[1]
        additional_energy = additional_data[-1]

    # Settings input
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Color calling
    colors = color_sampling(point_color)

    # Figure title
    # plt.title(f"Free energy versus lattice and spacing for {matter}")
    plt.title(f"Free energy {suptitle}")
    plt.xlabel(r"Lattice constant (Å)"); plt.ylabel(r"Interlayer spacing (Å)")

    # Colormap
    cp = plt.pcolormesh(lattice_grid_fine, distance_grid_fine, free_energy_grid_fine, shading="auto", cmap=colormap, alpha = 0.75, vmax = energy_demo, zorder=1)
    cbar = plt.colorbar(cp)
    # plt.colorbar(cp)
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    cbar.ax.yaxis.set_major_formatter(formatter)

    # Extreme of source data
    plt.scatter(lattice_min, distance_min, s=48, c=colors[2], label="Extrema of source data", zorder=2)
    # Extreme of fitted data
    plt.scatter(lattice_fitted_min, distance_fitted_min, s=48, lw=1.5, facecolors="none", ec=colors[2], label="Extrema of fitted data", zorder=2)
    # Additional point
    if additional_work is not None:
        plt.scatter(additional_lattice, additional_distance, s=36, c=colors[3], label=f"Specific energy: {additional_energy:.3f} (eV)", zorder=3)

    plt.legend(loc=legend_loc)
    plt.tight_layout()

def plot_bilayer_lattice_double(suptitle, subtitle_1, subtitle_2, source_data_1, source_data_2, colormap_1, colormap_2, point_color_1, point_color_2, additional_work_1=None, additional_work_2=None, legend_loc="upper right"):
    # figure Settings
    fig_setting = canvas_setting(16, 6)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # colors calling
    annotate_color = color_sampling("Grey")
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    # order_labels = ["a","b"]

    # data sets
    source_data_set = [source_data_1, source_data_2]
    colormap_set = [colormap_1, colormap_2]
    point_color_set = [point_color_1, point_color_2]
    additional_set = [additional_work_1, additional_work_2]
    subtitles = [subtitle_1, subtitle_2]
    order_labels = subtitles

    # Title
    plt.suptitle(f"Free energy versus lattice and spacing {suptitle}", fontsize=fig_setting[3][0], y=1.00)

    # plot data
    for subplot_index in range(2):
        ax = axes_element[subplot_index]
        # Data input
        lattice_source, distance_source, free_energy_source = read_bilayer_lattice_data(source_data_set[subplot_index])
        lattice_source = np.array(lattice_source)
        distance_source = np.array(distance_source)
        free_energy_source = np.array(free_energy_source)

        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

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
        Fitted_data = extract_minimum_bilayer_lattice(source_data_set[subplot_index])
        lattice_fitted_min = Fitted_data[0]
        distance_fitted_min = Fitted_data[1]
        # free_energy_fitted_min = Fitted_data[-1]

        # Additional data
        if additional_set[subplot_index] is not None:
            additional_data = specify_bilayer_lattice(additional_set[subplot_index])
            additional_lattice = additional_data[0]
            additional_distance = additional_data[1]
            additional_energy = additional_data[-1]

        # Colormap
        cp = ax.pcolormesh(lattice_grid_fine, distance_grid_fine, free_energy_grid_fine, shading="auto", cmap=colormap_set[subplot_index], alpha = 0.75, vmax = energy_demo, zorder=1)
        cbar = fig.colorbar(cp, ax=ax)
        formatter = ScalarFormatter(useMathText=True, useOffset=False)
        formatter.set_powerlimits((-3, 3))
        cbar.ax.yaxis.set_major_formatter(formatter)
        colors = color_sampling(point_color_set[subplot_index])

        # Extreme of source data
        ax.scatter(lattice_min, distance_min, s=48, color=colors[2], label="Extrema of source data", zorder=2)
        # Extreme of fitted data
        ax.scatter(lattice_fitted_min, distance_fitted_min, s=48, lw=1.5, facecolors="none", edgecolors=colors[2], label="Extrema of fitted data", zorder=2)
        # Additional point
        if additional_set[subplot_index] is not None:
            ax.scatter(additional_lattice, additional_distance, s=36, color=colors[3], label=f"Specific energy: {additional_energy:.3f} (eV)", zorder=3)

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

        if lattice_min < np.mean(lattice_source) and distance_min < np.mean(distance_source):
            legend_loc = "lower right"
        else:
            legend_loc = "lower left"

        ax.set_xlabel(r"Lattice constant (Å)")
        if subplot_index == 0:
            ax.set_ylabel(r"Interlayer spacing (Å)")

        ax.legend(loc=legend_loc)

    plt.tight_layout()

def plot_bilayer_lattice_triple_row(suptitle, subtitle_1, subtitle_2, subtitle_3, source_data_1, source_data_2, source_data_3,
                                    colormap_1, colormap_2, colormap_3, point_color_1, point_color_2, point_color_3,
                                    additional_work_1=None, additional_work_2=None, additional_work_3=None,
                                    legend_loc="upper right"):
    # figure Settings
    fig_setting = canvas_setting(24, 6)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 3, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1], axs[2]]

    # colors calling
    annotate_color = color_sampling("Grey")
    # order_labels = ["a","b","c"]

    # data sets
    source_data_set = [source_data_1, source_data_2, source_data_3]
    colormap_set = [colormap_1, colormap_2, colormap_3]
    point_color_set = [point_color_1, point_color_2, point_color_3]
    additional_set = [additional_work_1, additional_work_2, additional_work_3]
    subtitles = [subtitle_1, subtitle_2, subtitle_3]
    order_labels = subtitles

    # Title
    plt.suptitle(f"Free energy versus lattice and spacing {suptitle}", fontsize=fig_setting[3][0], y=1.00)

    for subplot_index in range(3):
        ax = axes_element[subplot_index]
        # Data input
        lattice_source, distance_source, free_energy_source = read_bilayer_lattice_data(source_data_set[subplot_index])
        lattice_source = np.array(lattice_source)
        distance_source = np.array(distance_source)
        free_energy_source = np.array(free_energy_source)

        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

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
        Fitted_data = extract_minimum_bilayer_lattice(source_data_set[subplot_index])
        lattice_fitted_min = Fitted_data[0]
        distance_fitted_min = Fitted_data[1]
        # free_energy_fitted_min = Fitted_data[-1]

        # Additional data
        if additional_set[subplot_index] is not None:
            additional_data = specify_bilayer_lattice(additional_set[subplot_index])
            additional_lattice = additional_data[0]
            additional_distance = additional_data[1]
            additional_energy = additional_data[-1]

        # Colormap
        cp = ax.pcolormesh(lattice_grid_fine, distance_grid_fine, free_energy_grid_fine, shading="auto", cmap=colormap_set[subplot_index], alpha = 0.75, vmax = energy_demo, zorder=1)
        cbar = fig.colorbar(cp, ax=ax)
        formatter = ScalarFormatter(useMathText=True, useOffset=False)
        formatter.set_powerlimits((-3, 3))
        cbar.ax.yaxis.set_major_formatter(formatter)
        colors = color_sampling(point_color_set[subplot_index])

        # Extreme of source data
        ax.scatter(lattice_min, distance_min, s=48, color=colors[2], label="Extrema of source data", zorder=2)
        # Extreme of fitted data
        ax.scatter(lattice_fitted_min, distance_fitted_min, s=48, lw=1.5, facecolors="none", edgecolors=colors[2], label="Extrema of fitted data", zorder=2)
        # Additional point
        if additional_set[subplot_index] is not None:
            ax.scatter(additional_lattice, additional_distance, s=36, color=colors[3], label=f"Specific energy: {additional_energy:.3f} (eV)", zorder=3)

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

        if lattice_min < np.mean(lattice_source) and distance_min < np.mean(distance_source):
            legend_loc = "lower right"
        else:
            legend_loc = "lower left"

        ax.set_xlabel(r"Lattice constant (Å)")
        if subplot_index == 0:
            ax.set_ylabel(r"Interlayer spacing (Å)")

        ax.legend(loc=legend_loc)

    plt.tight_layout()

def plot_bilayer_lattice_quadruple(suptitle,
                                   subtitle_1, subtitle_2, subtitle_3, subtitle_4,
                                   source_data_1, source_data_2, source_data_3, source_data_4,
                                   colormap_1, colormap_2, colormap_3, colormap_4,
                                   point_color_1, point_color_2, point_color_3, point_color_4,
                                   additional_work_1=None, additional_work_2=None, additional_work_3=None, additional_work_4=None,
                                   legend_loc="upper right"):

    # figure Settings
    fig_setting = canvas_setting(16, 12)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

    # Colors calling
    annotate_color = color_sampling("Grey")
    # order_labels = ["a","b","c","d"]

    # data sets
    source_data_set = [source_data_1, source_data_2, source_data_3, source_data_4]
    colormap_set = [colormap_1, colormap_2, colormap_3, colormap_4]
    point_color_set = [point_color_1, point_color_2, point_color_3, point_color_4]
    additional_set = [additional_work_1, additional_work_2, additional_work_3, additional_work_4]
    subtitles = [subtitle_1, subtitle_2, subtitle_3, subtitle_4]
    order_labels = subtitles

    # Title
    plt.suptitle(f"Free energy versus lattice and spacing {suptitle}", fontsize=fig_setting[3][0], y=1.00)

    # plot data
    for subplot_index in range(4):
        ax = axes_element[subplot_index]
        # Data input
        lattice_source, distance_source, free_energy_source = read_bilayer_lattice_data(source_data_set[subplot_index])
        lattice_source = np.array(lattice_source)
        distance_source = np.array(distance_source)
        free_energy_source = np.array(free_energy_source)

        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

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
        Fitted_data = extract_minimum_bilayer_lattice(source_data_set[subplot_index])
        lattice_fitted_min = Fitted_data[0]
        distance_fitted_min = Fitted_data[1]
        # free_energy_fitted_min = Fitted_data[-1]

        # Additional data
        if additional_set[subplot_index] is not None:
            additional_data = specify_bilayer_lattice(additional_set[subplot_index])
            additional_lattice = additional_data[0]
            additional_distance = additional_data[1]
            additional_energy = additional_data[-1]

        # Colormap
        cp = ax.pcolormesh(lattice_grid_fine, distance_grid_fine, free_energy_grid_fine, shading="auto", cmap=colormap_set[subplot_index], alpha = 0.75, vmax = energy_demo, zorder=1)
        cbar = fig.colorbar(cp, ax=ax)
        formatter = ScalarFormatter(useMathText=True, useOffset=False)
        formatter.set_powerlimits((-3, 3))
        cbar.ax.yaxis.set_major_formatter(formatter)
        colors = color_sampling(point_color_set[subplot_index])

        # Extreme of source data
        ax.scatter(lattice_min, distance_min, s=48, color=colors[2], label="Extrema of source data", zorder=2)
        # Extreme of fitted data
        ax.scatter(lattice_fitted_min, distance_fitted_min, s=48, lw=1.5, facecolors="none", edgecolors=colors[2], label="Extrema of fitted data", zorder=2)
        # Additional point
        if additional_set[subplot_index] is not None:
            ax.scatter(additional_lattice, additional_distance, s=36, color=colors[3], label=f"Specific energy: {additional_energy:.3f} (eV)", zorder=3)

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

        if lattice_min < np.mean(lattice_source) and distance_min < np.mean(distance_source):
            legend_loc = "lower right"
        else:
            legend_loc = "lower left"

        if subplot_index >= 2:
            ax.set_xlabel(r"Lattice constant (Å)")
        if subplot_index %2 ==0:
            ax.set_ylabel(r"Interlayer spacing (Å)")
        
        ax.legend(loc=legend_loc)

    plt.tight_layout()

def plot_bilayer_lattice(subfigures_amount, suptitle, *data_sets):
    help_info = "Usage: plot_free_energy_lattice \n" + \
                "Use summarize_free_energy_lattice_directory to extract the free energy versus lattice into free_energy_lattice.dat firstly.\n"

    if subfigures_amount == 1:
        args = tuple(data_sets[0])
        return plot_bilayer_lattice_single(suptitle, *args)

    elif subfigures_amount == 2:
        data_set = data_sets
        args = [suptitle,
                data_set[0][0], data_set[1][0],
                data_set[0][1], data_set[1][1],
                data_set[0][2], data_set[1][2],
                data_set[0][3], data_set[1][3]]
        if len(data_set[0]) > 4:
            args.append(data_set[0][4])
        if len(data_set[1]) > 4:
            args.append(data_set[1][4])
        return plot_bilayer_lattice_double(*args)

    elif subfigures_amount == 3:
        data_set = data_sets
        args = [suptitle,
                data_set[0][0], data_set[1][0], data_set[2][0],
                data_set[0][1], data_set[1][1], data_set[2][1],
                data_set[0][2], data_set[1][2], data_set[2][2],
                data_set[0][3], data_set[1][3], data_set[2][3]]
        if len(data_set[0]) > 4:
            args.append(data_set[0][4])
        if len(data_set[1]) > 4:
            args.append(data_set[1][4])
        if len(data_set[2]) > 4:
            args.append(data_set[2][4])
        return plot_bilayer_lattice_triple_row(*args)

    elif subfigures_amount == 4:
        data_set = data_sets
        args = [suptitle,
                data_set[0][0], data_set[1][0], data_set[2][0], data_set[3][0],
                data_set[0][1], data_set[1][1], data_set[2][1], data_set[3][1],
                data_set[0][2], data_set[1][2], data_set[2][2], data_set[3][2],
                data_set[0][3], data_set[1][3], data_set[2][3], data_set[3][3]]
        if len(data_set[0]) > 4:
            args.append(data_set[0][4])
        if len(data_set[1]) > 4:
            args.append(data_set[1][4])
        if len(data_set[2]) > 4:
            args.append(data_set[2][4])
        if len(data_set[3]) > 4:
            args.append(data_set[3][4])
        return plot_bilayer_lattice_quadruple(*args)

    # help information
    else:
        print(help_info)
