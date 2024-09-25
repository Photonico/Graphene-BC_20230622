#### Kpoints
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import xml.etree.ElementTree as ET
import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import ScalarFormatter
from vmatplot.lattice import check_vasprun
from vmatplot.output import canvas_setting, color_sampling

def identify_kpoints(directory="."):
    """Find folders with KPOINTS and print its type."""
    # Key words
    automatic = "Automatic k-point grid"
    explicit = "Explicit k-points listed"
    linear = "Linear mode"

    # Check if the user asked for help
    if directory == "help":
        print("Please use this function on the project directory.")
        return "Help provided."

    kpoints_path = os.path.join(directory, "KPOINTS")
    if not os.path.exists(kpoints_path):
        return "KPOINTS file not found in the specified directory."

    with open(kpoints_path, "r", encoding="utf-8") as kpoints_file:
        lines = kpoints_file.readlines()
        if len(lines) < 3:
            return "Invalid file format, unable to identify"
        second_line = lines[1].strip()
        third_line = lines[2].strip()
        if second_line == "0":
            if "Gamma" in third_line or "Monkhorst" in third_line:
                return automatic
            else: return "Invalid file format, unable to identify"
        elif second_line.isdigit():
            if "Explicit" in third_line:
                return explicit
            elif "Line-mode" in third_line:
                return linear
            else: return "Invalid file format, unable to identify"
        else: return "Invalid file format, unable to identify"

def specify_kpoints_free_energy(directory):
    if directory == "help":
        print("Please use this function on the directory of the specific work folder.")
        return []

    xml_path = os.path.join(directory, "vasprun.xml")
    kpoints_path = os.path.join(directory, "KPOINTS")

    if os.path.isfile(xml_path) and os.path.isfile(kpoints_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract free energy
            free_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)

            # Extract lattice constant
            # Assuming the lattice constant is the length of the "a" vector from the final structure
            basis_vectors = root.findall(".//calculation/structure/crystal/varray[@name='basis']")[-1]
            a_vector = basis_vectors[0].text.split()
            lattice_constant = (float(a_vector[0])**2 + float(a_vector[1])**2 + float(a_vector[2])**2)**0.5

            with open (kpoints_path, "r", encoding="utf-8") as kpoints_file:
                lines = kpoints_file.readlines()
                for index, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in ["gamma","explicit","line-mode"]):
                        kpoints_index = index + 1
                        break
                else: raise ValueError("Kpoints type keyword not found in KPOINTS file.")

                kpoints_values = lines[kpoints_index].split()
                x_kpoints = int(kpoints_values[0])
                y_kpoints = int(kpoints_values[1])
                z_kpoints = int(kpoints_values[2])
                tot_kpoints = x_kpoints * y_kpoints * z_kpoints

            return tot_kpoints, (x_kpoints,y_kpoints,z_kpoints), lattice_constant, free_energy

        except (ET.ParseError, ValueError, IndexError) as e:
            print("Error parsing vasprun.xml:", e)
            return None
    else:
        print("vasprun.xml or KPOINTS is not found in the current directory")
        return None

def summarize_kpoints_free_energy(directory=".", lattice_boundary = None):
    result_file = "energy_kpoint.dat"
    result_file_path = os.path.join(directory, result_file)

    if directory == "help":
        print("Please use this function on the parent directory of the project's main folder.")
        return []

    dirs_to_walk = check_vasprun(directory)
    results = []

    lattice_within_start = True
    lattice_within_end = True

    for work_dir in dirs_to_walk:
        xml_path = os.path.join(work_dir, "vasprun.xml")
        kpoints_path = os.path.join(work_dir, "KPOINTS")

        if os.path.isfile(xml_path) and os.path.isfile(kpoints_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                free_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)
                basis_vectors = root.findall(".//calculation/structure/crystal/varray[@name='basis']")[-1]
                a_vector = basis_vectors[0].text.split()
                lattice_constant = (float(a_vector[0])**2 + float(a_vector[1])**2 + float(a_vector[2])**2)**0.5

                with open(kpoints_path, "r", encoding="utf-8") as kpoints_file:
                    lines = kpoints_file.readlines()
                    for index, line in enumerate(lines):
                        if any(keyword in line.lower() for keyword in ["gamma", "explicit", "line-mode"]):
                            kpoints_index = index + 1
                            break
                    else:
                        raise ValueError("Kpoints type keyword not found in KPOINTS file.")

                    kpoints_values = lines[kpoints_index].split()
                    x_kpoints = int(kpoints_values[0])
                    y_kpoints = int(kpoints_values[1])
                    z_kpoints = int(kpoints_values[2])
                    tot_kpoints = x_kpoints * y_kpoints * z_kpoints

                TOLERANCE = 1e-6
                if lattice_boundary is not None:
                    lattice_start, lattice_end = lattice_boundary
                    lattice_within_start = lattice_start in [None, ""] or lattice_constant >= lattice_start - TOLERANCE
                    lattice_within_end = lattice_end in [None, ""] or lattice_constant <= lattice_end + TOLERANCE

                if lattice_within_start and lattice_within_end:
                    results.append((tot_kpoints, (x_kpoints, y_kpoints, z_kpoints), lattice_constant, free_energy))

            except (ET.ParseError, ValueError, IndexError) as e:
                print(f"Error parsing vasprun.xml or KPOINTS in {work_dir}:", e)

        else:
            print(f"vasprun.xml or KPOINTS is not found in {work_dir}.")

    # Sort the results by total kpoints, then by lattice constant if you prefer
    results.sort(key=lambda x: x[0])

    # Now write the sorted results to the file
    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            f.write("Total Kpoints\tKpoints(X Y Z)\tLattice Constant\tFree Energy\n")
            for tot_kpoints, kpoints_xyz, lattice_constant, free_energy in results:
                f.write(f"{tot_kpoints}\t{kpoints_xyz}\t{lattice_constant}\t{free_energy}\n")
    except IOError as e:
        print(f"Error writing to file at {result_file_path}:", e)

def read_kpoints_free_energy(data_path):
    help_info = "Usage: read_kpoints_free_energy(data_path)\n" + \
                "data_path: Path to the data file containing lattice and free energy values.\n"

    # Check if the user asked for help
    if data_path == "help":
        print(help_info)
        return

    # Initialize the returns
    kpoints, direct_kpoints, lattice, free_energy = [], [], [], []

    with open(data_path, "r", encoding="utf-8") as data_file:
        lines = data_file.readlines()[1:]
        for line in lines:
            split_line = line.strip().split('\t')
            kpoints.append(int(split_line[0]))
            # Extract x, y, z from the tuple format (x, y, z)
            x, y, z = map(int, split_line[1][1:-1].split(','))
            direct_kpoints.append((x, y, z))
            lattice.append(float(split_line[2]))
            free_energy.append(float(split_line[3]))

    return kpoints, direct_kpoints, lattice, free_energy

def plot_kpoints_free_energy_data(matter, source_data=None, direction="Total", kpoints_boundary=None, color_family="blue"):
    help_info = "Usage: read_kpoints_free_energy(data_path)\n" + \
                "data_path: Path to the data file containing lattice and free energy values.\n"

    # Check if the user asked for help
    if (matter == "help") and (source_data is None):
        print(help_info)
        return

    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color calling
    colors = color_sampling(color_family)

    # Data input
    data_input = read_kpoints_free_energy(source_data)
    tot_kpoints = data_input[0]
    sep_kpoints = data_input[1]
    x_kpoints, y_kpoints, z_kpoints = [], [], []
    for coord in sep_kpoints:
        x_kpoints.append(coord[0])
        y_kpoints.append(coord[1])
        z_kpoints.append(coord[2])
    # lattice = data_input[2]
    energy = data_input[3]

    # Direction selecting
    if direction in ["X","x"]:
        kpoints_type = "X - Kpoints"
        kpoints = x_kpoints
    elif direction in ["Y","y"]:
        kpoints_type = "Y - Kpoints"
        kpoints = y_kpoints
    elif direction in ["Z","z"]:
        kpoints_type = "Z - Kpoints"
        kpoints = z_kpoints
    else:
        kpoints_type = "Total Kpoints"
        kpoints = tot_kpoints

    # Figure title
    plt.title(f"Total energy versus {kpoints_type} {matter}")
    plt.xlabel(f"{kpoints_type}"); plt.ylabel(r"Energy (eV)")

    # Axis style
    plt.ticklabel_format(style="sci", axis="y", scilimits=(-3,3))

    # Boundary
    kpoints_start = kpoints_boundary[0]
    kpoints_end = kpoints_boundary[1]

    if kpoints_end in ["", None]:
        kpoints_end = np.max(kpoints)
    if kpoints_start in ["", None]:
        kpoints_start = 1

    start_index = kpoints.index(kpoints_start)
    end_index = kpoints.index(kpoints_end)

    kpoints_plotting = kpoints[start_index:end_index+1]
    energy_plotting = energy[start_index:end_index+1]

    # Plotting
    plt.scatter(kpoints_plotting, energy_plotting, s=5, c=colors[1], zorder =1)
    plt.scatter(kpoints_plotting, energy_plotting, edgecolors=colors[1], s=45, facecolors="none", linewidths=1)
    plt.plot(kpoints_plotting, energy_plotting, c=colors[1], lw = 1.5)
    plt.xticks(kpoints_plotting)
    plt.tight_layout()
