#### Free energy extracting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt

from Store.output import canvas_setting, color_sampling

def lattice_free_energy_input():
    while True:
        free_energy_directory = input("Please input the directory of free energy calculation: ")
        if os.path.exists(free_energy_directory):
            print(f"Your free energy calculation is located in {free_energy_directory}.")
            return free_energy_directory
        else:
            print("The directory does not exist. Please try again.")

def specify_lattice_free_energy(directory):
    if directory == "help":
        print("Please use this function on the directory of the specific work folder.")
        return []
    xml_path = os.path.join(directory, "vasprun.xml")
    poscar_path = os.path.join(directory, "POSCAR")
    if os.path.isfile(xml_path) and os.path.isfile(poscar_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            e_fr_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)
            with open(poscar_path, "r", encoding="utf-8") as poscar_file:
                first_line = poscar_file.readline()
                a_var = float(first_line.split()[-1])
            return a_var, e_fr_energy
        except IOError as e:
            print("Error parsing files:", e)
            return None

def check_lattice_free_energy(directory="."):
    """Find folders with complete vasprun.xml and print incomplete ones."""
    # Check if the user asked for help
    if directory == "help":
        print("Please use this function on the parent directory of the project's main folder.")
        return []

    complete_folders = []
    # Traverse all folders under "directory"
    for dirpath, _, filenames in os.walk(directory):
        if "vasprun.xml" in filenames:
            file_name_xml = os.path.join(dirpath, "vasprun.xml")

            # Check if vasprun.xml is complete
            try:
                with open(file_name_xml, "r", encoding="utf-8") as f:
                    # Check the last few lines for the closing tag
                    last_lines = f.readlines()[-10:]    # read the last 10 lines
                    for line in last_lines:
                        if "</modeling>" in line or "</vasp>" in line:
                            complete_folders.append(dirpath)
                            break
                    else:
                        print(f"vasprun.xml in {dirpath} is incomplete.")
            except IOError as e:                        # Change from Exception to IOError
                print(f"Error reading {file_name_xml}: {e}")

    return complete_folders

def summarize_free_energy_directory(directory="."):
    result_file = "lattice_free_energy.dat"
    result_file_path = os.path.join(directory, result_file)     # Save the result file in the specified directory
    if directory == "help":
        print("Please use this function on the parent directory of the project's main folder.")
        return []

    # Use check_lattice_free_energy to get folders with complete vasprun.xml
    dirs_to_walk = check_lattice_free_energy(directory)
    results = []

    for dest_dir in dirs_to_walk:
        file_name_xml = os.path.join(dest_dir, "vasprun.xml")
        file_name_poscar = os.path.join(dest_dir, "POSCAR")

        if os.path.isfile(file_name_xml) and os.path.isfile(file_name_poscar):
            # Extract e_fr_energy from vasprun.xml
            tree = ET.parse(file_name_xml)
            root = tree.getroot()
            e_fr_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)

            # Extract a_var from POSCAR
            with open(file_name_poscar, "r", encoding="utf-8") as poscar_file:
                first_line = poscar_file.readline()
                a_var = float(first_line.split()[-1])  # assuming a_var is the last part on the first line
            # Add the result to the list
            results.append((a_var, e_fr_energy))

    # Sort the results by a_var (the first element of the tuple)
    results.sort(key=lambda x: x[0])

    # Now write the sorted results to the file
    with open(result_file_path, "w", encoding="utf-8") as f:
        f.write("Lattice\t Free energy\n")
        for a_var, e_fr_energy in results:
            f.write(f"{a_var:.3f}\t{e_fr_energy}\n")

def read_lattice_free_energy_data(lattice_path, density = 1):
    help_info = "Usage: read_lattice_free_energy_data(lattice_path, density)\n" + \
                "lattice_path: Path to the data file containing lattice and free energy values.\n" + \
                "density: Sampling density. For instance, use 1 for 100% sampling (default), 0.1 for 10%, and 0.5 for 50%."
    # Check if the user asked for help
    if lattice_path == "help":
        print(help_info)
        return
    # Initialize the lists for lattice constant and free energy
    lattice, free_energy = [], []
    # Calculate the interval based on density
    interval = int(1 / density) if density != 0 else 1
    with open(lattice_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for idx, line in enumerate(lines):
            # Only process lines based on the specified interval
            if idx % interval == 0:
                split_line = line.strip().split()
                lattice_value = float(split_line[0])
                lattice.append(lattice_value)
                free_energy.append(float(split_line[1]))
    return lattice, free_energy

def read_lattice_free_energy_count(lattice_path, count):
    help_info = "Usage: read_lattice_free_energy_count(lattice_path, count)\n" + \
                "lattice_path: Path to the data file containing lattice and free energy values.\n" + \
                "count: The number of samples."
    # Check if the user asked for help
    if lattice_path == "help":
        print(help_info)
        return
    # Initialize the lists for lattice constant and free energy
    lattice_value, free_energy_value = [], []
    lattice_sample_init, lattice_sample, free_energy_sample = [], [], []
    with open(lattice_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for line in lines:
            values = line.split("\t")
            lattice_value.append(float(values[0]))
            free_energy_value.append(float(values[1]))
        lattice_range = max(lattice_value) - min(lattice_value)
        interval = lattice_range / (count - 1)
        for sample in range(count):
            lattice_sample_init.append(lattice_value[0] + interval*sample)
        for current_lattice_sample in lattice_sample_init:
            # Find the index of the closest value in lattice_value to sample_lattice
            closest_index = min(range(len(lattice_value)), key=lambda i, current_sample=current_lattice_sample: abs(lattice_value[i] - current_sample))
            # Append the corresponding free energy value to free_energy_sampling
            free_energy_sample.append(free_energy_value[closest_index])
            # Append the actual lattice value to lattice_sample
            lattice_sample.append(lattice_value[closest_index])
        return lattice_sample, free_energy_sample

def polynomially_fit_curve(lattice_list, free_energy_list = None, fit_method = None, sample_count = None):
    help_info = "Usage: polynomially_fit_curve(lattice_list, free_energy_list, fit_method, sample_count)\n" + \
                "sample_count here means the sampling numbers.\n"
    # Check if the user asked for help
    if lattice_list == "help":
        print(help_info)
        return
    # Ensure the other parameters are provided
    if free_energy_list is None or fit_method is None or sample_count is None:
        raise ValueError("Missing required parameters. Use 'help' for more information.")
    # Apply polynomial fitting to the data
    p = np.polyfit(lattice_list, free_energy_list, fit_method)
    # Generate a polynomial function from the fitted parameters
    f = np.poly1d(p)
    # Generate new x and y data using the polynomial function
    fitted_lattice = np.linspace(min(lattice_list), max(lattice_list), num=sample_count, endpoint=True)
    fitted_free_energy = f(fitted_lattice)
    return fitted_lattice, fitted_free_energy

def plot_lattice_free_energy_single(matter, source_data, selected_data, sample_count, color_family):
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    plt.title(f"Free energy versus lattice for {matter}"); plt.xlabel(r"Lattice constant (Å)"); plt.ylabel(r"Energy (eV)")

    # Color calling
    color = color_sampling(color_family)
    # Data reading
    lattice_source, free_energy_source = read_lattice_free_energy_data(source_data)
    lattice_sample, free_energy_sample = read_lattice_free_energy_count(source_data, sample_count)
    fitted_lattice, fitted_free_energy = polynomially_fit_curve(lattice_source, free_energy_source, 3, 4000)
    selected_lattice, select_free_energy = specify_lattice_free_energy(selected_data)

    # Minimum free energy and the corresponding lattice
    min_energy_index = free_energy_source.index(min(free_energy_source))
    min_lattice = lattice_source[min_energy_index]
    min_free_energy = min(free_energy_source)

    plt.plot(fitted_lattice, fitted_free_energy, c=color[1], label="Fitted data", zorder=1)
    plt.scatter(lattice_sample, free_energy_sample, s=48, fc="#FFF", ec=color[1], label="Source data", zorder=1)
    plt.scatter(min_lattice, min_free_energy, s=96, fc="#FFF", ec=color[2], label="Source lowest point", zorder=1)
    plt.scatter(selected_lattice,  select_free_energy, s=24,  facecolors="#FFF", ec=color[4], label="Selected data", zorder=2)

    plt.legend(loc="best")
    plt.show()
