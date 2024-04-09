#### lattice information extracting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import xml.etree.ElementTree as ET
import os
import numpy as np

from vmatplot.algorithms import extract_midpoint
from vmatplot.algorithms import polynomially_fit_curve

def check_vasprun(directory="."):
    """Find folders with complete vasprun.xml and print incomplete ones."""
    # Check if the user asked for help
    if directory == "help":
        print("Please use this function on the parent directory of the project's main folder.")
        return []
    complete_folders = []
    # Traverse all folders under "directory"
    for dirpath, _, filenames in os.walk(directory):
        if "vasprun.xml" in filenames:
            xml_path = os.path.join(dirpath, "vasprun.xml")

            # Check if vasprun.xml is complete
            try:
                with open(xml_path, "r", encoding="utf-8") as xml_file:
                    # Check the last few lines for the closing tag
                    last_lines = xml_file.readlines()[-10:] # read the last 10 lines
                    for line in last_lines:
                        if "</modeling>" in line or "</vasp>" in line:
                            complete_folders.append(dirpath)
                            break
                    else:
                        print(f"vasprun.xml in {dirpath} is incomplete.")
            except IOError as e:                        # Change from Exception to IOError
                print(f"Error reading {xml_path}: {e}")
    return complete_folders

def lattice_constant(directory):
    if directory == "help":
        print("Please use this function on the directory of the specific work folder.")
        return []

    xml_path = os.path.join(directory, "vasprun.xml")
    if os.path.isfile(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract lattice constant
            # Assuming the lattice constant is the length of the "a" vector from the final structure
            basis_vectors = root.findall(".//calculation/structure/crystal/varray[@name='basis']")[-1]
            a_vector = basis_vectors[0].text.split()
            a_length = (float(a_vector[0])**2 + float(a_vector[1])**2 + float(a_vector[2])**2)**0.5

            return a_length

        except ET.ParseError as e:
            print("Error parsing vasprun.xml:", e)
            return None
    else:
        print("vasprun.xml not found in the specified directory.")
        return None

def lattice_free_energy(directory):
    if directory == "help":
        print("Please use this function on the directory of the specific work folder.")
        return []

    xml_path = os.path.join(directory, "vasprun.xml")
    if os.path.isfile(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract free energy
            free_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)

            return free_energy

        except ET.ParseError as e:
            print("Error parsing vasprun.xml:", e)
            return None
    else:
        print("vasprun.xml not found in the specified directory.")
        return None

def lattice_free_energy_input():
    while True:
        free_energy_directory = input("Please input the directory of free energy calculation: ")
        if os.path.exists(free_energy_directory):
            print(f"Your free energy calculation is located in {free_energy_directory}.")
            return free_energy_directory
        else:
            print("The directory does not exist. Please try again.")

def specify_free_energy_lattice(directory):
    if directory == "help":
        print("Please use this function on the directory of the specific work folder.")
        return []

    xml_path = os.path.join(directory, "vasprun.xml")
    if os.path.isfile(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract free energy
            free_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)

            # Extract lattice constant
            # Assuming the lattice constant is the length of the "a" vector from the final structure
            basis_vectors = root.findall(".//calculation/structure/crystal/varray[@name='basis']")[-1]
            a_vector = basis_vectors[0].text.split()
            a_length = (float(a_vector[0])**2 + float(a_vector[1])**2 + float(a_vector[2])**2)**0.5

            return a_length, free_energy

        except ET.ParseError as e:
            print("Error parsing vasprun.xml:", e)
            return None
    else:
        print("vasprun.xml not found in the specified directory.")
        return None

def summarize_free_energy_lattice_directory(directory=".", lattice_boundary=None):

    result_file = "free_energy_lattice.dat"
    result_file_path = os.path.join(directory, result_file)

    if directory == "help":
        print("Please use this function on the parent directory of the project's main folder.")
        return []

    # Use check_vasprun to get folders with complete vasprun.xml
    dirs_to_walk = check_vasprun(directory)
    results = []

    for work_dir in dirs_to_walk:
        xml_path = os.path.join(work_dir, "vasprun.xml")

        if os.path.isfile(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                free_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)

                basis_vectors = root.findall(".//calculation/structure/crystal/varray[@name='basis']")[-1]
                a_vector = basis_vectors[0].text.split()
                a_length = (float(a_vector[0])**2 + float(a_vector[1])**2 + float(a_vector[2])**2)**0.5

                within_start = True
                within_end = True
                if lattice_boundary is not None:
                    lattice_start, lattice_end = lattice_boundary
                    TOLERANCE = 1e-6
                    within_start = lattice_start is None or a_length >= lattice_start - TOLERANCE
                    within_end = lattice_end is None or a_length <= lattice_end + TOLERANCE

                if within_start and within_end:
                    results.append((a_length, free_energy))

            except (ET.ParseError, ValueError, IndexError) as e:
                print(f"Error processing {work_dir}/vasprun.xml:", e)

        else:
            print(f"vasprun.xml not found in {work_dir}.")

    results.sort(key=lambda x: x[0])

    if results:
        try:
            with open(result_file_path, "w", encoding="utf-8") as xml_file:
                xml_file.write("Lattice\t Free energy\n")
                for lattice_const, free_energy in results:
                    xml_file.write(f"{lattice_const}\t{free_energy}\n")
        except IOError as e:
            print("Error writing to file:", e)
    else:
        print("No suitable data found to write.")

def read_free_energy_lattice_data(lattice_path, density = 1):
    help_info = "Usage: read_free_energy_lattice_data(lattice_path, density)\n" + \
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
    with open(lattice_path, "r", encoding="utf-8") as dat_file:
        lines = dat_file.readlines()[1:]
        for idx, line in enumerate(lines):
            # Only process lines based on the specified interval
            if idx % interval == 0:
                split_line = line.strip().split()
                lattice_value = float(split_line[0])
                lattice.append(lattice_value)
                free_energy.append(float(split_line[-1]))
    return lattice, free_energy

def read_free_energy_lattice_count(lattice_path, count=None):
    help_info = "Usage: read_free_energy_lattice_count(lattice_path, count)\n" + \
                "lattice_path: Path to the data file containing lattice and free energy values.\n" + \
                "count: The number of samples, you can also enter 'all' for all the data."
    # Check if the user asked for help
    if lattice_path == "help":
        print(help_info)
        return
    # Initialize the lists for lattice constant and free energy
    lattice_value, free_energy_value = [], []
    lattice_sample_init, lattice_sample, free_energy_sample = [], [], []
    if count in [None, "ALL", "All", "all"]:
        return read_free_energy_lattice_data(lattice_path)
    elif count in [0, "NONE", "None", "none"]:
        return None
    elif count == 1:
        lattice_mid, energy_mid = extract_midpoint(read_free_energy_lattice_data(lattice_path)[0], read_free_energy_lattice_data(lattice_path)[1])
        return lattice_mid, energy_mid
    elif (count is not None) and (count not in ("ALL","All","all")):
        with open(lattice_path, "r", encoding="utf-8") as data_file:
            lines = data_file.readlines()[1:]
            for line in lines:
                values = line.split("\t")
                lattice_value.append(float(values[0]))
                free_energy_value.append(float(values[-1]))
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

def extract_fitted_minimum_free_energy_lattice(source_data):
    lattice_source, free_energy_source = read_free_energy_lattice_data(source_data)
    fitted_lattice, fitted_free_energy = polynomially_fit_curve(lattice_source, free_energy_source, 3, 4000)

    # Find the index of the minimum value in fitted_free_energy
    min_index = np.argmin(fitted_free_energy)

    # Use this index to get the minimum value of fitted_free_energy and its corresponding fitted_lattice value
    fitted_free_energy_min = fitted_free_energy[min_index]
    fitted_lattice_min = fitted_lattice[min_index]

    return fitted_lattice_min, fitted_free_energy_min

def extract_fitted_maximum_free_energy_lattice(source_data):
    lattice_source, free_energy_source = read_free_energy_lattice_data(source_data)
    fitted_lattice, fitted_free_energy = polynomially_fit_curve(lattice_source, free_energy_source, 3, 4000)

    # Find the index of the maximum value in fitted_free_energy
    max_index = np.argmax(fitted_free_energy)

    # Use this index to get the maximum value of fitted_free_energy and its corresponding fitted_lattice value
    fitted_free_energy_max = fitted_free_energy[max_index]
    fitted_lattice_max = fitted_lattice[max_index]

    return fitted_lattice_max, fitted_free_energy_max

def extract_fitted_extreme_free_energy_lattice(extract_type, source_data):
    if extract_type in ("Minium", "minium", "MIN", "Min", "min"):
        return extract_fitted_minimum_free_energy_lattice(source_data)
    if extract_type in ("Maxium", "maxium", "MAX", "Max", "max"):
        return extract_fitted_maximum_free_energy_lattice(source_data)
