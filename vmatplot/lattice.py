#### lattice information extracting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt

from vmatplot.algorithms import fit_eos
from vmatplot.output import canvas_setting, color_sampling
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

def summarize_free_energy_lattice_directory(directory=".", lattice_start = None, lattice_end = None):
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
                # Extract free_energy from vasprun.xml
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Extract free energy
                free_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)

                # Extract lattice constant
                # Assuming the lattice constant is the length of the 'a' vector from the final structure
                basis_vectors = root.findall(".//calculation/structure/crystal/varray[@name='basis']")[-1]
                a_vector = basis_vectors[0].text.split()
                a_length = (float(a_vector[0])**2 + float(a_vector[1])**2 + float(a_vector[2])**2)**0.5
                lattice_const = a_length

                # Check if lattice constant is within the specified range
                TOLERANCE = 1e-6
                within_start = lattice_start is None or lattice_const >= lattice_start - TOLERANCE
                within_end = lattice_end is None or lattice_const <= lattice_end + TOLERANCE

                if within_start and within_end:
                    results.append((lattice_const, free_energy))

            except (ET.ParseError, ValueError, IndexError) as e:
                print(f"Error processing {work_dir}/vasprun.xml:", e)

        else:
            print(f"vasprun.xml not found in {work_dir}.")

    # Sort the results by lattice_constant (the first element of the tuple)
    results.sort(key=lambda x: x[0])

    # Now write the sorted results to the file
    try:
        with open(result_file_path, "w", encoding="utf-8") as xml_file:
            xml_file.write("Lattice\t Free energy\n")
            for lattice_const, free_energy in results:
                xml_file.write(f"{lattice_const}\t{free_energy}\n")
    except IOError as e:
        print("Error writing to file:", e)

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
    if (count is not None) or (count not in ("ALL","All","all")):
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
    else:
        return read_free_energy_lattice_data(lattice_path)

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

def plot_free_energy_lattice_solo(matter, sample_count, source_data, color_family, selected_data=None):
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
    lattice_source, free_energy_source = read_free_energy_lattice_data(source_data)
    lattice_sample, free_energy_sample = read_free_energy_lattice_count(source_data, sample_count)

    # fitted_lattice, fitted_free_energy = polynomially_fit_curve(lattice_source, free_energy_source, 3, 4000)
    fitted_lattice, fitted_free_energy = fit_eos(lattice_source, free_energy_source)
    if selected_data is not None:
        selected_lattice, select_free_energy = specify_free_energy_lattice(selected_data)

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

def plot_free_energy_lattice_duo(title, sample_count, matter1, source_data1, color_family1, matter2, source_data2, color_family2, selected_data1=None, selected_data2=None):
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    plt.title(f"Free energy versus lattice for {title}"); plt.xlabel(r"Lattice constant (Å)"); plt.ylabel(r"Energy (eV)")

    # Color calling
    colors1 = color_sampling(color_family1)
    colors2 = color_sampling(color_family2)

    # Data input
    lattice_source1, free_energy_source1 = read_free_energy_lattice_data(source_data1)
    lattice_source2, free_energy_source2 = read_free_energy_lattice_data(source_data2)

    # fitted_lattice1, fitted_free_energy1 = polynomially_fit_curve(lattice_source1, free_energy_source1, 3, 4000)
    # fitted_lattice2, fitted_free_energy2 = polynomially_fit_curve(lattice_source2, free_energy_source2, 3, 4000)
    fitted_lattice1, fitted_free_energy1 = fit_eos(lattice_source1, free_energy_source1)
    fitted_lattice2, fitted_free_energy2 = fit_eos(lattice_source2, free_energy_source2)

    lattice_sample1, free_energy_sample1 = read_free_energy_lattice_count(source_data1, sample_count)
    lattice_sample2, free_energy_sample2 = read_free_energy_lattice_count(source_data2, sample_count)

    if selected_data1 is not None:
        selected_lattice1, select_free_energy1 = specify_free_energy_lattice(selected_data1)
    if selected_data2 is not None:
        selected_lattice2, select_free_energy2 = specify_free_energy_lattice(selected_data2)

    # Minimum free energy and the corresponding lattice
    min_energy_index1 = free_energy_source1.index(min(free_energy_source1))
    min_lattice1 = lattice_source1[min_energy_index1]
    min_free_energy1 = min(free_energy_source1)

    min_energy_index2 = free_energy_source2.index(min(free_energy_source2))
    min_lattice2 = lattice_source1[min_energy_index2]
    min_free_energy2 = min(free_energy_source2)

    # label processing
    if matter1 != "":
        label_matter1 = f"({matter1})"
    elif matter1 == "":
        label_matter1 = ""
    if matter2 != "":
        label_matter2 = f"({matter2})"
    elif matter2 == "":
        label_matter2 = ""

    # Plotting 1
    plt.plot(fitted_lattice1, fitted_free_energy1, c=colors1[1], label=f"Fitted data {label_matter1}", zorder=1)
    plt.scatter(lattice_sample1, free_energy_sample1, s=48, fc="#FFF", ec=colors1[1], label=f"Source data {label_matter1}", zorder=1)
    plt.scatter(min_lattice1, min_free_energy1, s=48, ec=colors1[2], fc=colors1[2], label=f"Source lowest point {label_matter1}", zorder=1)
    if selected_data1 is not None:
        plt.scatter(selected_lattice1,  select_free_energy1, s=24, ec=colors1[0], fc=colors1[0], label=f"Selected data {label_matter1}", zorder=2)

    # Plotting 2
    plt.plot(fitted_lattice2, fitted_free_energy2, c=colors2[1], label=f"Fitted data {label_matter2}", zorder=1)
    plt.scatter(lattice_sample2, free_energy_sample2, s=48, fc="#FFF", ec=colors2[1], label=f"Source data {label_matter2}", zorder=1)
    plt.scatter(min_lattice2, min_free_energy2, s=48, ec=colors2[2], fc=colors2[2], label=f"Source lowest point {label_matter2}", zorder=1)
    if selected_data2 is not None:
        plt.scatter(selected_lattice2,  select_free_energy2, s=24, ec=colors2[0], fc=colors2[0], label=f"Selected data {label_matter2}", zorder=2)

    plt.legend(loc=fig_setting[3])
    plt.show()

def plot_free_energy_lattice_tri(title, sample_count,
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
    lattice_source1, free_energy_source1 = read_free_energy_lattice_data(source_data1)
    lattice_source2, free_energy_source2 = read_free_energy_lattice_data(source_data2)
    lattice_source3, free_energy_source3 = read_free_energy_lattice_data(source_data3)

    # fitted_lattice1, fitted_free_energy1 = polynomially_fit_curve(lattice_source1, free_energy_source1, 3, 4000)
    # fitted_lattice2, fitted_free_energy2 = polynomially_fit_curve(lattice_source2, free_energy_source2, 3, 4000)
    # fitted_lattice3, fitted_free_energy3 = polynomially_fit_curve(lattice_source3, free_energy_source3, 3, 4000)
    fitted_lattice1, fitted_free_energy1 = fit_eos(lattice_source1, free_energy_source1)
    fitted_lattice2, fitted_free_energy2 = fit_eos(lattice_source2, free_energy_source2)
    fitted_lattice3, fitted_free_energy3 = fit_eos(lattice_source3, free_energy_source3)

    lattice_sample1, free_energy_sample1 = read_free_energy_lattice_count(source_data1, sample_count)
    lattice_sample2, free_energy_sample2 = read_free_energy_lattice_count(source_data2, sample_count)
    lattice_sample3, free_energy_sample3 = read_free_energy_lattice_count(source_data3, sample_count)

    if selected_data1 is not None:
        selected_lattice1, select_free_energy1 = specify_free_energy_lattice(selected_data1)
    if selected_data2 is not None:
        selected_lattice2, select_free_energy2 = specify_free_energy_lattice(selected_data2)
    if selected_data3 is not None:
        selected_lattice3, select_free_energy3 = specify_free_energy_lattice(selected_data3)

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

    # label processing
    if matter1 != "":
        label_matter1 = f"({matter1})"
    elif matter1 == "":
        label_matter1 = ""
    if matter2 != "":
        label_matter2 = f"({matter2})"
    elif matter2 == "":
        label_matter2 = ""
    if matter3 != "":
        label_matter3 = f"({matter3})"
    elif matter3 == "":
        label_matter3 = ""

    # Plotting 1
    plt.plot(fitted_lattice1, fitted_free_energy1, c=colors1[1], label=f"Fitted data {label_matter1}", zorder=1)
    plt.scatter(lattice_sample1, free_energy_sample1, s=48, fc="#FFF", ec=colors1[1], label=f"Source data {label_matter1}", zorder=1)
    plt.scatter(min_lattice1, min_free_energy1, s=48, ec=colors1[2], fc=colors1[2], label=f"Source lowest point {label_matter1}", zorder=1)
    if selected_data1 is not None:
        plt.scatter(selected_lattice1,  select_free_energy1, s=24, ec=colors1[0], fc=colors1[0], label=f"Selected data {label_matter1}", zorder=2)

    # Plotting 2
    plt.plot(fitted_lattice2, fitted_free_energy2, c=colors2[1], label=f"Fitted data {label_matter2}", zorder=1)
    plt.scatter(lattice_sample2, free_energy_sample2, s=48, fc="#FFF", ec=colors2[1], label=f"Source data {label_matter2}", zorder=1)
    plt.scatter(min_lattice2, min_free_energy2, s=48, ec=colors2[2], fc=colors2[2], label=f"Source lowest point {label_matter2}", zorder=1)
    if selected_data2 is not None:
        plt.scatter(selected_lattice2,  select_free_energy2, s=24, ec=colors2[0], fc=colors2[0], label=f"Selected data {label_matter2}", zorder=2)

    # Plotting 3
    plt.plot(fitted_lattice3, fitted_free_energy3, c=colors3[1], label=f"Fitted data {label_matter3}", zorder=1)
    plt.scatter(lattice_sample3, free_energy_sample3, s=48, fc="#FFF", ec=colors3[1], label=f"Source data {label_matter3}", zorder=1)
    plt.scatter(min_lattice3, min_free_energy3, s=48, ec=colors3[2], fc=colors3[2], label=f"Source lowest point {label_matter3}", zorder=1)
    if selected_data3 is not None:
        plt.scatter(selected_lattice3,  select_free_energy3, s=24, ec=colors3[0], fc=colors3[0], label=f"Selected data {label_matter3}", zorder=2)

    plt.legend(loc=fig_setting[3])
    plt.show()

def plot_free_energy_lattice_quad(title, sample_count,
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
    lattice_source1, free_energy_source1 = read_free_energy_lattice_data(source_data1)
    lattice_source2, free_energy_source2 = read_free_energy_lattice_data(source_data2)
    lattice_source3, free_energy_source3 = read_free_energy_lattice_data(source_data3)
    lattice_source4, free_energy_source4 = read_free_energy_lattice_data(source_data4)

    # fitted_lattice1, fitted_free_energy1 = polynomially_fit_curve(lattice_source1, free_energy_source1, 3, 4000)
    # fitted_lattice2, fitted_free_energy2 = polynomially_fit_curve(lattice_source2, free_energy_source2, 3, 4000)
    # fitted_lattice3, fitted_free_energy3 = polynomially_fit_curve(lattice_source3, free_energy_source3, 3, 4000)
    # fitted_lattice4, fitted_free_energy4 = polynomially_fit_curve(lattice_source4, free_energy_source4, 3, 4000)
    fitted_lattice1, fitted_free_energy1 = fit_eos(lattice_source1, free_energy_source1)
    fitted_lattice2, fitted_free_energy2 = fit_eos(lattice_source2, free_energy_source2)
    fitted_lattice3, fitted_free_energy3 = fit_eos(lattice_source3, free_energy_source3)
    fitted_lattice4, fitted_free_energy4 = fit_eos(lattice_source4, free_energy_source4)

    lattice_sample1, free_energy_sample1 = read_free_energy_lattice_count(source_data1, sample_count)
    lattice_sample2, free_energy_sample2 = read_free_energy_lattice_count(source_data2, sample_count)
    lattice_sample3, free_energy_sample3 = read_free_energy_lattice_count(source_data3, sample_count)
    lattice_sample4, free_energy_sample4 = read_free_energy_lattice_count(source_data4, sample_count)

    if selected_data1 is not None:
        selected_lattice1, select_free_energy1 = specify_free_energy_lattice(selected_data1)
    if selected_data2 is not None:
        selected_lattice2, select_free_energy2 = specify_free_energy_lattice(selected_data2)
    if selected_data3 is not None:
        selected_lattice3, select_free_energy3 = specify_free_energy_lattice(selected_data3)
    if selected_data4 is not None:
        selected_lattice4, select_free_energy4 = specify_free_energy_lattice(selected_data4)

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

    # label processing
    if matter1 != "":
        label_matter1 = f"({matter1})"
    elif matter1 == "":
        label_matter1 = ""
    if matter2 != "":
        label_matter2 = f"({matter2})"
    elif matter2 == "":
        label_matter2 = ""
    if matter3 != "":
        label_matter3 = f"({matter3})"
    elif matter3 == "":
        label_matter3 = ""
    if matter4 != "":
        label_matter4 = f"({matter4})"
    elif matter4 == "":
        label_matter4 = ""

    # Plotting 1
    # plt.plot(fitted_lattice1, fitted_free_energy1, c=colors1[1], label=f"Fitted data {label_matter1}", zorder=1)
    plt.plot(fitted_lattice1, fitted_free_energy1, c=colors1[1], zorder=1)
    plt.scatter(lattice_sample1, free_energy_sample1, s=48, fc="#FFF", ec=colors1[1], label=f"Source data {label_matter1}", zorder=1)
    plt.scatter(min_lattice1, min_free_energy1, s=48, ec=colors1[2], fc=colors1[2], label=f"Source lowest point {label_matter1}", zorder=1)
    if selected_data1 is not None:
        plt.scatter(selected_lattice1,  select_free_energy1, s=24, ec=colors1[0], fc=colors1[0], label=f"Selected data {label_matter1}", zorder=2)

    # Plotting 2
    # plt.plot(fitted_lattice2, fitted_free_energy2, c=colors2[1], label=f"Fitted data {label_matter2}", zorder=1)
    plt.plot(fitted_lattice2, fitted_free_energy2, c=colors2[1], zorder=1)
    plt.scatter(lattice_sample2, free_energy_sample2, s=48, fc="#FFF", ec=colors2[1], label=f"Source data {label_matter2}", zorder=1)
    plt.scatter(min_lattice2, min_free_energy2, s=48, ec=colors2[2], fc=colors2[2], label=f"Source lowest point {label_matter2}", zorder=1)
    if selected_data2 is not None:
        plt.scatter(selected_lattice2,  select_free_energy2, s=24, ec=colors2[0], fc=colors2[0], label=f"Selected data {label_matter2}", zorder=2)

    # Plotting 3
    # plt.plot(fitted_lattice3, fitted_free_energy3, c=colors3[1], label=f"Fitted data {label_matter3}", zorder=1)
    plt.plot(fitted_lattice3, fitted_free_energy3, c=colors3[1], zorder=1)
    plt.scatter(lattice_sample3, free_energy_sample3, s=48, fc="#FFF", ec=colors3[1], label=f"Source data {label_matter3}", zorder=1)
    plt.scatter(min_lattice3, min_free_energy3, s=48, ec=colors3[2], fc=colors3[2], label=f"Source lowest point {label_matter3}", zorder=1)
    if selected_data3 is not None:
        plt.scatter(selected_lattice3,  select_free_energy3, s=24, ec=colors3[0], fc=colors3[0], label=f"Selected data {label_matter3}", zorder=2)

    # Plotting 4
    # plt.plot(fitted_lattice4, fitted_free_energy4, c=colors4[1], label=f"Fitted data {label_matter4}", zorder=1)
    plt.plot(fitted_lattice4, fitted_free_energy4, c=colors4[1], zorder=1)
    plt.scatter(lattice_sample4, free_energy_sample4, s=48, fc="#FFF", ec=colors4[1], label=f"Source data {label_matter4}", zorder=1)
    plt.scatter(min_lattice4, min_free_energy4, s=48, ec=colors4[2], fc=colors4[2], label=f"Source lowest point {label_matter4}", zorder=1)
    if selected_data4 is not None:
        plt.scatter(selected_lattice4,  select_free_energy4, s=24, ec=colors4[0], fc=colors4[0], label=f"Selected data {label_matter4}", zorder=2)

    plt.legend(loc=fig_setting[3])
    plt.show()

lattice_free_energy_ploting_help_info = "help information"

def plot_free_energy_lattice(matter_count, *args):
    if matter_count == "help":
        print(lattice_free_energy_ploting_help_info)

    if matter_count == 1:
        if len(args) == 4:
            return plot_free_energy_lattice_solo(args[0], args[1], args[2], args[3])
        elif len(args) == 5:
            return plot_free_energy_lattice_solo(args[0], args[1], args[2], args[3], args[4])

    if matter_count == 2:
        if len(args) == 8:
            return plot_free_energy_lattice_duo(args[0], args[1], args[2], args[3],
                                                args[4], args[5], args[6], args[7])
        elif len(args) == 10:
            return plot_free_energy_lattice_duo(args[0], args[1], args[2], args[3],
                                                args[4], args[5], args[6], args[7],
                                                args[8], args[9])
    if matter_count == 3:
        if len(args) == 11:
            return plot_free_energy_lattice_tri(args[0], args[1], args[2], args[3],
                                                args[4], args[5], args[6], args[7],
                                                args[8], args[9], args[10])
        elif len(args) == 14:
            return plot_free_energy_lattice_tri(args[0], args[1], args[2], args[3],
                                                args[4], args[5], args[6], args[7],
                                                args[8], args[9], args[10],args[11],
                                                args[12], args[13])
    if matter_count == 4:
        if len(args) == 14:
            return plot_free_energy_lattice_quad(args[0], args[1], args[2], args[3],
                                                 args[4], args[5], args[6], args[7],
                                                 args[8], args[9], args[10],args[11],
                                                 args[12], args[13])
        elif len(args) == 18:
            return plot_free_energy_lattice_quad(args[0], args[1], args[2], args[3],
                                                 args[4], args[5], args[6], args[7],
                                                 args[8], args[9], args[10],args[11],
                                                 args[12], args[13],args[14], args[15],
                                                 args[16], args[17])
