#### Declarations of process functions for Dielectric function
# pylint: disable = C0103, C0114, C0116, C0301, R0914

### Necessary packages invoking

import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py

from vmatplot.commons import extract_fermi, process_boundary, process_boundaries_rescaling, extract_part
from vmatplot.output import canvas_setting, color_sampling
from vmatplot.algorithms import energy_to_wavelength, wavelength_to_energy

from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

### Physical constants
hbar = 4.135667662e-15
c_vacuum = 2.99792458e8

### Data extracting
def extract_dielectric_vasprun(directory):
    # Construct the full path to the vasprun.xml file
    vasprun_path = os.path.join(directory, "vasprun.xml")
    # Check if the vasprun.xml file exists in the given directory
    if not os.path.isfile(vasprun_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory}.")
        return

    ## Analysis vasprun.xml file
    tree = ET.parse(vasprun_path)
    root = tree.getroot()

    ## Algorithm type
    # algorithm_type = identify_algorithm(directory)

    data_label = "dielectricfunction"

    ## Extract NEDOS
    nedos_element = root.find(".//i[@name='NEDOS']")
    nedos = int(nedos_element.text.strip())

    ## Extract NBANDS
    nbands_element = root.find(".//i[@name='NBANDS']")
    nbands = int(nbands_element.text.strip())

    ## Loop variables
    # Define prefixes:
        # e_ for Density-Density (Charge density response)
        # c_ for Current-Current (Current response)
    prefixes = ["e_", "c_"]
    # Initialize a dictionary to store dynamic variables
    data = {}

    ## Extract imaginary part of Density-Density and Current-Current
    imag_path = f".//{data_label}/imag/array/set"
    imag_elements = root.findall(imag_path)
    for loop_index, imag_element in enumerate(imag_elements[0:2]):
        # Select prefix based on the loop index
        prefix = prefixes[loop_index]
        # Initialize columns as lists
        columns = ["energy_imag_col",
                   "xx_imag_col", "yy_imag_col", "zz_imag_col", "xy_imag_col", "yz_imag_col", "zx_imag_col"]
        for col in columns:
            data[prefix + col] = []
        # Append data to lists
        for imag_index in imag_element.findall("r"):
            imag_values = list(map(float, imag_index.text.split()))
            for value_index, col in enumerate(columns):
                data[prefix + col].append(imag_values[value_index])
        # Convert lists to numpy arrays
        for col in columns:
            data[prefix + col] = np.array(data[prefix + col])

    ## Extract real part of Density-Density and Current-Current
    real_path = f".//{data_label}/real/array/set"
    real_elements = root.findall(real_path)

    for loop_index, real_element in enumerate(real_elements[0:2]):
        # Select prefix based on the loop index
        prefix = prefixes[loop_index]
        # Initialize columns as lists
        columns = ["energy_real_col",
                   "xx_real_col", "yy_real_col", "zz_real_col", "xy_real_col", "yz_real_col", "zx_real_col"]
        for col in columns:
            data[prefix + col] = []
        # Append data to lists
        for real_index in real_element.findall("r"):
            real_values = list(map(float, real_index.text.split()))
            for value_index, col in enumerate(columns):
                data[prefix + col].append(real_values[value_index])
        # Convert lists to numpy arrays
        for col in columns:
            data[prefix + col] = np.array(data[prefix + col])

    ## Extract imaginary part of Current-Current
    current_imag_path = f".//{data_label}[@comment='current-current']/imag/array/set"
    current_imag_elements = root.findall(current_imag_path)
    current_imag_columns = ["c_energy_imag_col",
                            "c_xx_imag_col", "c_yy_imag_col", "c_zz_imag_col",
                            "c_xy_imag_col", "c_yz_imag_col", "c_zx_imag_col"]
    for _, element in enumerate(current_imag_elements[0:2]):
        for column in current_imag_columns:
            data[column] = []
        # Append data to lists
        for index in element.findall("r"):
            values = list(map(float, index.text.split()))
            for value_index, column in enumerate(current_imag_columns):
                data[column].append(values[value_index])
        # Convert lists to numpy arrays
        for column in current_imag_columns:
            data[column]=np.array(data[column])

    ## Extract real part of Current-Current
    current_real_path = f".//{data_label}[@comment='current-current']/real/array/set"
    current_real_elements = root.findall(current_real_path)
    current_real_columns = ["c_energy_real_col",
                            "c_xx_real_col", "c_yy_real_col", "c_zz_real_col",
                            "c_xy_real_col", "c_yz_real_col", "c_zx_real_col"]
    for _, element in enumerate(current_real_elements[0:2]):
        for column in current_real_columns :
            data[column] = []
        # Append data to lists
        for index in element.findall("r"):
            values = list(map(float, index.text.split()))
            for value_index, column in enumerate(current_real_columns):
                data[column].append(values[value_index])
        # Convert lists to numpy arrays
        for column in current_real_columns:
            data[column]=np.array(data[column])

    ## Extract Fermi energy
    # efermi_element = root.find(".//dos/i[@name='efermi']")
    # fermi_energy = float(efermi_element.text.strip())
    fermi_energy = extract_fermi(directory)

    ## Extract Conductivity
    conductivity_path = ".//conductivity[@comment='spin=1']/array/set"
    conductivity_set_element = tree.find(conductivity_path)
    # Initialize columns as lists
    columns = ["conductivity_energy",
               "conductivity_xx", "conductivity_yy", "conductivity_zz", "conductivity_xy", "conductivity_yz", "conductivity_zx"]
    # Initialize dictionary
    conductivity_data = {col: [] for col in columns}
    for conductivity_index in conductivity_set_element.findall("r"):
        values = list(map(float, conductivity_index.text.split()))
        conductivity_data["conductivity_energy"].append(values[0])
        conductivity_data["conductivity_xx"].append(values[1])
        conductivity_data["conductivity_yy"].append(values[2])
        conductivity_data["conductivity_zz"].append(values[3])
        conductivity_data["conductivity_xy"].append(values[4])
        conductivity_data["conductivity_yz"].append(values[5])
        conductivity_data["conductivity_zx"].append(values[6])

    ## Extract DOS
    dos_path = ".//dos/total/array/set/set[@comment='spin 1']"
    dos_set_element = tree.find(dos_path)
    # Initialize columns as lists
    columns = ["dos_energy", "total_dos", "integrated_dos"]
    # Initialize dictionary
    dos_data = {col: [] for col in columns}
    for dos_index in dos_set_element.findall("r"):
        values = list(map(float, dos_index.text.split()))
        dos_data["dos_energy"].append(values[0])
        dos_data["total_dos"].append(values[1])
        dos_data["integrated_dos"].append(values[2])

    return {
        "nedos and nbands": (nedos, nbands),                                # [0]: Current NEDOS and NBANDS
        "density_energy_imag": data["e_energy_imag_col"],                   # [1]: Imaginary part of energy of Density-Density
        "density_xx_imag": data["e_xx_imag_col"],                           # [2]: Imaginary part of xx direction of Density-Density
        "density_yy_imag": data["e_yy_imag_col"],                           # [3]: Imaginary part of yy direction of Density-Density
        "density_zz_imag": data["e_zz_imag_col"],                           # [4]: Imaginary part of zz direction of Density-Density
        "density_xy_imag": data["e_xy_imag_col"],                           # [5]: Imaginary part of xy direction of Density-Density
        "density_yz_imag": data["e_yz_imag_col"],                           # [6]: Imaginary part of yz direction of Density-Density
        "density_zx_imag": data["e_zx_imag_col"],                           # [7]: Imaginary part of zx direction of Density-Density
        "current_energy_imag": data["c_energy_imag_col"],                   # [8]: Imaginary part of energy of Current-Current
        "current_xx_imag": data["c_xx_imag_col"],                           # [9]: Imaginary part of xx direction of Current-Current
        "current_yy_imag": data["c_yy_imag_col"],                           # [10]: Imaginary part of yy direction of Current-Current
        "current_zz_imag": data["c_zz_imag_col"],                           # [11]: Imaginary part of zz direction of Current-Current
        "current_xy_imag": data["c_xy_imag_col"],                           # [12]: Imaginary part of xy direction of Current-Current
        "current_yz_imag": data["c_yz_imag_col"],                           # [13]: Imaginary part of yz direction of Current-Current
        "current_zx_imag": data["c_zx_imag_col"],                           # [14]: Imaginary part of zx direction of Current-Current
        "density_energy_real": data["e_energy_real_col"],                   # [15]: Real part of energy of Density-Density
        "density_xx_real": data["e_xx_real_col"],                           # [16]: Real part of xx direction of Density-Density
        "density_yy_real": data["e_yy_real_col"],                           # [17]: Real part of yy direction of Density-Density
        "density_zz_real": data["e_zz_real_col"],                           # [18]: Real part of zz direction of Density-Density
        "density_xy_real": data["e_xy_real_col"],                           # [19]: Real part of xy direction of Density-Density
        "density_yz_real": data["e_yz_real_col"],                           # [20]: Real part of yz direction of Density-Density
        "density_zx_real": data["e_zx_real_col"],                           # [21]: Real part of zx direction of Density-Density
        "current_energy_real": data["c_energy_real_col"],                   # [22]: Real part of energy of Current-Current
        "current_xx_real": data["c_xx_real_col"],                           # [23]: Real part of xx direction of Current-Current
        "current_yy_real": data["c_yy_real_col"],                           # [24]: Real part of yy direction of Current-Current
        "current_zz_real": data["c_zz_real_col"],                           # [25]: Real part of zz direction of Current-Current
        "current_xy_real": data["c_xy_real_col"],                           # [26]: Real part of xy direction of Current-Current
        "current_yz_real": data["c_yz_real_col"],                           # [27]: Real part of yz direction of Current-Current
        "current_zx_real": data["c_zx_real_col"],                           # [28]: Real part of zx direction of Current-Current
        "data_label": data_label,                                           # [29]: Data label
        "fermi_energy": fermi_energy,                                       # [30]: System Fermi energy
        "conductivity_energy": conductivity_data["conductivity_energy"],    # [31]: Energy of conductivity
        "conductivity_xx": conductivity_data["conductivity_xx"],            # [32]: xx direction of conductivity
        "conductivity_yy": conductivity_data["conductivity_yy"],            # [33]: yy direction of conductivity
        "conductivity_zz": conductivity_data["conductivity_zz"],            # [34]: zz direction of conductivity
        "conductivity_xy": conductivity_data["conductivity_xy"],            # [35]: xy direction of conductivity
        "conductivity_yz": conductivity_data["conductivity_yz"],            # [36]: yz direction of conductivity
        "conductivity_zx": conductivity_data["conductivity_zx"],            # [37]: zx direction of conductivity
        "dos_energy": dos_data["dos_energy"],                               # [38]: Energy list of DOS
        "total_dos": dos_data["total_dos"],                                 # [39]: Total DOS
        "integrated_dos": dos_data["integrated_dos"],                       # [40]: Integrated DOS
    }

def extract_dielectric_hdf5(directory):
    # Construct the full path to the vaspout.h5 file
    h5_path = os.path.join(directory, "vaspout.h5")
    # Check if the vaspout.h5 file exists in the given directory
    if not os.path.isfile(h5_path):
        print(f"Error: The file vaspout.h5 does not exist in the directory {directory}.")
        return

    # Open the vaspout.h5 file and extract the energies_dielectric_function
    with h5py.File(h5_path, "r") as f:
        # Extract the energies_dielectric_function dataset
        energy_list = f["results/linear_response/energies_dielectric_function"][()]
        current_list = f["results/linear_response/current_current_dielectric_function"][()]
        density_list = f["results/linear_response/density_density_dielectric_function"][()]

    # Return the extracted data as a dictionary
    return {
        "energy": energy_list,
        "current_energy_real": energy_list,
        "current_xx_real": current_list[0,0,:,0],
        "current_yy_real": current_list[1,1,:,0],
        "current_zz_real": current_list[2,2,:,0],
        "current_xy_real": current_list[0,1,:,0],
        "current_yx_real": current_list[1,0,:,0],
        "current_yz_real": current_list[1,2,:,0],
        "current_zy_real": current_list[2,1,:,0],
        "current_zx_real": current_list[2,0,:,0],
        "current_xz_real": current_list[0,2,:,0],
        "current_energy_imag": energy_list,
        "current_xx_imag": current_list[0,0,:,1],
        "current_yy_imag": current_list[1,1,:,1],
        "current_zz_imag": current_list[2,2,:,1],
        "current_xy_imag": current_list[0,1,:,1],
        "current_yx_imag": current_list[1,0,:,1],
        "current_yz_imag": current_list[1,2,:,1],
        "current_zy_imag": current_list[2,1,:,1],
        "current_zx_imag": current_list[2,0,:,1],
        "current_xz_imag": current_list[0,2,:,1],
        "density_energy_real": energy_list,
        "density_xx_real": density_list[0,0,:,0],
        "density_yy_real": density_list[1,1,:,0],
        "density_zz_real": density_list[2,2,:,0],
        "density_xy_real": density_list[0,1,:,0],
        "density_yx_real": density_list[1,0,:,0],
        "density_yz_real": density_list[1,2,:,0],
        "density_zy_real": density_list[2,1,:,0],
        "density_zx_real": density_list[2,0,:,0],
        "density_xz_real": density_list[0,2,:,0],
        "density_energy_imag": energy_list,
        "density_xx_imag": density_list[0,0,:,1],
        "density_yy_imag": density_list[1,1,:,1],
        "density_zz_imag": density_list[2,2,:,1],
        "density_xy_imag": density_list[0,1,:,1],
        "density_yx_imag": density_list[1,0,:,1],
        "density_yz_imag": density_list[1,2,:,1],
        "density_zy_imag": density_list[2,1,:,1],
        "density_zx_imag": density_list[2,0,:,1],
        "density_xz_imag": density_list[0,2,:,1],
    }

def extract_dielectric_hdf5opt(directory):
    # Use "myhdf5" to extract dielectric function
    # Address: results / linear_response_kpoints_opt
    # [0] Phono energy: energies_dielectric_function
    # [1] Real part xx plane: D2 0,0,:,0
    # [2] Imag part xx plane: D2 0,0,:,1
    # [3] Real part yy plane: D2 1,1,:,0
    # [4] Imag part yy plane: D2 1,1,:,1
    # [5] Real part zz plane: D2 2,2,:,0
    # [6] Imag part zz plane: D2 2,2,:,1
    # Construct the full path to the vaspout.h5 file
    h5_path = os.path.join(directory, "vaspout.h5")
    # Check if the vaspout.h5 file exists in the given directory
    if not os.path.isfile(h5_path):
        print(f"Error: The file vaspout.h5 does not exist in the directory {directory}.")
        return

    # Open the vaspout.h5 file and extract the energies_dielectric_function
    with h5py.File(h5_path, "r") as f:
        # Extract the energies_dielectric_function dataset
        energy_list = f["results/linear_response_kpoints_opt/energies_dielectric_function"][()]
        current_list = f['results/linear_response_kpoints_opt/current_current_dielectric_function'][()]
        density_list = f['results/linear_response_kpoints_opt/density_density_dielectric_function'][()]

    # Return the extracted data as a dictionary
    return {
        "energy": energy_list,
        "current_energy_real": energy_list,
        "current_xx_real": current_list[0,0,:,0],
        "current_yy_real": current_list[1,1,:,0],
        "current_zz_real": current_list[2,2,:,0],
        "current_xy_real": current_list[0,1,:,0],
        "current_yx_real": current_list[1,0,:,0],
        "current_yz_real": current_list[1,2,:,0],
        "current_zy_real": current_list[2,1,:,0],
        "current_zx_real": current_list[2,0,:,0],
        "current_xz_real": current_list[0,2,:,0],
        "current_energy_imag": energy_list,
        "current_xx_imag": current_list[0,0,:,1],
        "current_yy_imag": current_list[1,1,:,1],
        "current_zz_imag": current_list[2,2,:,1],
        "current_xy_imag": current_list[0,1,:,1],
        "current_yx_imag": current_list[1,0,:,1],
        "current_yz_imag": current_list[1,2,:,1],
        "current_zy_imag": current_list[2,1,:,1],
        "current_zx_imag": current_list[2,0,:,1],
        "current_xz_imag": current_list[0,2,:,1],
        "density_energy_real": energy_list,
        "density_xx_real": density_list[0,0,:,0],
        "density_yy_real": density_list[1,1,:,0],
        "density_zz_real": density_list[2,2,:,0],
        "density_xy_real": density_list[0,1,:,0],
        "density_yx_real": density_list[1,0,:,0],
        "density_yz_real": density_list[1,2,:,0],
        "density_zy_real": density_list[2,1,:,0],
        "density_zx_real": density_list[2,0,:,0],
        "density_xz_real": density_list[0,2,:,0],
        "density_energy_imag": energy_list,
        "density_xx_imag": density_list[0,0,:,1],
        "density_yy_imag": density_list[1,1,:,1],
        "density_zz_imag": density_list[2,2,:,1],
        "density_xy_imag": density_list[0,1,:,1],
        "density_yx_imag": density_list[1,0,:,1],
        "density_yz_imag": density_list[1,2,:,1],
        "density_zy_imag": density_list[2,1,:,1],
        "density_zx_imag": density_list[2,0,:,1],
        "density_xz_imag": density_list[0,2,:,1],
    }

def extract_dielectric_function(directory):
    hdf5_path = os.path.join(directory, "vaspout.h5")
    opt_path = os.path.join(directory, "KPOINTS_OPT")
    if os.path.isfile(hdf5_path):
        if os.path.isfile(opt_path):
            return extract_dielectric_hdf5opt(directory)
        else:
            return extract_dielectric_hdf5(directory)
    else:
        return extract_dielectric_vasprun(directory)

### Visualization
def dielectric_systems_list(systems):
    # data = dielectric_systems_list(systems)
    # data[0] = current curve label
    # data[1] = dielectric function data
    # data[2] = color family
    # data[3] = linestyle
    # data[4] = alpha
    # data[5] = linewidth
    data = []
    for values_dir in systems:
        if len(values_dir) == 2:
            label, directory = values_dir
            color = "blue"
            linestyle = "solid"
            alpha = 1.0
            linewidth = None
        elif len(values_dir) == 3:
            label, directory, color = values_dir
            linestyle = "solid"
            alpha = 1.0
            linewidth = None
        elif len(values_dir) == 4:
            label, directory, color, linestyle = values_dir
            alpha = 1.0
            linewidth = None
        elif len(values_dir) == 5:
            label, directory, color, linestyle, alpha = values_dir
            linewidth = None
        else:
            label, directory, color, linestyle, alpha, linewidth = values_dir
        dielectric_data = extract_dielectric_function(directory)
        data.append([label,dielectric_data,color,linestyle,alpha,linewidth])
    return data

def identify_components(component_key):
    components = {
        "xx": ("density_xx_real", "density_xx_imag"),
        "yy": ("density_yy_real", "density_yy_imag"),
        "zz": ("density_zz_real", "density_zz_imag"),
        "xy": ("density_xy_real", "density_xy_imag"),
        "yx": ("density_yx_real", "density_yx_imag"),
        "yz": ("density_yz_real", "density_yz_imag"),
        "zy": ("density_zy_real", "density_zy_imag"),
        "zx": ("density_zx_real", "density_zx_imag"),
        "xz": ("density_xz_real", "density_xz_imag"),
    }
    return components.get(component_key)

def plot_dielectric_help():
    help_info = "Usage: plot_dielectric_function \n" + \
                "\t Demonstrate dielectric function by each component \n" +\
                "The independent value includes \n" +\
                "\t suptitle: the suptitle; \n" +\
                "\t systems: dielectric function data list; \n" +\
                "\t components: select components in a list ({'xx', 'yy', 'zz', 'xy', 'yx', 'yz', 'zy', 'zx', 'xz'}); \n" +\
                "\t layout: subfigures layout (horizontal<default>, vertical); \n" +\
                "\t expansion_label: whether to expand the real and imaginary parts (True<default>, False); \n" +\
                "\t unit: x-axis unit (eV<default>, nm); \n" +\
                "\t boundary: a-axis range <optional>; \n" +\
                "\t figure_size: figure size <optional>. \n"
    return help_info

def plot_dielectric_monocomp(suptitle, systems=None, component=None,layout="horizontal", expansion_label=True,
                             unit=None, boundary=(None,None), figure_size=(None,None)):
    ## Help information
    dielectric_help =  plot_dielectric_help()
    if suptitle in ["help", "Help"]:
        print(dielectric_help)
        return

    ## scale flag and databoundaries
    rescale_flag, source_start, source_end, scaled_start, scaled_end = process_boundaries_rescaling(boundary)

    ## components aliases
    comp_label, comp_aliase = None, None
    if isinstance(component, list):
        component = component[0]
    if isinstance(component, dict):
        key, value = next(iter(component.items()))
        comp_label = key.lower()
        comp_aliase = value
    else:
        comp_label = component.lower()
        comp_aliase = f"{component}-component"

    ## expansion flag
    if isinstance(expansion_label, bool):
        expansion_flag = expansion_label
    elif expansion_label.lower() not in ["true", "yes", "t", "y", "combine"]:
        expansion_flag = False
    else:
        expansion_flag = True

    ## figure settings
    single_subfigure = False
    all_expansion = False
    layout_flag = "horizontal" if layout.lower() not in ["vertical", "ver","v"] else "vertical"
    if rescale_flag is False and expansion_flag is False:
        single_subfigure = True
        fig_setting = canvas_setting() if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
        plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
        params = fig_setting[2]
        plt.rcParams.update(params)
    elif rescale_flag is True and expansion_flag is True:
        all_expansion = True
        if layout_flag == "horizontal":
            fig_setting = canvas_setting(16, 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(2, 2, figsize=fig_setting[0], dpi=fig_setting[1])
            axes_element = [axs[i, j] for i in range(2) for j in range(2)] # [[0, 0], [0, 1], [1, 0], [1, 1]]
        else:
            fig_setting = canvas_setting(16, 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(2, 2, figsize=fig_setting[0], dpi=fig_setting[1])

            axes_element = [axs[i, j] for j in range(2) for i in range(2)] # [[0, 0], [1, 0], [0, 1], [1, 1]]
    elif rescale_flag is not expansion_flag:
        if layout_flag == "horizontal":
            fig_setting = canvas_setting(16, 6) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(1, 2, figsize=fig_setting[0], dpi=fig_setting[1])
            axes_element = [axs[i] for i in range(2)]
        else:
            fig_setting = canvas_setting(8, 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1])
            axes_element = [axs[i] for i in range(2)]

    ## identify x-axis unit
    var_label = "wavelength" if unit and unit.lower() == "nm" else "energy"
    xaxis_label = "Photon wavelength (nm)" if var_label == "wavelength" else "Photon energy (eV)"

    ## systems information
    dataset = dielectric_systems_list(systems)

    ## suptitle
    if single_subfigure is True:
        plt.title(f"Dielectric function for {comp_aliase} {suptitle}", fontsize=fig_setting[3][0])
    else:
        fig.suptitle(f"Dielectric function for {comp_aliase} {suptitle}", fontsize=fig_setting[3][0])

    ## data boundary
    # photon_start, photon_end = process_boundary(boundary)

    ## data plotting
    if single_subfigure is True:
        plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # initialization
        wavelength_starts, wavelength_ends, energy_starts, energy_ends = [], [], [], []

        # component key
        current_component = comp_label.lower()
        data_key_real = f"density_{current_component}_real"
        data_key_imag = f"density_{current_component}_imag"

        for _, data in enumerate(dataset):
            energy_real, density_energy_real = extract_part(data[1]["density_energy_real"], data[1][data_key_real], source_start, source_end)
            energy_imag, density_energy_imag = extract_part(data[1]["density_energy_imag"], data[1][data_key_imag], source_start, source_end)
            if var_label == "energy":
                plt.plot(energy_real, density_energy_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                plt.plot(energy_imag, density_energy_imag, color=color_sampling(data[2])[1], ls="dashed", alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")
                energy_starts.append(min(energy_real))
                energy_ends.append(max(energy_real))
            else:
                wavelength_real, density_wl_real = extract_part(energy_to_wavelength(data[1]["density_energy_real"]), data[1][data_key_real], source_start, source_end)
                wavelength_imag, density_wl_imag = extract_part(energy_to_wavelength(data[1]["density_energy_imag"]), data[1][data_key_imag], source_start, source_end)
                plt.plot(wavelength_real, density_wl_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                plt.plot(wavelength_imag, density_wl_imag, color=color_sampling(data[2])[1], ls="dashed", alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")
                wavelength_starts.append(min(wavelength_real))
                wavelength_ends.append(np.max(np.array(wavelength_real)[np.isfinite(wavelength_real)]))

            # plasmon resonance line and rescale rate
            if var_label == "energy":
                plasmon_start = min(energy_starts)
                plasmon_end = max(energy_ends)
                plt.plot([plasmon_start, plasmon_end],[0,0], color=color_sampling("grey")[1], linestyle="dashed")
            else:
                plasmon_start=min(wavelength_starts)
                plasmon_end=max(wavelength_ends)
                plt.plot([plasmon_start, plasmon_end],[0,0],color=color_sampling("grey")[1],linestyle="dashed")

        plt.ylabel(r"Dielectric function")
        plt.xlabel(xaxis_label)

        # ## Start: visible lights
        # if var_label == "energy":
        #     # Define the visible light range in wavelength (in nm) and convert to energy (in eV)
        #     visible_wavelengths_nm = np.linspace(380, 750, 1000)  # 380 nm (violet) to 750 nm (red)
        #     visible_energies_eV = wavelength_to_energy(visible_wavelengths_nm)

        #     # Create a non-linear mapping for the color map
        #     cmap = plt.get_cmap("nipy_spectral")
        #     colors = cmap(np.linspace(0, 1, 1000))

        #     sorted_indices = np.argsort(visible_energies_eV)
        #     colors_energy_based = colors[sorted_indices]

        #     # No need to sort indices since energies are now increasing
        #     # Create a new colormap based on energy
        #     energy_cmap = ListedColormap(colors_energy_based)

        #     ax = plt.gca()

        #     visible_min = np.min(visible_energies_eV)
        #     visible_max = np.max(visible_energies_eV)

        #     gradient = np.linspace(0, 1, 1000).reshape(1, -1)
        #     gradient = np.vstack([gradient] * 10)

        #     x_min, x_max = ax.get_xlim()
        #     y_min, y_max = ax.get_ylim()

        # # Create an alpha gradient using a sine function for transparency (0 -> 0.4 -> 0)
        # alpha_vals = np.sin(np.linspace(0, np.pi, 1000)) * 2.0
        # alpha_vals = np.clip(alpha_vals, 0, 0.325)
        # alpha_gradient = alpha_vals.reshape(1, -1)
        # alpha_gradient = np.vstack([alpha_gradient] * 10)

        # plt.imshow(gradient,aspect="auto",extent=[visible_min, visible_max, y_min, y_max],cmap=energy_cmap,alpha=alpha_gradient)

        # plt.xlim(x_min, x_max)
        # plt.ylim(y_min, y_max)
        # ## End: visible lights

        plt.legend(loc="best")
        plt.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

        plt.tight_layout()

    elif all_expansion is True:
        supplot_subtitles = ["Original view for real part", "Original view for Imaginary",
                             "Rescaled view for real part", "Original view for Imaginary"]
        for subplot_index in range(4):
            ax = axes_element[subplot_index]
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

            # initialization
            wavelength_starts, wavelength_ends, energy_starts, energy_ends = [], [], [], []
            if subplot_index < 2:
                x_start = source_start
                x_end = source_end
            else:
                x_start = scaled_start
                x_end = scaled_end

            # component key
            current_component = comp_label.lower()
            data_key_real = f"density_{current_component}_real"
            data_key_imag = f"density_{current_component}_imag"

            # curve plotting: real part
            if subplot_index%2 == 0:
                # for each system
                for _, data in enumerate(dataset):
                    energy_real, density_energy_real = extract_part(data[1]["density_energy_real"], data[1][data_key_real], x_start, x_end)
                    if var_label == "energy":
                        ax.plot(energy_real, density_energy_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                        # plasmon resonance line for photon energy
                        energy_starts.append(min(energy_real))
                        energy_ends.append(max(energy_real))

                    else:
                        wavelength_real, density_wl_real = extract_part(energy_to_wavelength(data[1]["density_energy_real"]),data[1][data_key_real], x_start, x_end)
                        ax.plot(wavelength_real, density_wl_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                        # plasmon resonance line for photon wavelength
                        wavelength_starts.append(min(wavelength_real))
                        wavelength_ends.append(np.max(np.array(wavelength_real)[np.isfinite(wavelength_real)]))
                # plasmon resonance line
                if var_label == "energy":
                    energy_start=min(energy_starts)
                    energy_end=max(energy_ends)
                    ax.plot([energy_start, energy_end],[0,0],color=color_sampling("grey")[1],linestyle="--")
                else:
                    wavelength_start=min(wavelength_starts)
                    wavelength_end=max(wavelength_ends)
                    ax.plot([wavelength_start, wavelength_end],[0,0],color=color_sampling("grey")[1],linestyle="--")

            # curve plotting: imag part
            elif subplot_index%2 != 0:
                for _, data in enumerate(dataset):
                    energy_imag, density_energy_imag = extract_part(data[1]["density_energy_imag"], data[1][data_key_imag], x_start, x_end)
                    if var_label == "energy":
                        ax.plot(energy_imag, density_energy_imag, color=color_sampling(data[2])[2], ls=data[3], alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")
                    else:
                        wavelength_imag, density_wl_imag = extract_part(energy_to_wavelength(data[1]["density_energy_imag"]), data[1][data_key_imag], x_start, x_end)
                        ax.plot(wavelength_imag, density_wl_imag, color=color_sampling(data[2])[2], ls=data[3], alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")

            # subtitles and axis label (self-assertive): subtitles
            ax.set_title(supplot_subtitles[subplot_index])
            if layout_flag == "horizontal":
                if subplot_index%2 == 0:
                    ax.set_ylabel("Dielectric function")
                if subplot_index >= 2:
                    ax.set_xlabel(xaxis_label)
            else:
                if subplot_index < 2:
                    ax.set_ylabel("Dielectric function")
                if subplot_index%2 == 1:
                    ax.set_xlabel(xaxis_label)

        # Legend
        ax.legend(loc="best")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

        plt.tight_layout()

    elif expansion_flag is True and rescale_flag is False:
        supplot_subtitles = [f"Real part for {comp_aliase}",f"Imaginary part for {comp_aliase}"]
        for subplot_index in range(2):
            ax = axes_element[subplot_index]
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

            # initialization
            wavelength_starts, wavelength_ends, energy_starts, energy_ends = [], [], [], []
            x_start = source_start
            x_end = source_end

            # component key
            current_component = comp_label.lower()
            data_key_real = f"density_{current_component}_real"
            data_key_imag = f"density_{current_component}_imag"

             # curve plotting: real part
            if subplot_index%2 == 0:
                # for each system
                for _, data in enumerate(dataset):
                    energy_real, density_energy_real = extract_part(data[1]["density_energy_real"], data[1][data_key_real], x_start, x_end)
                    if var_label == "energy":
                        ax.plot(energy_real, density_energy_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                        # plasmon resonance line for photon energy
                        energy_starts.append(min(energy_real))
                        energy_ends.append(max(energy_real))

                    else:
                        wavelength_real, density_wl_real = extract_part(energy_to_wavelength(data[1]["density_energy_real"]),data[1][data_key_real], x_start, x_end)
                        ax.plot(wavelength_real, density_wl_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                        # plasmon resonance line for photon wavelength
                        wavelength_starts.append(min(wavelength_real))
                        wavelength_ends.append(np.max(np.array(wavelength_real)[np.isfinite(wavelength_real)]))
                # plasmon resonance line
                if var_label == "energy":
                    energy_start=min(energy_starts)
                    energy_end=max(energy_ends)
                    ax.plot([energy_start, energy_end],[0,0],color=color_sampling("grey")[1],linestyle="--")
                else:
                    wavelength_start=min(wavelength_starts)
                    wavelength_end=max(wavelength_ends)
                    ax.plot([wavelength_start, wavelength_end],[0,0],color=color_sampling("grey")[1],linestyle="--")

            # curve plotting: imag part
            elif subplot_index%2 != 0:
                for _, data in enumerate(dataset):
                    energy_imag, density_energy_imag = extract_part(data[1]["density_energy_imag"], data[1][data_key_imag], x_start, x_end)
                    if var_label == "energy":
                        ax.plot(energy_imag, density_energy_imag, color=color_sampling(data[2])[2], ls=data[3], alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")
                    else:
                        wavelength_imag, density_wl_imag = extract_part(energy_to_wavelength(data[1]["density_energy_imag"]), data[1][data_key_imag], x_start, x_end)
                        ax.plot(wavelength_imag, density_wl_imag, color=color_sampling(data[2])[2], ls=data[3], alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")

            # subtitles and axis label (self-assertive): subtitles
            ax.set_title(supplot_subtitles[subplot_index])
            if layout_flag == "horizontal":
                ax.set_xlabel(xaxis_label)
                if subplot_index == 0:
                    ax.set_ylabel("Dielectric function")
            else:
                ax.set_ylabel("Dielectric function")
                if subplot_index == 1:
                    ax.set_xlabel(xaxis_label)

        # Legend
        ax.legend(loc="best")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

        plt.tight_layout()

    elif expansion_flag is False and rescale_flag is True:
        supplot_subtitles = ["Original view", "Rescaled view"]
        for subplot_index in range(2):
            ax = axes_element[subplot_index]
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

            # initialization
            wavelength_starts, wavelength_ends, energy_starts, energy_ends = [], [], [], []
            if subplot_index == 0:
                x_start = source_start
                x_end = source_end
            else:
                x_start = scaled_start
                x_end = scaled_end

            # component key
            current_component = comp_label.lower()
            data_key_real = f"density_{current_component}_real"
            data_key_imag = f"density_{current_component}_imag"

            # curve plotting: real part and imaginary part for each system
            for _, data in enumerate(dataset):
                energy_real, density_energy_real = extract_part(data[1]["density_energy_real"], data[1][data_key_real], x_start, x_end)
                energy_imag, density_energy_imag = extract_part(data[1]["density_energy_imag"], data[1][data_key_imag], x_start, x_end)
                if var_label == "energy":
                    ax.plot(energy_real, density_energy_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                    ax.plot(energy_imag, density_energy_imag, color=color_sampling(data[2])[1], ls="dashed", alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")
                    energy_starts.append(min(energy_real))
                    energy_ends.append(max(energy_real))
                else:
                    wavelength_real, density_wl_real = extract_part(energy_to_wavelength(data[1]["density_energy_real"]), data[1][data_key_real], x_start, x_end)
                    wavelength_imag, density_wl_imag = extract_part(energy_to_wavelength(data[1]["density_energy_imag"]), data[1][data_key_imag], x_start, x_end)
                    ax.plot(wavelength_real, density_wl_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                    ax.plot(wavelength_imag, density_wl_imag, color=color_sampling(data[2])[1], ls="dashed", alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")
                    wavelength_starts.append(min(wavelength_real))
                    wavelength_ends.append(np.max(np.array(wavelength_real)[np.isfinite(wavelength_real)]))

            # plasmon resonance line and rescale rate
            if var_label == "energy":
                plasmon_start = min(energy_starts)
                plasmon_end = max(energy_ends)
                ax.plot([plasmon_start, plasmon_end],[0,0], color=color_sampling("grey")[1], linestyle="dashed")
            else:
                plasmon_start=min(wavelength_starts)
                plasmon_end=max(wavelength_ends)
                ax.plot([plasmon_start, plasmon_end],[0,0],color=color_sampling("grey")[1],linestyle="dashed")

            # subtitles and axis label (self-assertive): subtitles
            ax.set_title(supplot_subtitles[subplot_index])
            if layout_flag == "horizontal":
                ax.set_xlabel(xaxis_label)
                if subplot_index == 0:
                    ax.set_ylabel("Dielectric function")
            else:
                ax.set_ylabel("Dielectric function")
                if subplot_index == 1:
                    ax.set_xlabel(xaxis_label)

            # Legend
            ax.legend(loc="best")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

        plt.tight_layout()

def plot_dielectric_function(suptitle, systems=None, components=None,
                             layout="horizontal", expansion_label=True,
                             unit=None, boundary=(None,None), figure_size=(None,None)):
    ## Help information
    dielectric_help =  plot_dielectric_help()
    if suptitle in ["help", "Help"]:
        print(dielectric_help)
        return None

    ## multi components flag
    if isinstance(components, str) or isinstance(components, dict):
        return plot_dielectric_monocomp(suptitle, systems, components,layout, expansion_label,unit, boundary, figure_size)
    elif isinstance(components, list) and len(components) == 1:
        return plot_dielectric_monocomp(suptitle, systems, components,layout, expansion_label,unit, boundary, figure_size)

    ## expansion flag
    if isinstance(expansion_label, bool):
        expansion_flag = expansion_label
    elif expansion_label.lower() not in ["true", "yes", "t", "y", "combine"]:
        expansion_flag = False
    else:
        expansion_flag = True

    ## components aliases
    comp_labels, comp_aliases = [], []
    for comp in components:
        if isinstance(comp, dict):
            for key, value in comp.items():
                comp_labels.append(key.lower())
                comp_aliases.append(value)
        else:
            comp_labels.append(comp.lower())
            comp_aliases.append(f"{comp}-component")

    ## figure settings
    folding_flag = None
    allcomps_flag = None
    layout_flag = "horizontal" if layout.lower() not in ["vertical", "ver","v"] else "vertical"
    if expansion_flag is True:
        if layout_flag == "horizontal":
            fig_setting = canvas_setting(8*len(components), 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(2, len(components), figsize=fig_setting[0], dpi=fig_setting[1])
            axes_element = [axs[i, j] for j in range(len(components)) for i in range(2)] if len(components) != 1 else [axs[0], axs[1]]
        else:
            fig_setting = canvas_setting(16, 6*len(components)) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(len(components), 2, figsize=fig_setting[0], dpi=fig_setting[1])
            axes_element = [axs[i, j] for i in range(len(components)) for j in range(2)] if len(components) != 1 else [axs[0], axs[1]]
    elif expansion_flag is False and len(components) == 2:
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
    elif expansion_flag is False and len(components)%2 == 0:
        folding_flag = True
        if layout_flag == "horizontal":
            fig_setting = canvas_setting(8*len(components)/2, 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(2, int(len(components)/2), figsize=fig_setting[0], dpi=fig_setting[1])
            # axes_element = [axs[i, j] for j in range(int(len(components)/2)) for i in range(2)]
            axes_element = [axs[i, j] for i in range(2) for j in range(int(len(components)/2))]
        else:
            fig_setting = canvas_setting(16, 6*len(components)/2+1) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(int(len(components)/2), 2, figsize=fig_setting[0], dpi=fig_setting[1])
            axes_element = [axs[i, j] for j in range(2) for i in range(int(len(components)/2))]
    elif expansion_flag is False and len(components) == 9:
        allcomps_flag = True
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
    else:
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

    ## identify x-axis unit
    var_label = "wavelength" if unit and unit.lower() == "nm" else "energy"
    xaxis_label = "Photon wavelength (nm)" if var_label == "wavelength" else "Photon energy (eV)"

    ## systems information
    dataset = dielectric_systems_list(systems)

    ## suptitle
    fig.suptitle(f"Dielectric function {suptitle}\n", fontsize=fig_setting[3][0])

    ## data boundary
    photon_start, photon_end = process_boundary(boundary)

    ## data plotting
    # for each subplot
    if expansion_flag is True:
        for subplot_index in range(2*len(components)):
            ax = axes_element[subplot_index]
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

            # current component index and label
            component_index = subplot_index // 2
            current_component = comp_labels[component_index].lower()

            data_key = f"density_{current_component}_real" if subplot_index % 2 == 0 else f"density_{current_component}_imag"

            ## subtitles and axis label (self-assertive)
            # subtitles
            ax.set_title([f"Real part for {comp_aliases[component_index]}", f"Imaginary part for {comp_aliases[component_index]}"][subplot_index%2])
            # ylabel
            if layout_flag == "vertical" and subplot_index%2 == 0:
                ax.set_ylabel("Dielectric function")
            elif layout_flag == "horizontal" and subplot_index in range(2):
                ax.set_ylabel("Dielectric function")
            # xlabel
            if layout_flag == "vertical" and subplot_index >= 2*len(components)-2:
                ax.set_xlabel(xaxis_label)
            elif layout_flag == "horizontal" and subplot_index%2 == 1:
                ax.set_xlabel(xaxis_label)

            # initialization
            wavelength_starts, wavelength_ends, energy_starts, energy_ends = [], [], [], []

            # curve plotting: real part
            if subplot_index%2 == 0:
                # for each system
                for _, data in enumerate(dataset):
                    energy_real, density_energy_real = extract_part(data[1]["density_energy_real"], data[1][data_key], photon_start, photon_end)
                    if var_label == "energy":
                        ax.plot(energy_real, density_energy_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                        # plasmon resonance line for photon energy
                        energy_starts.append(min(energy_real))
                        energy_ends.append(max(energy_real))

                    else:
                        wavelength_real, density_wl_real = extract_part(energy_to_wavelength(data[1]["density_energy_real"]),data[1][data_key], photon_start, photon_end)
                        ax.plot(wavelength_real, density_wl_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                        # plasmon resonance line for photon wavelength
                        wavelength_starts.append(min(wavelength_real))
                        wavelength_ends.append(np.max(np.array(wavelength_real)[np.isfinite(wavelength_real)]))
                # plasmon resonance line
                if var_label == "energy":
                    energy_start=min(energy_starts)
                    energy_end=max(energy_ends)
                    ax.plot([energy_start, energy_end],[0,0],color=color_sampling("grey")[1],linestyle="--")
                else:
                    wavelength_start=min(wavelength_starts)
                    wavelength_end=max(wavelength_ends)
                    ax.plot([wavelength_start, wavelength_end],[0,0],color=color_sampling("grey")[1],linestyle="--")

            # curve plotting: imaginary part
            else:
                for _, data in enumerate(dataset):
                    energy_imag, density_energy_imag = extract_part(data[1]["density_energy_imag"], data[1][data_key], photon_start, photon_end)
                    if var_label == "energy":
                        ax.plot(energy_imag, density_energy_imag, color=color_sampling(data[2])[2], ls=data[3], alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")
                    else:
                        wavelength_imag, density_wl_imag = extract_part(energy_to_wavelength(data[1]["density_energy_imag"]), data[1][data_key], photon_start, photon_end)
                        ax.plot(wavelength_imag, density_wl_imag, color=color_sampling(data[2])[2], ls=data[3], alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")
            # Legend
            ax.legend(loc="best")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

    else:
        for subplot_index in range(len(components)):
            ax = axes_element[subplot_index]
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

            # initialization
            wavelength_starts, wavelength_ends, energy_starts, energy_ends = [], [], [], []

            # current component index and label
            component_index = subplot_index
            current_component = comp_labels[component_index].lower()
            data_key_real = f"density_{current_component}_real"
            data_key_imag = f"density_{current_component}_imag"

            # curve plotting: real part and imaginary part
            for _, data in enumerate(dataset):
                energy_real, density_energy_real = extract_part(data[1]["density_energy_real"], data[1][data_key_real], photon_start, photon_end)
                energy_imag, density_energy_imag = extract_part(data[1]["density_energy_imag"], data[1][data_key_imag], photon_start, photon_end)
                if var_label == "energy":
                    ax.plot(energy_real, density_energy_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                    ax.plot(energy_imag, density_energy_imag, color=color_sampling(data[2])[1], ls="dashed", alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")
                    energy_starts.append(min(energy_real))
                    energy_ends.append(max(energy_real))
                else:
                    wavelength_real, density_wl_real = extract_part(energy_to_wavelength(data[1]["density_energy_real"]), data[1][data_key_real], photon_start, photon_end)
                    wavelength_imag, density_wl_imag = extract_part(energy_to_wavelength(data[1]["density_energy_imag"]), data[1][data_key_imag], photon_start, photon_end)
                    ax.plot(wavelength_real, density_wl_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                    ax.plot(wavelength_imag, density_wl_imag, color=color_sampling(data[2])[1], ls="dashed", alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")
                    wavelength_starts.append(min(wavelength_real))
                    wavelength_ends.append(np.max(np.array(wavelength_real)[np.isfinite(wavelength_real)]))

            # plasmon resonance line and rescale rate
            if var_label == "energy":
                plasmon_start = min(energy_starts)
                plasmon_end = max(energy_ends)
                ax.plot([plasmon_start, plasmon_end],[0,0], color=color_sampling("grey")[1], linestyle="dashed")
            else:
                plasmon_start=min(wavelength_starts)
                plasmon_end=max(wavelength_ends)
                ax.plot([plasmon_start, plasmon_end],[0,0],color=color_sampling("grey")[1],linestyle="dashed")

            # subtitles and axis label (self-assertive): subtitles
            ax.set_title(comp_aliases[component_index])
            if allcomps_flag is True and layout_flag == "horizontal":
                if subplot_index in [0, len(components)/3, 2*len(components)/3]:
                    ax.set_ylabel("Dielectric function")
                if subplot_index >= 2*len(components)/3:
                    ax.set_xlabel(xaxis_label)
            elif allcomps_flag is True and layout_flag == "vertical":
                if subplot_index < len(components)/3:
                    ax.set_ylabel("Dielectric function")
                if subplot_index in [len(components)/3-1, 2*len(components)/3-1, len(components)-1]:
                    ax.set_xlabel(xaxis_label)
            elif folding_flag is True and layout_flag == "horizontal":
                if subplot_index in [0, len(components)/2]:
                    ax.set_ylabel("Dielectric function")
                if subplot_index >= len(components)/2:
                    ax.set_xlabel(xaxis_label)
            elif folding_flag is True and layout_flag == "vertical":
                if subplot_index < len(components)/2:
                    ax.set_ylabel("Dielectric function")
                if subplot_index in [len(components)/2-1, len(components)-1]:
                    ax.set_xlabel(xaxis_label)
            elif layout_flag == "vertical":
                ax.set_ylabel("Dielectric function")
                if layout_flag == "vertical" and subplot_index == len(components)-1:
                    ax.set_xlabel(xaxis_label)
            else:
                ax.set_xlabel(xaxis_label)
                if subplot_index == 0:
                    ax.set_ylabel("Dielectric function")

            ax.legend(loc="best")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

    plt.tight_layout()

def plot_dielectric_function_rescaled(suptitle, systems=None, components=None,
                                      layout="horizontal", unit=None, boundary=(None,None), figure_size=(None,None)):

    ## Help information
    help_info = "Usage: plot_dielectric_function_scaled \n" + \
                "\t Demonstrate dielectric function by each component \n" +\
                "The independent value includes \n" +\
                "\t suptitle: the suptitle; \n" +\
                "\t systems: dielectric function data list; \n" +\
                "\t components: planes ('xx'<default>, 'yy', 'zz', 'xy', 'yx', 'yz', 'zy', 'zx', 'xz'); \n" +\
                "\t layout: subfigures layout (horizontal<default>, vertical); \n" +\
                "\t unit: x-axis unit (eV<default>, nm); \n" +\
                "\t boundary: x-axis range, you can input a simple tuple, a nested tuple, or a simple tuple with a rate;\n" +\
                "\t figure_size: figure size <optional>. \n"
    if suptitle in ["help", "Help"]:
        print(help_info)
        return

    ## multi components flag
    if isinstance(components, str) or isinstance(components, dict):
        return plot_dielectric_monocomp(suptitle, systems, components,layout, unit, boundary, figure_size)
    elif isinstance(components, list) and len(components) == 1:
        return plot_dielectric_monocomp(suptitle, systems, components,layout, unit, boundary, figure_size)

    ## rescale flag and databoundaries
    rescale_flag, source_start, source_end, scaled_start, scaled_end = process_boundaries_rescaling(boundary)

    ## components aliases
    comp_labels, comp_aliases = [], []
    for comp in components:
        if isinstance(comp, dict):
            for key, value in comp.items():
                comp_labels.append(key.lower())
                comp_aliases.append(value)
        else:
            comp_labels.append(comp.lower())
            comp_aliases.append(f"{comp}-component")

    ## figure settings
    layout_flag = "horizontal" if layout.lower() not in ["vertical", "ver","v"] else "vertical"
    if rescale_flag is True:
        if layout_flag == "horizontal":
            fig_setting = canvas_setting(8*len(components), 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(2, len(components), figsize=fig_setting[0], dpi=fig_setting[1])
            axes_element = [axs[i, j] for j in range(len(components)) for i in range(2)] if len(components) != 1 else [axs[0], axs[1]]
        else:
            fig_setting = canvas_setting(16, 6*len(components)) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(len(components), 2, figsize=fig_setting[0], dpi=fig_setting[1])
            axes_element = [axs[i, j] for i in range(len(components)) for j in range(2)] if len(components) != 1 else [axs[0], axs[1]]
    else:
        return plot_dielectric_function(suptitle, systems, components, layout, False, unit, boundary, figure_size)

    ## identify x-axis unit
    var_label = "wavelength" if unit and unit.lower() == "nm" else "energy"
    xaxis_label = "Photon wavelength (nm)" if var_label == "wavelength" else "Photon energy (eV)"

    ## systems information
    dataset = dielectric_systems_list(systems)

    ## suptitle
    fig.suptitle(f"Dielectric function {suptitle}\n", fontsize=fig_setting[3][0])

    ## data plotting
    for subplot_index in range(2*len(components)):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # initialization
        wavelength_starts, wavelength_ends, energy_starts, energy_ends = [], [], [], []
        if subplot_index%2 == 0:
            x_start = source_start
            x_end = source_end
        else:
            x_start = scaled_start
            x_end = scaled_end

        # current component index and label
        component_index = subplot_index // 2
        current_component = comp_labels[component_index].lower()

        data_key_real = f"density_{current_component}_real"
        data_key_imag = f"density_{current_component}_imag"

        # curve plotting: real part and imaginary part for each system
        for _, data in enumerate(dataset):
            energy_real, density_energy_real = extract_part(data[1]["density_energy_real"], data[1][data_key_real], x_start, x_end)
            energy_imag, density_energy_imag = extract_part(data[1]["density_energy_imag"], data[1][data_key_imag], x_start, x_end)
            if var_label == "energy":
                ax.plot(energy_real, density_energy_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                ax.plot(energy_imag, density_energy_imag, color=color_sampling(data[2])[1], ls="dashed", alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")
                energy_starts.append(min(energy_real))
                energy_ends.append(max(energy_real))
            else:
                wavelength_real, density_wl_real = extract_part(energy_to_wavelength(data[1]["density_energy_real"]), data[1][data_key_real], x_start, x_end)
                wavelength_imag, density_wl_imag = extract_part(energy_to_wavelength(data[1]["density_energy_imag"]), data[1][data_key_imag], x_start, x_end)
                ax.plot(wavelength_real, density_wl_real, color=color_sampling(data[2])[1], ls=data[3], alpha=data[4], lw=data[5], label=f"Real part {data[0]}")
                ax.plot(wavelength_imag, density_wl_imag, color=color_sampling(data[2])[1], ls="dashed", alpha=data[4], lw=data[5], label=f"Imaginary part {data[0]}")
                wavelength_starts.append(min(wavelength_real))
                wavelength_ends.append(np.max(np.array(wavelength_real)[np.isfinite(wavelength_real)]))

        # plasmon resonance line and rescale rate
        if var_label == "energy":
            plasmon_start = min(energy_starts)
            plasmon_end = max(energy_ends)
            ax.plot([plasmon_start, plasmon_end],[0,0], color=color_sampling("grey")[1], linestyle="dashed")
        else:
            plasmon_start=min(wavelength_starts)
            plasmon_end=max(wavelength_ends)
            ax.plot([plasmon_start, plasmon_end],[0,0],color=color_sampling("grey")[1],linestyle="dashed")

        # subtitles and axis label (self-assertive): subtitles
        ax.set_title([f"Original view for {comp_aliases[component_index]}", f"Rescaled view for {comp_aliases[component_index]}"][subplot_index%2])

        # ylabel
        if layout_flag == "vertical" and subplot_index%2 == 0:
            ax.set_ylabel("Dielectric function")
        elif layout_flag == "horizontal" and subplot_index in range(2):
            ax.set_ylabel("Dielectric function")

        # xlabel
        if layout_flag == "vertical" and subplot_index >= 2*len(components)-2:
            ax.set_xlabel(xaxis_label)
        elif layout_flag == "horizontal" and subplot_index%2 == 1:
            ax.set_xlabel(xaxis_label)

        ax.legend(loc="best")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

    plt.tight_layout()
