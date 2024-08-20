#### Declarations of process functions for Dielectric function
# pylint: disable = C0103, C0114, C0116, C0301, R0914

### Necessary packages invoking

import xml.etree.ElementTree as ET
import numpy as np
import h5py
import os

from vmatplot.commons import extract_fermi, identify_algorithm

### Physical constants
hbar = 4.135667662e-15
c_vacuum = 2.99792458e8

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
    algorithm_type = identify_algorithm(directory)

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

    # ## Extract imaginary part of Density-Density
    # density_imag_path = f".//{data_label}[@comment='density-density']/imag/array/set"
    # density_imag_elements = root.findall(density_imag_path)
    # density_imag_columns = ["e_energy_imag_col",
    #                         "e_xx_imag_col", "e_yy_imag_col", "e_zz_imag_col",
    #                         "e_xy_imag_col", "e_yz_imag_col", "e_zx_imag_col"]
    # for _, element in enumerate(density_imag_elements[0:2]):
    #     for column in density_imag_columns:
    #         data[column] = []
    #     # Append data to lists
    #     for index in element.findall("r"):
    #         values = list(map(float, index.text.split()))
    #         for value_index, column in enumerate(density_imag_columns):
    #             data[column].append(values[value_index])
    #     # Convert lists to numpy arrays
    #     for column in density_imag_columns:
    #         data[column]=np.array(data[column])

    # ## Extract real part of Density-Density
    # density_real_path = f".//{data_label}[@comment='density-density']/real/array/set"
    # density_real_elements = root.findall(density_real_path)
    # density_real_columns = ["e_energy_real_col",
    #                         "e_xx_real_col", "e_yy_real_col", "e_zz_real_col",
    #                         "e_xy_real_col", "e_yz_real_col", "e_zx_real_col"]
    # for _, element in enumerate(density_real_elements[0:2]):
    #     for column in density_real_columns:
    #         data[column] = []
    #     # Append data to lists
    #     for index in element.findall("r"):
    #         values = list(map(float, index.text.split()))
    #         for value_index, column in enumerate(density_real_columns):
    #             data[column].append(values[value_index])
    #     # Convert lists to numpy arrays
    #     for column in density_real_columns:
    #         data[column]=np.array(data[column])

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
        "current_yz_real": current_list[1,2,:,0],
        "current_zx_real": current_list[2,0,:,0],
        "current_energy_imag": energy_list,
        "current_xx_imag": current_list[0,0,:,1],
        "current_yy_imag": current_list[1,1,:,1],
        "current_zz_imag": current_list[2,2,:,1],
        "current_xy_imag": current_list[0,1,:,1],
        "current_yz_imag": current_list[1,2,:,1],
        "current_zx_imag": current_list[2,0,:,1],
        "density_energy_real": energy_list,
        "density_xx_real": density_list[0,0,:,0],
        "density_yy_real": density_list[1,1,:,0],
        "density_zz_real": density_list[2,2,:,0],
        "density_xy_real": density_list[0,1,:,0],
        "density_yz_real": density_list[1,2,:,0],
        "density_zx_real": density_list[2,0,:,0],
        "density_energy_imag": energy_list,
        "density_xx_imag": density_list[0,0,:,1],
        "density_yy_imag": density_list[1,1,:,1],
        "density_zz_imag": density_list[2,2,:,1],
        "density_xy_imag": density_list[0,1,:,1],
        "density_yz_imag": density_list[1,2,:,1],
        "density_zx_imag": density_list[2,0,:,1],
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
        "current_yz_real": current_list[1,2,:,0],
        "current_zx_real": current_list[2,0,:,0],
        "current_energy_imag": energy_list,
        "current_xx_imag": current_list[0,0,:,1],
        "current_yy_imag": current_list[1,1,:,1],
        "current_zz_imag": current_list[2,2,:,1],
        "current_xy_imag": current_list[0,1,:,1],
        "current_yz_imag": current_list[1,2,:,1],
        "current_zx_imag": current_list[2,0,:,1],
        "density_energy_real": energy_list,
        "density_xx_real": density_list[0,0,:,0],
        "density_yy_real": density_list[1,1,:,0],
        "density_zz_real": density_list[2,2,:,0],
        "density_xy_real": density_list[0,1,:,0],
        "density_yz_real": density_list[1,2,:,0],
        "density_zx_real": density_list[2,0,:,0],
        "density_energy_imag": energy_list,
        "density_xx_imag": density_list[0,0,:,1],
        "density_yy_imag": density_list[1,1,:,1],
        "density_zz_imag": density_list[2,2,:,1],
        "density_xy_imag": density_list[0,1,:,1],
        "density_yz_imag": density_list[1,2,:,1],
        "density_zx_imag": density_list[2,0,:,1],
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

def plot_dielectric_function(suptitle, matters_list=None, components=None, comp_aliases=None,
                             unit=None, layout=None, energy_boundary=(None, None)):

    return 0
