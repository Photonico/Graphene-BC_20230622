#### Declarations of process functions for PDoS with vectorized programming
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# Necessary packages invoking
import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vmatplot.commons import extract_fermi, get_elements
from vmatplot.output import canvas_setting, color_sampling

def cal_type_pdos(directory_path):
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        return "GGA-PBE"
    elif os.path.exists(kpoints_file_path):
        return "HSE06"

# Extract Kpoints number
def extract_kpoints_number(directory_path):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    tree = ET.parse(file_path)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    ## Extract the number of kpoints
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        kpointlist = root.find(".//eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']")
        kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
        kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
        kpointlist_matrix = kpointlist_array.reshape(-1, 3)
        kpoints_number = kpointlist_matrix.shape[0]
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        kpointlist = root.find(".//varray[@name='kpointlist']")
        kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
        kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
        kpointlist_matrix = kpointlist_array.reshape(-1, 3)
        kpoints_number = kpointlist_matrix.shape[0]
    return kpoints_number

## Extract eigen, occupancy number
def extract_eigen_occupancy(directory_path):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    tree = ET.parse(file_path)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    kpoints_number = extract_kpoints_number(directory_path)
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for kpoints_index in range(1, kpoints_number+1):
            xpath_expr = f"./calculation/projected_kpoints_opt/eigenvalues/array/set/set[@comment='spin 1']/set[@comment='kpoint {kpoints_index}']"
            eigen_column = np.empty(0)
            occu_column  = np.empty(0)
            kpoint_set = root.find(xpath_expr)
            for eigen_occ_element in kpoint_set:
                eigen_values = list(map(float, eigen_occ_element.text.split()))
                eigen_column = np.append(eigen_column, eigen_values[0])
                occu_column = np.append(occu_column, eigen_values[1])
            if kpoints_index == 1 :
                eigen_matrix = eigen_column.reshape(-1, 1)
                occu_matrix = occu_column.reshape(-1, 1)
            else:
                eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
                occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        for kpoints_index in range(1, kpoints_number+1):
            xpath_expr = f".//set[@comment='kpoint {kpoints_index}']"
            eigen_column = np.empty(0)
            occu_column  = np.empty(0)
            kpoint_set = root.find(xpath_expr)
            for eigen_occ_element in kpoint_set:
                eigen_values = list(map(float, eigen_occ_element.text.split()))
                eigen_column = np.append(eigen_column, eigen_values[0])
                occu_column = np.append(occu_column, eigen_values[1])
            if kpoints_index == 1 :
                eigen_matrix = eigen_column.reshape(-1, 1)
                occu_matrix = occu_column.reshape(-1, 1)
            else:
                eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
                occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))
    return (eigen_matrix, occu_matrix)

# Extract energy list
def extract_energy_list(directory_path):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    tree = ET.parse(file_path)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    ## Initialization
    energy_dos_list     = np.array([])
    efermi = extract_fermi(directory_path)
    if os.path.exists(kpoints_opt_path):
        path_dos = "./calculation/dos[@comment='kpoints_opt']/total/array/set/set[@comment='spin 1']/r"
    elif os.path.exists(kpoints_file_path):
        path_dos = ".//total/array/set/set[@comment='spin 1']/r"
    for element_dos in root.findall(path_dos):
        values_dos = list(map(float, element_dos.text.split()))
        energy_dos_list = np.append(energy_dos_list, values_dos[0])
    shift = efermi
    return energy_dos_list

def extract_energy_shift(directory_path):
    energy_dos_list = extract_energy_list(directory_path)
    shift = extract_fermi(directory_path)
    energy_dos_shift = energy_dos_list - shift
    return energy_dos_shift

# Total PDoS: univseral elements and layers
def extract_pdos(directory_path):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    # Check if the vasprun.xml file exists in the given directory
    if not os.path.isfile(file_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")
        return

    ## Analysis vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")

    ## Extract Fermi energy
    # efermi_element = root.find(".//dos/i[@name='efermi']")
    # efermi = float(efermi_element.text.strip())
    efermi = extract_fermi(directory_path)

    ## Extract the number of ions
    first_positions = root.find(".//varray[@name='positions'][1]")
    positions_concatenated_text = " ".join([position.text for position in first_positions.findall("v")])
    positions_array = np.fromstring(positions_concatenated_text, sep=" ")
    positions_matrix = positions_array.reshape(-1, 3)
    ions_number = positions_matrix.shape[0]

    ## Extract the number of kpoints
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        kpointlist = root.find(".//eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']")
        kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
        kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
        kpointlist_matrix = kpointlist_array.reshape(-1, 3)
        kpoints_number = kpointlist_matrix.shape[0]
    # PBE algorithms
    elif os.path.exists(kpoints_file_path):
        kpointlist = root.find(".//varray[@name='kpointlist']")
        kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
        kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
        kpointlist_matrix = kpointlist_array.reshape(-1, 3)
        kpoints_number = kpointlist_matrix.shape[0]

    ## Extract eigen, occupancy number
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for kpoints_index in range(1, kpoints_number+1):
            xpath_expr = f"./calculation/projected_kpoints_opt/eigenvalues/array/set/set[@comment='spin 1']/set[@comment='kpoint {kpoints_index}']"
            eigen_column = np.empty(0)
            occu_column  = np.empty(0)
            kpoint_set = root.find(xpath_expr)
            for eigen_occ_element in kpoint_set:
                values_eigen = list(map(float, eigen_occ_element.text.split()))
                eigen_var = values_eigen[0]
                eigen_column = np.append(eigen_column, eigen_var)
                occu_var = values_eigen[1]
                occu_column = np.append(occu_column, occu_var)
            if kpoints_index == 1 :
                eigen_matrix = eigen_column.reshape(-1, 1)
                occu_matrix = occu_column.reshape(-1, 1)
            else:
                eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
                occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        for kpoints_index in range(1, kpoints_number+1):
            xpath_expr = f".//set[@comment='kpoint {kpoints_index}']"
            eigen_column = np.empty(0)
            occu_column  = np.empty(0)
            kpoint_set = root.find(xpath_expr)
            for eigen_occ_element in kpoint_set:
                values_eigen = list(map(float, eigen_occ_element.text.split()))
                eigen_var = values_eigen[0]
                eigen_column = np.append(eigen_column, eigen_var)
                occu_var = values_eigen[1]
                occu_column = np.append(occu_column, occu_var)
            if kpoints_index == 1 :
                eigen_matrix = eigen_column.reshape(-1, 1)
                occu_matrix = occu_column.reshape(-1, 1)
            else:
                eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
                occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))

    ## Extract energy, total DoS, and integrated DoS
    # lists initialization
    energy_dos_list     = np.array([])
    total_dos_list      = np.array([])
    integrated_dos_list = np.array([])

    if os.path.exists(kpoints_opt_path):
        path_dos_spin_1 = "./calculation/dos[@comment='kpoints_opt']/total/array/set/set[@comment='spin 1']/r"
        path_dos_spin_2 = "./calculation/dos[@comment='kpoints_opt']/total/array/set/set[@comment='spin 2']/r"
    elif os.path.exists(kpoints_file_path):
        path_dos_spin_1 = ".//total/array/set/set[@comment='spin 1']/r"
        path_dos_spin_2 = ".//total/array/set/set[@comment='spin 2']/r"

    spin2_exists = root.find(path_dos_spin_2) is not None

    for element_dos in root.findall(path_dos_spin_1):
        dos_values = list(map(float, element_dos.text.split()))
        energy_var = dos_values[0]
        energy_dos_list = np.append(energy_dos_list, energy_var)
        total_dos_var = dos_values[1]
        total_dos_list = np.append(total_dos_list, total_dos_var)
        integrated_dos_var = dos_values[2]
        integrated_dos_list = np.append(integrated_dos_list, integrated_dos_var)
    shift = efermi
    energy_dos_shift = energy_dos_list - shift

    ## Extract energy, s-PDoS, p_y-PDoS, p_z-PDoS, p_x-PDoS, d_xy-PDoS, d_yz-PDoS, d_z2-PDoS, d_XXZZ-PDoS, x2-y2-PDoS
    # Matrices initialization
    for ions_index in range(1, ions_number + 1):
        path_ions = f".//set[@comment='ion {ions_index}']/set[@comment='spin 1']/r"
        # Columns initialization
        energy_pdos_column  = np.empty(0)
        s_pdos_column       = np.empty(0)
        p_y_pdos_column     = np.empty(0)
        p_z_pdos_column     = np.empty(0)
        p_x_pdos_column     = np.empty(0)
        d_xy_pdos_column    = np.empty(0)
        d_yz_pdos_column    = np.empty(0)
        d_z2_pdos_column    = np.empty(0)
        d_XXZZ_pdos_column    = np.empty(0)
        x2_y2_pdos_column   = np.empty(0)
        for pdos_element in root.findall(path_ions):
            pdos_values = list(map(float, pdos_element.text.split()))
            # Columns of energy
            energy_pdos_column = np.append(energy_pdos_column, pdos_values[0])
            # Columns of s-PDoS
            s_pdos_column = np.append(s_pdos_column, pdos_values[1])
            # Columns of p_y-PDoS
            p_y_pdos_column = np.append(p_y_pdos_column, pdos_values[2])
            # Columns of p_z-PDoS
            p_z_pdos_column = np.append(p_z_pdos_column, pdos_values[3])
            # Columns of p_x-PDoS
            p_x_pdos_column = np.append(p_x_pdos_column, pdos_values[4])
            # Columns of d_xy-PDoS
            d_xy_pdos_column = np.append(d_xy_pdos_column, pdos_values[5])
            # Columns of d_yz-PDoS
            d_yz_pdos_column = np.append(d_yz_pdos_column, pdos_values[6])
            # Columns of d_z2-PDoS
            d_z2_pdos_column = np.append(d_z2_pdos_column, pdos_values[7])
            # Columns of d_XXZZ-PDoS
            d_XXZZ_pdos_column = np.append(d_XXZZ_pdos_column, pdos_values[8])
            # Columns of x2-y2-PDoS
            x2_y2_pdos_column = np.append(x2_y2_pdos_column, pdos_values[9])
        if ions_index == 1:
            energy_pdos_matrix = energy_pdos_column.reshape(-1, 1)
            s_pdos_matrix = s_pdos_column.reshape(-1, 1)
            p_y_pdos_matrix = p_y_pdos_column.reshape(-1, 1)
            p_z_pdos_matrix = p_z_pdos_column.reshape(-1, 1)
            p_x_pdos_matrix = p_x_pdos_column.reshape(-1, 1)
            d_xy_pdos_matrix = d_xy_pdos_column.reshape(-1, 1)
            d_yz_pdos_matrix = d_yz_pdos_column.reshape(-1, 1)
            d_z2_pdos_matrix = d_z2_pdos_column.reshape(-1, 1)
            d_XXZZ_pdos_matrix = d_XXZZ_pdos_column.reshape(-1, 1)
            x2_y2_pdos_matrix = x2_y2_pdos_column.reshape(-1, 1)
        else:
            energy_pdos_matrix = np.hstack((energy_pdos_matrix, energy_pdos_column.reshape(-1, 1)))
            s_pdos_matrix = np.hstack((s_pdos_matrix, s_pdos_column.reshape(-1, 1)))
            p_y_pdos_matrix = np.hstack((p_y_pdos_matrix, p_y_pdos_column.reshape(-1, 1)))
            p_z_pdos_matrix = np.hstack((p_z_pdos_matrix, p_z_pdos_column.reshape(-1, 1)))
            p_x_pdos_matrix = np.hstack((p_x_pdos_matrix, p_x_pdos_column.reshape(-1, 1)))
            d_xy_pdos_matrix = np.hstack((d_xy_pdos_matrix, d_xy_pdos_column.reshape(-1, 1)))
            d_yz_pdos_matrix = np.hstack((d_yz_pdos_matrix, d_yz_pdos_column.reshape(-1, 1)))
            d_z2_pdos_matrix = np.hstack((d_z2_pdos_matrix, d_z2_pdos_column.reshape(-1, 1)))
            d_XXZZ_pdos_matrix = np.hstack((d_XXZZ_pdos_matrix, d_XXZZ_pdos_column.reshape(-1, 1)))
            x2_y2_pdos_matrix = np.hstack((x2_y2_pdos_matrix, x2_y2_pdos_column.reshape(-1, 1)))
    energy_pdos_sum = energy_pdos_matrix[:,0]
    s_pdos_sum = np.sum(s_pdos_matrix, axis=1)
    p_y_pdos_sum = np.sum(p_y_pdos_matrix, axis=1)
    p_z_pdos_sum = np.sum(p_z_pdos_matrix, axis=1)
    p_x_pdos_sum = np.sum(p_x_pdos_matrix, axis=1)
    d_xy_pdos_sum = np.sum(d_xy_pdos_matrix, axis=1)
    d_yz_pdos_sum = np.sum(d_yz_pdos_matrix, axis=1)
    d_z2_pdos_sum = np.sum(d_z2_pdos_matrix, axis=1)
    d_XXZZ_pdos_sum = np.sum(d_XXZZ_pdos_matrix, axis=1)
    x2_y2_pdos_sum = np.sum(x2_y2_pdos_matrix, axis=1)
    energy_pdos_shift = energy_pdos_sum - shift
    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
            energy_dos_shift, total_dos_list, integrated_dos_list,                      # 5 ~ 7
            energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
            d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_XXZZ_pdos_sum,                 # 13 ~ 16
            x2_y2_pdos_sum)

# Extract PDoS for elements
def extract_element_pdos(directory_path, element):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    # Check if the vasprun.xml file exists in the given directory
    if not os.path.isfile(file_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")
        return

    ## Analysis vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()
    # kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    # kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")

    ## Extract Fermi energy
    # efermi_element = root.find(".//dos/i[@name='efermi']")
    # efermi = float(efermi_element.text.strip())
    efermi = extract_fermi(directory_path)

    ## Extract the number of ions
    first_positions = root.find(".//varray[@name='positions'][1]")
    positions_concatenated_text = " ".join([position.text for position in first_positions.findall("v")])
    positions_array = np.fromstring(positions_concatenated_text, sep=" ")
    positions_matrix = positions_array.reshape(-1, 3)
    ions_number = positions_matrix.shape[0]

    ## Analysis elements
    index_start = get_elements(directory_path)[element][0]
    index_end = get_elements(directory_path)[element][1]

    ## Extract the number of kpoints
    kpoints_number =extract_kpoints_number(directory_path)

    ## Extract eigen, occupancy number
    eigen_matrix = extract_eigen_occupancy(directory_path)[0]
    occu_matrix  = extract_eigen_occupancy(directory_path)[1]

    ## Extract energy list
    # lists initialization
    total_pdos_list     = np.array([])
    integrated_dos_list = np.array([])

    shift = extract_fermi(directory_path)
    energy_dos_shift = extract_energy_shift(directory_path)

    ## Extract energy, s-PDoS, p_y-PDoS, p_z-PDoS, p_x-PDoS, d_xy-PDoS, d_yz-PDoS, d_z2-PDoS, d_XXZZ-PDoS, x2-y2-PDoS
    # Matrices initialization
    for ions_index in range(index_start, index_end + 1):
        path_ions = f".//set[@comment='ion {ions_index}']/set[@comment='spin 1']/r"
        # Columns initialization
        energy_pdos_column      = np.array([])
        s_pdos_column           = np.array([])
        p_y_pdos_column         = np.array([])
        p_z_pdos_column         = np.array([])
        p_x_pdos_column         = np.array([])
        d_xy_pdos_column        = np.array([])
        d_yz_pdos_column        = np.array([])
        d_z2_pdos_column        = np.array([])
        d_XXZZ_pdos_column        = np.array([])
        x2_y2_pdos_column       = np.array([])
        for pdos_element in root.findall(path_ions):
            pdos_values = list(map(float, pdos_element.text.split()))
            # Columns of energy
            energy_pdos_column = np.append(energy_pdos_column, pdos_values[0])
            # Columns of s-PDoS
            s_pdos_column = np.append(s_pdos_column, pdos_values[1])
            # Columns of p_y-PDoS
            p_y_pdos_column = np.append(p_y_pdos_column, pdos_values[2])
            # Columns of p_z-PDoS
            p_z_pdos_column = np.append(p_z_pdos_column, pdos_values[3])
            # Columns of p_x-PDoS
            p_x_pdos_column = np.append(p_x_pdos_column, pdos_values[4])
            # Columns of d_xy-PDoS
            d_xy_pdos_column = np.append(d_xy_pdos_column, pdos_values[5])
            # Columns of d_yz-PDoS
            d_yz_pdos_column = np.append(d_yz_pdos_column, pdos_values[6])
            # Columns of d_z2-PDoS
            d_z2_pdos_column = np.append(d_z2_pdos_column, pdos_values[7])
            # Columns of d_XXZZ-PDoS
            d_XXZZ_pdos_column = np.append(d_XXZZ_pdos_column, pdos_values[8])
            # Columns of x2-y2-PDoS
            x2_y2_pdos_column = np.append(x2_y2_pdos_column, pdos_values[9])
        if ions_index == index_start:
            energy_pdos_matrix = energy_pdos_column.reshape(-1, 1)
            s_pdos_matrix = s_pdos_column.reshape(-1, 1)
            p_y_pdos_matrix = p_y_pdos_column.reshape(-1, 1)
            p_z_pdos_matrix = p_z_pdos_column.reshape(-1, 1)
            p_x_pdos_matrix = p_x_pdos_column.reshape(-1, 1)
            d_xy_pdos_matrix = d_xy_pdos_column.reshape(-1, 1)
            d_yz_pdos_matrix = d_yz_pdos_column.reshape(-1, 1)
            d_z2_pdos_matrix = d_z2_pdos_column.reshape(-1, 1)
            d_XXZZ_pdos_matrix = d_XXZZ_pdos_column.reshape(-1, 1)
            x2_y2_pdos_matrix = x2_y2_pdos_column.reshape(-1, 1)
        else:
            energy_pdos_matrix = np.hstack((energy_pdos_matrix, energy_pdos_column.reshape(-1, 1)))
            s_pdos_matrix = np.hstack((s_pdos_matrix, s_pdos_column.reshape(-1, 1)))
            p_y_pdos_matrix = np.hstack((p_y_pdos_matrix, p_y_pdos_column.reshape(-1, 1)))
            p_z_pdos_matrix = np.hstack((p_z_pdos_matrix, p_z_pdos_column.reshape(-1, 1)))
            p_x_pdos_matrix = np.hstack((p_x_pdos_matrix, p_x_pdos_column.reshape(-1, 1)))
            d_xy_pdos_matrix = np.hstack((d_xy_pdos_matrix, d_xy_pdos_column.reshape(-1, 1)))
            d_yz_pdos_matrix = np.hstack((d_yz_pdos_matrix, d_yz_pdos_column.reshape(-1, 1)))
            d_z2_pdos_matrix = np.hstack((d_z2_pdos_matrix, d_z2_pdos_column.reshape(-1, 1)))
            d_XXZZ_pdos_matrix = np.hstack((d_XXZZ_pdos_matrix, d_XXZZ_pdos_column.reshape(-1, 1)))
            x2_y2_pdos_matrix = np.hstack((x2_y2_pdos_matrix, x2_y2_pdos_column.reshape(-1, 1)))
    energy_pdos_sum = energy_pdos_matrix[:,0]
    s_pdos_sum = np.sum(s_pdos_matrix, axis=1)
    p_y_pdos_sum = np.sum(p_y_pdos_matrix, axis=1)
    p_z_pdos_sum = np.sum(p_z_pdos_matrix, axis=1)
    p_x_pdos_sum = np.sum(p_x_pdos_matrix, axis=1)
    d_xy_pdos_sum = np.sum(d_xy_pdos_matrix, axis=1)
    d_yz_pdos_sum = np.sum(d_yz_pdos_matrix, axis=1)
    d_z2_pdos_sum = np.sum(d_z2_pdos_matrix, axis=1)
    d_XXZZ_pdos_sum = np.sum(d_XXZZ_pdos_matrix, axis=1)
    x2_y2_pdos_sum = np.sum(x2_y2_pdos_matrix, axis=1)
    total_pdos_list = s_pdos_sum + p_y_pdos_sum + p_z_pdos_sum + p_x_pdos_sum + d_xy_pdos_sum + d_yz_pdos_sum + d_z2_pdos_sum + d_XXZZ_pdos_sum + x2_y2_pdos_sum
    integrated_dos_list = np.trapz(total_pdos_list, x = energy_dos_shift)
    energy_pdos_shift = energy_pdos_sum - shift
    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
            energy_dos_shift,                                                           # 5
            total_pdos_list, integrated_dos_list,                                       # 6 ~ 7
            energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
            d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_XXZZ_pdos_sum,                 # 13 ~ 16
            x2_y2_pdos_sum)

# PDoS for customized range
def extract_segment_pdos(directory_path, start, end):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    # Check if the vasprun.xml file exists in the given directory
    if not os.path.isfile(file_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")
        return

    ## Analysis vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()

    ## Analysis elements
    index_start = start
    index_end = end

    ## Extract Fermi energy
    efermi = extract_fermi(directory_path)

    ## Extract the number of ions
    first_positions = root.find(".//varray[@name='positions'][1]")
    positions_concatenated_text = " ".join([position.text for position in first_positions.findall("v")])
    positions_array = np.fromstring(positions_concatenated_text, sep=" ")
    positions_matrix = positions_array.reshape(-1, 3)
    ions_number = positions_matrix.shape[0]

    ## Extract the number of kpoints
    kpoints_number =extract_kpoints_number(directory_path)

    ## Extract eigen, occupancy number
    ## Extract eigen, occupancy number
    eigen_matrix = extract_eigen_occupancy(directory_path)[0]
    occu_matrix  = extract_eigen_occupancy(directory_path)[1]
    # eigen_sum = np.sum(eigen_matrix, axis=1)
    # occu_sum  = np.sum(occu_matrix, axis=1)

    ## Extract energy list
    # lists initialization
    total_pdos_list     = np.array([])
    integrated_dos_list = np.array([])

    shift = extract_fermi(directory_path)
    energy_dos_shift = extract_energy_shift(directory_path)

    ## Extract energy, s-PDoS, p_y-PDoS, p_z-PDoS, p_x-PDoS, d_xy-PDoS, d_yz-PDoS, d_z2-PDoS, d_XXZZ-PDoS, x2-y2-PDoS
    # Matrices initialization
    for ions_index in range(index_start, index_end + 1):
        path_ions = f".//set[@comment='ion {ions_index}']/set[@comment='spin 1']/r"
        # Columns initialization
        energy_pdos_column  = np.empty(0)
        s_pdos_column       = np.empty(0)
        p_y_pdos_column     = np.empty(0)
        p_z_pdos_column     = np.empty(0)
        p_x_pdos_column     = np.empty(0)
        d_xy_pdos_column    = np.empty(0)
        d_yz_pdos_column    = np.empty(0)
        d_z2_pdos_column    = np.empty(0)
        d_XXZZ_pdos_column    = np.empty(0)
        x2_y2_pdos_column   = np.empty(0)
        for pdos_element in root.findall(path_ions):
            pdos_values = list(map(float, pdos_element.text.split()))
            # Columns of energy
            energy_pdos_column = np.append(energy_pdos_column, pdos_values[0])
            # Columns of s-PDoS
            s_pdos_column = np.append(s_pdos_column, pdos_values[1])
            # Columns of p_y-PDoS
            p_y_pdos_column = np.append(p_y_pdos_column, pdos_values[2])
            # Columns of p_z-PDoS
            p_z_pdos_column = np.append(p_z_pdos_column, pdos_values[3])
            # Columns of p_x-PDoS
            p_x_pdos_column = np.append(p_x_pdos_column, pdos_values[4])
            # Columns of d_xy-PDoS
            d_xy_pdos_column = np.append(d_xy_pdos_column, pdos_values[5])
            # Columns of d_yz-PDoS
            d_yz_pdos_column = np.append(d_yz_pdos_column, pdos_values[6])
            # Columns of d_z2-PDoS
            d_z2_pdos_column = np.append(d_z2_pdos_column, pdos_values[7])
            # Columns of d_XXZZ-PDoS
            d_XXZZ_pdos_column = np.append(d_XXZZ_pdos_column, pdos_values[8])
            # Columns of x2-y2-PDoS
            x2_y2_pdos_column = np.append(x2_y2_pdos_column, pdos_values[9])
        if ions_index == index_start:
            energy_pdos_matrix = energy_pdos_column.reshape(-1, 1)
            s_pdos_matrix = s_pdos_column.reshape(-1, 1)
            p_y_pdos_matrix = p_y_pdos_column.reshape(-1, 1)
            p_z_pdos_matrix = p_z_pdos_column.reshape(-1, 1)
            p_x_pdos_matrix = p_x_pdos_column.reshape(-1, 1)
            d_xy_pdos_matrix = d_xy_pdos_column.reshape(-1, 1)
            d_yz_pdos_matrix = d_yz_pdos_column.reshape(-1, 1)
            d_z2_pdos_matrix = d_z2_pdos_column.reshape(-1, 1)
            d_XXZZ_pdos_matrix = d_XXZZ_pdos_column.reshape(-1, 1)
            x2_y2_pdos_matrix = x2_y2_pdos_column.reshape(-1, 1)
        else:
            energy_pdos_matrix = np.hstack((energy_pdos_matrix, energy_pdos_column.reshape(-1, 1)))
            s_pdos_matrix = np.hstack((s_pdos_matrix, s_pdos_column.reshape(-1, 1)))
            p_y_pdos_matrix = np.hstack((p_y_pdos_matrix, p_y_pdos_column.reshape(-1, 1)))
            p_z_pdos_matrix = np.hstack((p_z_pdos_matrix, p_z_pdos_column.reshape(-1, 1)))
            p_x_pdos_matrix = np.hstack((p_x_pdos_matrix, p_x_pdos_column.reshape(-1, 1)))
            d_xy_pdos_matrix = np.hstack((d_xy_pdos_matrix, d_xy_pdos_column.reshape(-1, 1)))
            d_yz_pdos_matrix = np.hstack((d_yz_pdos_matrix, d_yz_pdos_column.reshape(-1, 1)))
            d_z2_pdos_matrix = np.hstack((d_z2_pdos_matrix, d_z2_pdos_column.reshape(-1, 1)))
            d_XXZZ_pdos_matrix = np.hstack((d_XXZZ_pdos_matrix, d_XXZZ_pdos_column.reshape(-1, 1)))
            x2_y2_pdos_matrix = np.hstack((x2_y2_pdos_matrix, x2_y2_pdos_column.reshape(-1, 1)))
    energy_pdos_sum = energy_pdos_matrix[:,0]
    s_pdos_sum = np.sum(s_pdos_matrix, axis=1)
    p_y_pdos_sum = np.sum(p_y_pdos_matrix, axis=1)
    p_z_pdos_sum = np.sum(p_z_pdos_matrix, axis=1)
    p_x_pdos_sum = np.sum(p_x_pdos_matrix, axis=1)
    d_xy_pdos_sum = np.sum(d_xy_pdos_matrix, axis=1)
    d_yz_pdos_sum = np.sum(d_yz_pdos_matrix, axis=1)
    d_z2_pdos_sum = np.sum(d_z2_pdos_matrix, axis=1)
    d_XXZZ_pdos_sum = np.sum(d_XXZZ_pdos_matrix, axis=1)
    x2_y2_pdos_sum = np.sum(x2_y2_pdos_matrix, axis=1)
    energy_pdos_shift = energy_pdos_sum - shift
    total_pdos_list = s_pdos_sum + p_y_pdos_sum + p_z_pdos_sum + p_x_pdos_sum + d_xy_pdos_sum + d_yz_pdos_sum + d_z2_pdos_sum + d_XXZZ_pdos_sum + x2_y2_pdos_sum
    integrated_dos_list = np.trapz(total_pdos_list, x=energy_dos_shift)

    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
            energy_dos_shift, total_pdos_list, integrated_dos_list,                     # 5 ~ 7
            energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
            d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_XXZZ_pdos_sum,                 # 13 ~ 16
            x2_y2_pdos_sum)

def create_matters_totpdos(matters_list):
    # matters = create_matters_dos(matters_list)
    # matters[0] = label
    # matters[1][0] = energy: x-axis
    # matters[1][6] = Total DoS
    # matters[1][7] = integral dos
    # matters[2] = color family
    # matters[3] = alpha
    # matters[4] = linestyle
    matters = []
    for matter_dir in matters_list:
        if len(matter_dir) == 2:
            label, directory = matter_dir
            color = "blue"
            alpha = 1.0
            linestyle = "solid"
        elif len(matter_dir) == 3:
            label, directory, color = matter_dir
            alpha = 1.0
            linestyle = "solid"
        elif len(matter_dir) == 4:
            label, directory, color, alpha = matter_dir
            linestyle = "solid"
        else:
            label, directory, color, alpha, linestyle = matter_dir
        dos_data = extract_pdos(directory)
        matters.append([label, dos_data, color, alpha, linestyle])
    return matters

def plot_totpdos(title, x_range = None, y_top = None, pdos_type = None, matters_list = None):
    # Help information
    help_info = "Usage: plot_dos \n" + \
                "Use extract_dos to extract the DoS data into a two-dimensional list firstly.\n"

    if title.lower() == "help":
        print(help_info)
        return

    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Color calling
    fermi_color = color_sampling("Violet")

    # Matter list
    matters = create_matters_totpdos(matters_list)
    for _, matter in enumerate(matters):
        # label
        current_label = matter[0]
        if pdos_type in ["All", "all"]:
            # for _, matter in enumerate(matters):
            plt.plot(matter[1][8], matter[1][6], c=color_sampling(matter[2])[1], label=f"Total PDoS {current_label}", alpha=matter[3], linestyle=matter[4], zorder = 2)
            plt.plot(matter[1][8], matter[1][7], c=color_sampling(matter[2])[2], label=f"Integrated PDoS {current_label}", alpha=matter[3], linestyle=matter[4], zorder = 3)
            efermi = matter[1][0]
        if pdos_type in ["Total", "total"]:
            # for _, matter in enumerate(matters):
            plt.plot(matter[1][8], matter[1][6], c=color_sampling(matter[2])[1], label=f"Total PDoS {current_label}", alpha=matter[3], linestyle=matter[4], zorder = 2)
            efermi = matter[1][0]
        if pdos_type in ["Integrated", "integrated"]:
            # for _, matter in enumerate(matters):
            plt.plot(matter[1][8], matter[1][7], c=color_sampling(matter[2])[2], label=f"Integrated PDoS {current_label}", alpha=matter[3], linestyle=matter[4], zorder = 3)
            efermi = matter[1][0]
        plt.plot(matter[1][8], matter[1][9],  c=color_sampling(matter[2])[3], alpha=matter[3], linestyle=matter[4], label=f"$s$ PDoS {current_label}",  zorder=4)
        plt.plot(matter[1][8], matter[1][12], c=color_sampling(matter[2])[4], alpha=matter[3], linestyle=matter[4], label=f"$p_x$ PDoS {current_label}",zorder=5)
        plt.plot(matter[1][8], matter[1][10], c=color_sampling(matter[2])[5], alpha=matter[3], linestyle=matter[4], label=f"$p_y$ PDoS {current_label}",zorder=6)
        plt.plot(matter[1][8], matter[1][11], c=color_sampling(matter[2])[6], alpha=matter[3], linestyle=matter[4], label=f"$p_z$ PDoS {current_label}",zorder=7)

    # Plot Fermi energy as a vertical line
    shift = efermi
    plt.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder = 1)
    fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
    plt.text(efermi-shift-x_range*0.02, y_top*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")

    # Title
    # plt.title(f"Electronic density of state {title} ({supplement})")
    plt.title(f"{pdos_type} PDoS {title} ")
    plt.ylabel(r"Density of States"); plt.xlabel(r"Energy (eV)")

    plt.ylim(0, y_top)
    plt.xlim(x_range*(-1), x_range)
    # plt.legend(loc="best")
    plt.legend(loc="upper right")

def plot_total_segment(title, matters_list):
    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Color calling
    fermi_color = color_sampling("Violet")

    # Ranges
    x_range = matters_list[0][1]
    y_top = matters_list[0][2]

    # Matter
    matter = matters_list
    num_elements = (len(matter[0]) - 3) // 2
    # Plot PDoS
    for index in range(num_elements):
        i_0 = 2 * index
        i_1 = 3 * index
        current_label = matter[0][3+i_0]
        plt.plot(matter[0][4+i_0][8], matter[0][4+i_0][6],  c=color_sampling(matter[1][0+i_1])[1], alpha=matter[1][1+i_1], linestyle=matter[1][2+i_1], label=f"Total PDoS {current_label}", zorder=2)
        plt.plot(matter[0][4+i_0][8], matter[0][4+i_0][9],  c=color_sampling(matter[1][0+i_1])[3], alpha=matter[1][1+i_1], linestyle=matter[1][2+i_1], label=f"$s$ PDoS {current_label}", zorder=3)
        plt.plot(matter[0][4+i_0][8], matter[0][4+i_0][12], c=color_sampling(matter[1][0+i_1])[4], alpha=matter[1][1+i_1], linestyle=matter[1][2+i_1], label=f"$p_x$ PDoS {current_label}", zorder=4)
        plt.plot(matter[0][4+i_0][8], matter[0][4+i_0][10], c=color_sampling(matter[1][0+i_1])[5], alpha=matter[1][1+i_1], linestyle=matter[1][2+i_1], label=f"$p_y$ PDoS {current_label}", zorder=5)
        plt.plot(matter[0][4+i_0][8], matter[0][4+i_0][11], c=color_sampling(matter[1][0+i_1])[6], alpha=matter[1][1+i_1], linestyle=matter[1][2+i_1], label=f"$p_z$ PDoS {current_label}", zorder=6)
    efermi = matter[0][4][0]

    # Plot Fermi energy as a vertical line
    shift = efermi
    plt.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder = 1)
    fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
    plt.text(efermi-shift-x_range*0.02, y_top*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")

    # Title
    # plt.title(f"Electronic density of state {title} ({supplement})")
    plt.title(f"Total PDoS {title} ")
    plt.ylabel(r"Density of States"); plt.xlabel(r"Energy (eV)")

    plt.ylim(0, y_top)
    plt.xlim(x_range*(-1), x_range)
    # plt.legend(loc="best")
    plt.legend(loc="upper right")
    plt.tight_layout()

def plot_sol_segment_pdos_col(title, matters_list):
    # Figure settings
    fig_setting = canvas_setting(8, 11)    # 2 * 5 + 1
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b"]

    # Materials information
    num_elements = len(matters_list[-1])//3
    matter = matters_list
    efermi = matter[0][4][0]

    # Ranges
    x_range = []
    y_top   = []
    for subplot_index in range(3):
        x_range.append(matter[subplot_index][1])
        y_top.append(matter[subplot_index][2])

    # Data process
    titles = []
    labels = [[], []]
    pdoses = [[], []]
    for subplot_index in range(2):
        titles.append(matter[subplot_index][0])
        for matter_index in range(num_elements):
            labels[matter_index].append(matter[subplot_index][3+2*matter_index])
            pdoses[matter_index].append(matter[subplot_index][4+2*matter_index])

    # Style parameters
    color = []
    alpha = []
    lines = []
    for matter_index in range(num_elements):
        color.append(matter[-1][0+3*matter_index])
        alpha.append(matter[-1][1+3*matter_index])
        lines.append(matter[-1][2+3*matter_index])

    # fig.suptitle(f"PDoS {title}", fontsize=fig_setting[3][0], y=0.96)
    fig.suptitle(f"PDoS {title}", fontsize=fig_setting[3][0], y=1.00)

    for subplot_index in range(2):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # Data ploting
        ax.set_title(f"{titles[subplot_index]}", fontsize=fig_setting[3][1])
        for matter_index in range(num_elements):
            current_label = labels[matter_index][subplot_index]
            current_pdos  = pdoses[matter_index][subplot_index]
            ax.plot(current_pdos[8], current_pdos[6],c=color_sampling(color[matter_index])[1],alpha=alpha[matter_index],ls=lines[matter_index],label=f"Total PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[9],c=color_sampling(color[matter_index])[3],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$s$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[12],c=color_sampling(color[matter_index])[4],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_x$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[10],c=color_sampling(color[matter_index])[5],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_y$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[11],c=color_sampling(color[matter_index])[6],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_z$ PDoS {current_label}",zorder=2)
        ax.set_xlim(-x_range[subplot_index],x_range[subplot_index])
        ax.set_ylim(0, y_top[subplot_index])
        ax.set_ylabel(r"Density of states")
        if subplot_index == 1:
            ax.set_xlabel(r"Energy (eV)")
        shift = efermi
        fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
        ax.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder=1)

        # Fermi energy
        ax.text(efermi-shift-x_range[subplot_index]*0.02, y_top[subplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")
        ax.legend(loc="upper right")

        # Subplots label
        orderlab_shift = 0.05
        # x_loc = 1-orderlab_shift*0.75
        x_loc = 0+orderlab_shift*0.75
        y_loc = 1-orderlab_shift

        ax.annotate(f"({order_labels[subplot_index]})",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="center", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})
    plt.tight_layout()
    # print(fig.get_size_inches())
    # print(fig.dpi)

def plot_sol_segment_pdos_row(title, matters_list):
    # Figure settings
    fig_setting = canvas_setting(16, 6)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b"]

    # Materials information
    num_elements = len(matters_list[-1])//3
    matter = matters_list
    efermi = matter[0][4][0]

    # Ranges
    x_range = []
    y_top   = []
    for subplot_index in range(3):
        x_range.append(matter[subplot_index][1])
        y_top.append(matter[subplot_index][2])

    # Data process
    titles = []
    labels = [[], []]
    pdoses = [[], []]
    for subplot_index in range(2):
        titles.append(matter[subplot_index][0])
        for matter_index in range(num_elements):
            labels[matter_index].append(matter[subplot_index][3+2*matter_index])
            pdoses[matter_index].append(matter[subplot_index][4+2*matter_index])

    # Style parameters
    color = []
    alpha = []
    lines = []
    for matter_index in range(num_elements):
        color.append(matter[-1][0+3*matter_index])
        alpha.append(matter[-1][1+3*matter_index])
        lines.append(matter[-1][2+3*matter_index])

    # fig.suptitle(f"PDoS {title}", fontsize=fig_setting[3][0], y=0.96)
    fig.suptitle(f"PDoS {title}", fontsize=fig_setting[3][0], y=1.00)

    for subplot_index in range(2):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # Data ploting
        ax.set_title(f"{titles[subplot_index]}", fontsize=fig_setting[3][1])
        for matter_index in range(num_elements):
            current_label = labels[matter_index][subplot_index]
            current_pdos  = pdoses[matter_index][subplot_index]
            ax.plot(current_pdos[8], current_pdos[6],c=color_sampling(color[matter_index])[1],alpha=alpha[matter_index],ls=lines[matter_index],label=f"Total PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[9],c=color_sampling(color[matter_index])[3],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$s$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[12],c=color_sampling(color[matter_index])[4],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_x$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[10],c=color_sampling(color[matter_index])[5],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_y$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[11],c=color_sampling(color[matter_index])[6],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_z$ PDoS {current_label}",zorder=2)
        ax.set_xlim(-x_range[subplot_index],x_range[subplot_index])
        ax.set_ylim(0, y_top[subplot_index])

        ax.set_xlabel(r"Energy (eV)")
        if subplot_index == 0:
            ax.set_ylabel(r"Density of states")

        shift = efermi
        fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
        ax.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder=1)

        # Fermi energy
        ax.text(efermi-shift-x_range[subplot_index]*0.02, y_top[subplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")
        ax.legend(loc="upper right")

        # Subplots label
        orderlab_shift = 0.05
        x_loc = 0+orderlab_shift*0.75
        # y_loc = 0+orderlab_shift
        # x_loc = 1-orderlab_shift*0.75
        y_loc = 1-orderlab_shift

        ax.annotate(f"({order_labels[subplot_index]})",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="center", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})
    plt.tight_layout()
    # print(fig.get_size_inches())
    # print(fig.dpi)

def plot_duo_segment_pdos_col(title, matters_list):
    # Figure settings
    fig_setting = canvas_setting(8, 16)    # 3 * 5 + 1
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(3, 1, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1], axs[2]]

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b","c"]

    # Materials information
    num_elements = len(matters_list[-1])//3
    matter = matters_list
    efermi = matter[0][4][0]

    # Ranges
    x_range = []
    y_top   = []
    for subplot_index in range(3):
        x_range.append(matter[subplot_index][1])
        y_top.append(matter[subplot_index][2])

    # Data process
    titles = []
    labels = [[], []]
    pdoses = [[], []]
    for subplot_index in range(3):
        titles.append(matter[subplot_index][0])
        for matter_index in range(num_elements):
            labels[matter_index].append(matter[subplot_index][3+2*matter_index])
            pdoses[matter_index].append(matter[subplot_index][4+2*matter_index])

    # Style parameters
    color = []
    alpha = []
    lines = []
    for matter_index in range(num_elements):
        color.append(matter[-1][0+3*matter_index])
        alpha.append(matter[-1][1+3*matter_index])
        lines.append(matter[-1][2+3*matter_index])

    # fig.suptitle(f"PDoS {title}", fontsize=fig_setting[3][0], y=0.96)
    fig.suptitle(f"PDoS {title}", fontsize=fig_setting[3][0], y=1.00)

    for subplot_index in range(3):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # Data ploting
        ax.set_title(f"{titles[subplot_index]}", fontsize=fig_setting[3][1])
        for matter_index in range(num_elements):
            current_label = labels[matter_index][subplot_index]
            current_pdos  = pdoses[matter_index][subplot_index]
            ax.plot(current_pdos[8], current_pdos[6],c=color_sampling(color[matter_index])[1],alpha=alpha[matter_index],ls=lines[matter_index],label=f"Total PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[9],c=color_sampling(color[matter_index])[3],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$s$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[12],c=color_sampling(color[matter_index])[4],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_x$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[10],c=color_sampling(color[matter_index])[5],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_y$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[11],c=color_sampling(color[matter_index])[6],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_z$ PDoS {current_label}",zorder=2)
        ax.set_xlim(-x_range[subplot_index],x_range[subplot_index])
        ax.set_ylim(0, y_top[subplot_index])
        ax.set_ylabel(r"Density of states")
        if subplot_index == 2:
            ax.set_xlabel(r"Energy (eV)")
        shift = efermi
        fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
        ax.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder=1)

        # Fermi energy
        ax.text(efermi-shift-x_range[subplot_index]*0.02, y_top[subplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")
        ax.legend(loc="upper right")

        # Subplots label
        orderlab_shift = 0.05
        x_loc = 0+orderlab_shift*0.75
        # y_loc = 1-orderlab_shift
        # x_loc = 1-orderlab_shift*0.75
        y_loc = 1-orderlab_shift

        ax.annotate(f"({order_labels[subplot_index]})",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="center", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

def plot_duo_segment_pdos_row(title, matters_list):
    # Figure settings
    fig_setting = canvas_setting(24, 6)    # 3 * 5 + 1
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 3, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1], axs[2]]

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b","c"]

    # Materials information
    num_elements = len(matters_list[-1])//3
    matter = matters_list
    efermi = matter[0][4][0]

    # Ranges
    x_range = []
    y_top   = []
    for subplot_index in range(3):
        x_range.append(matter[subplot_index][1])
        y_top.append(matter[subplot_index][2])

    # Data process
    titles = []
    labels = [[], []]
    pdoses = [[], []]
    for subplot_index in range(3):
        titles.append(matter[subplot_index][0])
        for matter_index in range(num_elements):
            labels[matter_index].append(matter[subplot_index][3+2*matter_index])
            pdoses[matter_index].append(matter[subplot_index][4+2*matter_index])

    # Style parameters
    color = []
    alpha = []
    lines = []
    for matter_index in range(num_elements):
        color.append(matter[-1][0+3*matter_index])
        alpha.append(matter[-1][1+3*matter_index])
        lines.append(matter[-1][2+3*matter_index])

    # fig.suptitle(f"PDoS {title}", fontsize=fig_setting[3][0], y=0.96)
    fig.suptitle(f"PDoS {title}", fontsize=fig_setting[3][0], y=1.00)

    for subplot_index in range(3):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # Data ploting
        ax.set_title(f"{titles[subplot_index]}", fontsize=fig_setting[3][1])
        for matter_index in range(num_elements):
            current_label = labels[matter_index][subplot_index]
            current_pdos  = pdoses[matter_index][subplot_index]
            ax.plot(current_pdos[8], current_pdos[6],c=color_sampling(color[matter_index])[1],alpha=alpha[matter_index],ls=lines[matter_index],label=f"Total PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[9],c=color_sampling(color[matter_index])[3],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$s$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[12],c=color_sampling(color[matter_index])[4],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_x$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[10],c=color_sampling(color[matter_index])[5],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_y$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[11],c=color_sampling(color[matter_index])[6],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_z$ PDoS {current_label}",zorder=2)
        ax.set_xlim(-x_range[subplot_index],x_range[subplot_index])
        ax.set_ylim(0, y_top[subplot_index])
        ax.set_xlabel(r"Energy (eV)")
        if subplot_index == 0:
            ax.set_ylabel(r"Density of states")
        shift = efermi
        fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
        ax.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder=1)

        # Fermi energy
        ax.text(efermi-shift-x_range[subplot_index]*0.02, y_top[subplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")
        ax.legend(loc="upper right")

        # Subplots label
        orderlab_shift = 0.05
        # x_loc = 0+orderlab_shift*0.75
        # y_loc = 1-orderlab_shift
        x_loc = 0+orderlab_shift*0.75
        y_loc = 1-orderlab_shift

        ax.annotate(f"({order_labels[subplot_index]})",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="center", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    # tight figure
    plt.tight_layout()

def plot_tri_segment_pdos_col(title, matters_list):

    # Figure settings
    fig_setting = canvas_setting(8, 21)    # 4 * 5 + 1
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(4, 1, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1], axs[2], axs[3]]

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b","c","d"]

    # Materials information
    num_elements = len(matters_list[-1])//3
    matter = matters_list
    efermi = matter[0][4][0]

    # Ranges
    x_range = []
    y_top   = []
    for subplot_index in range(4):
        x_range.append(matter[subplot_index][1])
        y_top.append(matter[subplot_index][2])

    # Data process
    titles = []
    labels = [[], []]
    pdoses = [[], []]
    for subplot_index in range(4):
        titles.append(matter[subplot_index][0])
        for matter_index in range(num_elements):
            labels[matter_index].append(matter[subplot_index][3+2*matter_index])
            pdoses[matter_index].append(matter[subplot_index][4+2*matter_index])

    # Style parameters
    color = []
    alpha = []
    lines = []
    for matter_index in range(num_elements):
        color.append(matter[-1][0+3*matter_index])
        alpha.append(matter[-1][1+3*matter_index])
        lines.append(matter[-1][2+3*matter_index])

    # fig.suptitle(f"PDoS {title}", fontsize=fig_setting[3][0], y=0.96)
    fig.suptitle(f"PDoS {title}", fontsize=fig_setting[3][0], y=1.0)

    for subplot_index in range(4):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # Data ploting
        ax.set_title(f"{titles[subplot_index]}", fontsize=fig_setting[3][1])
        for matter_index in range(num_elements):
            current_label = labels[matter_index][subplot_index]
            current_pdos  = pdoses[matter_index][subplot_index]
            ax.plot(current_pdos[8], current_pdos[6],c=color_sampling(color[matter_index])[1],alpha=alpha[matter_index],ls=lines[matter_index],label=f"Total PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[9],c=color_sampling(color[matter_index])[3],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$s$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[12],c=color_sampling(color[matter_index])[4],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_x$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[10],c=color_sampling(color[matter_index])[5],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_y$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[11],c=color_sampling(color[matter_index])[6],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_z$ PDoS {current_label}",zorder=2)
        ax.set_xlim(-x_range[subplot_index],x_range[subplot_index])
        ax.set_ylim(0, y_top[subplot_index])
        ax.set_ylabel(r"Density of states")
        if subplot_index == 3:
            ax.set_xlabel(r"Energy (eV)")
        shift = efermi
        fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
        ax.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder=1)

        # Fermi energy
        ax.text(efermi-shift-x_range[subplot_index]*0.02, y_top[subplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")
        ax.legend(loc="upper right")

        # Subplots label
        orderlab_shift = 0.05
        x_loc = 0+orderlab_shift*0.75
        # y_loc = 1-orderlab_shift
        # x_loc = 1-orderlab_shift*0.75
        y_loc = 1-orderlab_shift

        ax.annotate(f"({order_labels[subplot_index]})",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="center", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})
    plt.tight_layout()

def plot_tri_segment_pdos_block(title, matters_list):
    # Figure settings
    fig_setting = canvas_setting(16, 12)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b","c","d"]

    # Materials information
    num_elements = len(matters_list[-1])//3
    matter = matters_list
    efermi = matter[0][4][0]

    # Ranges
    x_range = []
    y_top   = []
    for subplot_index in range(4):
        x_range.append(matter[subplot_index][1])
        y_top.append(matter[subplot_index][2])

    # Data process
    titles = []
    labels = [[], []]
    pdoses = [[], []]
    for subplot_index in range(4):
        titles.append(matter[subplot_index][0])
        for matter_index in range(num_elements):
            labels[matter_index].append(matter[subplot_index][3+2*matter_index])
            pdoses[matter_index].append(matter[subplot_index][4+2*matter_index])

    # Style parameters
    color = []
    alpha = []
    lines = []
    for matter_index in range(num_elements):
        color.append(matter[-1][0+3*matter_index])
        alpha.append(matter[-1][1+3*matter_index])
        lines.append(matter[-1][2+3*matter_index])

    # fig.suptitle(f"PDoS {title}", fontsize=fig_setting[3][0], y=0.96)
    fig.suptitle(f"PDoS {title}", fontsize=fig_setting[3][0])

    for subplot_index in range(4):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # testing area
        # if subplot_index == 1:
        #     axes_element[subplot_index].axis("off")
        #     continue

        # Data ploting
        ax.set_title(f"{titles[subplot_index]}", fontsize=fig_setting[3][1])
        for matter_index in range(num_elements):
            current_label = labels[matter_index][subplot_index]
            current_pdos  = pdoses[matter_index][subplot_index]
            ax.plot(current_pdos[8], current_pdos[6],c=color_sampling(color[matter_index])[1],alpha=alpha[matter_index],ls=lines[matter_index],label=f"Total PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[9],c=color_sampling(color[matter_index])[3],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$s$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[12],c=color_sampling(color[matter_index])[4],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_x$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[10],c=color_sampling(color[matter_index])[5],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_y$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[11],c=color_sampling(color[matter_index])[6],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_z$ PDoS {current_label}",zorder=2)
        ax.set_xlim(-x_range[subplot_index],x_range[subplot_index])
        ax.set_ylim(0, y_top[subplot_index])
        if subplot_index in [0,2]:
            ax.set_ylabel(r"Density of states")
        if subplot_index in [2,3]:
            ax.set_xlabel(r"Energy (eV)")
        shift = efermi
        fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
        ax.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder=1)
        # if subplot_index == 2:
        #     ax.text(efermi-shift+x_range[subplot_index]*0.02, y_top[subplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="left")
        #     ax.legend(loc="upper left")
        # else:
        #     ax.text(efermi-shift-x_range[subplot_index]*0.02, y_top[subplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")
        #     ax.legend(loc="upper right")
        ax.text(efermi-shift-x_range[subplot_index]*0.02, y_top[subplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")
        ax.legend(loc="upper right")

        orderlab_shift = 0.05
        # if subplot_index == 0:
        #     x_loc = 1-orderlab_shift*0.75
        #     y_loc = 0+orderlab_shift
        # elif subplot_index == 1:
        #     x_loc = 0+orderlab_shift*0.75
        #     y_loc = 0+orderlab_shift
        # elif subplot_index == 2:
        #     x_loc = 1-orderlab_shift*0.75
        #     y_loc = 1-orderlab_shift
        # elif subplot_index == 3:
        #     x_loc = 0+orderlab_shift*0.75
        #     y_loc = 1-orderlab_shift
        # x_loc = 0+orderlab_shift*0.75
        # y_loc = 0+orderlab_shift
        # x_loc = 1-orderlab_shift*0.75
        x_loc = 0+orderlab_shift*0.75
        y_loc = 1-orderlab_shift

        ax.annotate(f"({order_labels[subplot_index]})",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="center", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

plot_seg_helo_info = "help information"
plot_seg_usage = plot_seg_helo_info
def plot_segment_pdos(*args):
    if len(args) == 1:
        print(plot_seg_helo_info)
    if len(args) == 2:
        if len(args[1]) == 1:
            print("Format error")
            print(plot_seg_usage)
        if len(args[1]) == 2:
            return plot_total_segment(args[0], args[1])
        if len(args[1]) == 3:
            return plot_sol_segment_pdos_col(args[0], args[1])
        if len(args[1]) == 4:
            return plot_duo_segment_pdos_col(args[0], args[1])
        if len(args[1]) == 5:
            return plot_tri_segment_pdos_col(args[0], args[1])

def plot_segment_pdos_fit(*args):
    if len(args) == 1:
        print(plot_seg_helo_info)
    if len(args) == 2:
        if len(args[1]) == 1:
            print("Format error")
            print(plot_seg_usage)
        if len(args[1]) == 2:
            return plot_total_segment(args[0], args[1])
        if len(args[1]) == 3:
            return plot_sol_segment_pdos_row(args[0], args[1])
        if len(args[1]) == 4:
            return plot_duo_segment_pdos_row(args[0], args[1])
        if len(args[1]) == 5:
            return plot_tri_segment_pdos_block(args[0], args[1])
