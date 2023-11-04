#### Declarations of process functions for PDOS with vectorized programming
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# Necessary packages invoking
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from Store.output import canvas_setting, color_sampling

# File analysis
def analyze_dpos(file_path):
    # Help information
    if file_path == "help":
        print("Please use this function on the file: vasprun.xml.")
        return []
    # Parse the vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()
    # Flags to keep track of <ion 1> and <spin 1>
    ion_1_found = False
    spin_1_count = 0
    # Loop through <set> elements
    for set_element in root.findall(".//set"):
        comment = set_element.attrib.get("comment", "")
        # Track when <ion 1> is found
        if "ion 1" in comment:
            ion_1_found = True
        # Count <spin 1> only after <ion 1> is found
        if ion_1_found and "spin 1" in comment:
            spin_1_count += 1
            # Skip the first <spin 1> after <ion 1> is found
            if spin_1_count == 1:
                continue
            # Loop through the <r> elements and print the number of values and the values themselves
            for i, r_element in enumerate(set_element.findall("r")):
                values = list(map(float, r_element.text.split()))
                print(f"Number of values in row {i + 1}: {len(values)}")
                print(f"Values: {values}")
            # Break the loop after analyzing the second occurrence of <spin 1>
            break

# Elements analysis for PDoS calculation
def get_elements(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize an empty dictionary to store the element-ion pairs
    element_ions = {}

    # Use XPath to locate the <rc><c> tags under the path "atominfo/array[@name="atoms"]/set"
    for i, atom in enumerate(root.findall(".//atominfo//array[@name='atoms']//set//rc"), start=1):
        element = atom.find("c").text.strip()
        if element in element_ions:
            # Update the maximum index for the element
            element_ions[element][1] = i
        else:
            # Add a new entry for the element, with the minimum and maximum index being the same
            element_ions[element] = [i, i]

    # Convert the lists to tuples
    for element in element_ions:
        element_ions[element] = tuple(element_ions[element])

    return element_ions

# Total PDoS: univseral elements and layers
def extract_pdos(file_path):
    ## Analysis vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()

    ## Extract Fermi energy
    efermi_element = root.find(".//dos/i[@name='efermi']")
    efermi = float(efermi_element.text.strip())

    ## Extract the number of ions
    first_positions = root.find(".//varray[@name='positions'][1]")
    positions_concatenated_text = " ".join([position.text for position in first_positions.findall("v")])
    positions_array = np.fromstring(positions_concatenated_text, sep=" ")
    positions_matrix = positions_array.reshape(-1, 3)
    ions_number = positions_matrix.shape[0]

    ## Extract the number of kpoints
    kpointlist = root.find(".//varray[@name='kpointlist']")
    kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
    kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
    kpointlist_matrix = kpointlist_array.reshape(-1, 3)
    kpoints_number = kpointlist_matrix.shape[0]

    ## Extract eigen, occupancy number
    for kpoints_index in range(1, kpoints_number + 1):
        xpath_expr = f".//set[@comment='kpoint {kpoints_index}']"
        eigen_column = np.empty(0)
        occu_column  = np.empty(0)
        for eigen_occ_element in root.find(xpath_expr):
            eigen_values = list(map(float, eigen_occ_element.text.split()))
            eigen_column = np.append(eigen_column, eigen_values[0])
            occu_column = np.append(occu_column, eigen_values[1])
        if kpoints_index == 1 :
            eigen_matrix = eigen_column.reshape(-1, 1)
            occu_matrix = occu_column.reshape(-1, 1)
        else:
            eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
            occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))
    # eigen_sum = np.sum(eigen_matrix, axis=1)
    # occu_sum  = np.sum(occu_matrix, axis=1)

    ## Extract energy, total DOS, and integrated DOS
    # lists initialization
    energy_dos_list         = np.empty(0)
    total_dos_list          = np.empty(0)
    integrated_dos_list     = np.empty(0)

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

    ## Extract energy, s-PDOS, p_y-PDOS, p_z-PDOS, p_x-PDOS, d_xy-PDOS, d_yz-PDOS, d_z2-PDOS, d_xz-PDOS, x2-y2-PDOS
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
        d_xz_pdos_column    = np.empty(0)
        x2_y2_pdos_column   = np.empty(0)
        for pdos_element in root.findall(path_ions):
            pdos_values = list(map(float, pdos_element.text.split()))
            # Columns of energy
            energy_pdos_column = np.append(energy_pdos_column, pdos_values[0])
            # Columns of s-PDOS
            s_pdos_column = np.append(s_pdos_column, pdos_values[1])
            # Columns of p_y-PDOS
            p_y_pdos_column = np.append(p_y_pdos_column, pdos_values[2])
            # Columns of p_z-PDOS
            p_z_pdos_column = np.append(p_z_pdos_column, pdos_values[3])
            # Columns of p_x-PDOS
            p_x_pdos_column = np.append(p_x_pdos_column, pdos_values[4])
            # Columns of d_xy-PDOS
            d_xy_pdos_column = np.append(d_xy_pdos_column, pdos_values[5])
            # Columns of d_yz-PDOS
            d_yz_pdos_column = np.append(d_yz_pdos_column, pdos_values[6])
            # Columns of d_z2-PDOS
            d_z2_pdos_column = np.append(d_z2_pdos_column, pdos_values[7])
            # Columns of d_xz-PDOS
            d_xz_pdos_column = np.append(d_xz_pdos_column, pdos_values[8])
            # Columns of x2-y2-PDOS
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
            d_xz_pdos_matrix = d_xz_pdos_column.reshape(-1, 1)
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
            d_xz_pdos_matrix = np.hstack((d_xz_pdos_matrix, d_xz_pdos_column.reshape(-1, 1)))
            x2_y2_pdos_matrix = np.hstack((x2_y2_pdos_matrix, x2_y2_pdos_column.reshape(-1, 1)))
    energy_pdos_sum = energy_pdos_matrix[:,0]
    s_pdos_sum = np.sum(s_pdos_matrix, axis=1)
    p_y_pdos_sum = np.sum(p_y_pdos_matrix, axis=1)
    p_z_pdos_sum = np.sum(p_z_pdos_matrix, axis=1)
    p_x_pdos_sum = np.sum(p_x_pdos_matrix, axis=1)
    d_xy_pdos_sum = np.sum(d_xy_pdos_matrix, axis=1)
    d_yz_pdos_sum = np.sum(d_yz_pdos_matrix, axis=1)
    d_z2_pdos_sum = np.sum(d_z2_pdos_matrix, axis=1)
    d_xz_pdos_sum = np.sum(d_xz_pdos_matrix, axis=1)
    x2_y2_pdos_sum = np.sum(x2_y2_pdos_matrix, axis=1)
    energy_pdos_shift = energy_pdos_sum - shift
    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
            energy_dos_shift, total_dos_list, integrated_dos_list,                      # 5 ~ 7
            energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
            d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_xz_pdos_sum,                 # 13 ~ 16
            x2_y2_pdos_sum)

# PDOS for elements
def extract_element_pdos(file_path, element):
    ## Analysis vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()

    ## Extract Fermi energy
    efermi_element = root.find(".//dos/i[@name='efermi']")
    efermi = float(efermi_element.text.strip())

    ## Extract the number of ions
    first_positions = root.find(".//varray[@name='positions'][1]")
    positions_concatenated_text = " ".join([position.text for position in first_positions.findall("v")])
    positions_array = np.fromstring(positions_concatenated_text, sep=" ")
    positions_matrix = positions_array.reshape(-1, 3)
    ions_number = positions_matrix.shape[0]

    ## Analysis elements
    index_start = get_elements(file_path)[element][0]
    index_end = get_elements(file_path)[element][1]

    ## Extract the number of kpoints
    kpointlist = root.find(".//varray[@name='kpointlist']")
    kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
    kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
    kpointlist_matrix = kpointlist_array.reshape(-1, 3)
    kpoints_number = kpointlist_matrix.shape[0]

    ## Extract eigen, occupancy number
    for kpoints_index in range(1, kpoints_number + 1):
        xpath_expr = f".//set[@comment='kpoint {kpoints_index}']"
        eigen_column = np.empty(0)
        occu_column  = np.empty(0)
        for eigen_occ_element in root.find(xpath_expr):
            eigen_values = list(map(float, eigen_occ_element.text.split()))
            eigen_column = np.append(eigen_column, eigen_values[0])
            occu_column = np.append(occu_column, eigen_values[1])
        if kpoints_index == 1 :
            eigen_matrix = eigen_column.reshape(-1, 1)
            occu_matrix = occu_column.reshape(-1, 1)
        else:
            eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
            occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))
    # eigen_sum = np.sum(eigen_matrix, axis=1)
    # occu_sum  = np.sum(occu_matrix, axis=1)

    ## Extract energy, total DOS, and integrated DOS
    # lists initialization
    energy_dos_list     = np.empty(0)
    total_dos_list      = np.empty(0)
    integrated_dos_list = np.empty(0)
    path_dos = ".//total/array/set/set[@comment='spin 1']/r"
    for element_dos in root.findall(path_dos):
        dos_values = list(map(float, element_dos.text.split()))
        energy_var = dos_values[0]
        energy_dos_list = np.append(energy_dos_list, energy_var)
        # total_dos_var = dos_values[1]
        # total_dos_list = np.append(total_dos_list, total_dos_var)
        # integrated_dos_var = dos_values[2]
        # integrated_dos_list = np.append(integrated_dos_list, integrated_dos_var)
    shift = efermi
    energy_dos_shift = energy_dos_list - shift

    ## Extract energy, s-PDOS, p_y-PDOS, p_z-PDOS, p_x-PDOS, d_xy-PDOS, d_yz-PDOS, d_z2-PDOS, d_xz-PDOS, x2-y2-PDOS
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
        d_xz_pdos_column    = np.empty(0)
        x2_y2_pdos_column   = np.empty(0)
        for pdos_element in root.findall(path_ions):
            pdos_values = list(map(float, pdos_element.text.split()))
            # Columns of energy
            energy_pdos_column = np.append(energy_pdos_column, pdos_values[0])
            # Columns of s-PDOS
            s_pdos_column = np.append(s_pdos_column, pdos_values[1])
            # Columns of p_y-PDOS
            p_y_pdos_column = np.append(p_y_pdos_column, pdos_values[2])
            # Columns of p_z-PDOS
            p_z_pdos_column = np.append(p_z_pdos_column, pdos_values[3])
            # Columns of p_x-PDOS
            p_x_pdos_column = np.append(p_x_pdos_column, pdos_values[4])
            # Columns of d_xy-PDOS
            d_xy_pdos_column = np.append(d_xy_pdos_column, pdos_values[5])
            # Columns of d_yz-PDOS
            d_yz_pdos_column = np.append(d_yz_pdos_column, pdos_values[6])
            # Columns of d_z2-PDOS
            d_z2_pdos_column = np.append(d_z2_pdos_column, pdos_values[7])
            # Columns of d_xz-PDOS
            d_xz_pdos_column = np.append(d_xz_pdos_column, pdos_values[8])
            # Columns of x2-y2-PDOS
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
            d_xz_pdos_matrix = d_xz_pdos_column.reshape(-1, 1)
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
            d_xz_pdos_matrix = np.hstack((d_xz_pdos_matrix, d_xz_pdos_column.reshape(-1, 1)))
            x2_y2_pdos_matrix = np.hstack((x2_y2_pdos_matrix, x2_y2_pdos_column.reshape(-1, 1)))
    energy_pdos_sum = energy_pdos_matrix[:,0]
    s_pdos_sum = np.sum(s_pdos_matrix, axis=1)
    p_y_pdos_sum = np.sum(p_y_pdos_matrix, axis=1)
    p_z_pdos_sum = np.sum(p_z_pdos_matrix, axis=1)
    p_x_pdos_sum = np.sum(p_x_pdos_matrix, axis=1)
    d_xy_pdos_sum = np.sum(d_xy_pdos_matrix, axis=1)
    d_yz_pdos_sum = np.sum(d_yz_pdos_matrix, axis=1)
    d_z2_pdos_sum = np.sum(d_z2_pdos_matrix, axis=1)
    d_xz_pdos_sum = np.sum(d_xz_pdos_matrix, axis=1)
    x2_y2_pdos_sum = np.sum(x2_y2_pdos_matrix, axis=1)
    total_dos_list = s_pdos_sum + p_y_pdos_sum + p_z_pdos_sum + p_x_pdos_sum + d_xy_pdos_sum + d_yz_pdos_sum + d_z2_pdos_sum + d_xz_pdos_sum + x2_y2_pdos_sum
    integrated_dos_list = np.trapz(total_dos_list, x=energy_dos_shift)

    energy_pdos_shift = energy_pdos_sum - shift
    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
            energy_dos_shift, total_dos_list, integrated_dos_list,                      # 5 ~ 7
            energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
            d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_xz_pdos_sum,                 # 13 ~ 16
            x2_y2_pdos_sum)

# PDOS for customized range
def extract_segment_pdos(file_path, start, end):
    ## Analysis vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()

    ## Analysis elements
    index_start = start
    index_end = end

    ## Extract Fermi energy
    efermi_element = root.find(".//dos/i[@name='efermi']")
    efermi = float(efermi_element.text.strip())

    ## Extract the number of ions
    first_positions = root.find(".//varray[@name='positions'][1]")
    positions_concatenated_text = " ".join([position.text for position in first_positions.findall("v")])
    positions_array = np.fromstring(positions_concatenated_text, sep=" ")
    positions_matrix = positions_array.reshape(-1, 3)
    ions_number = positions_matrix.shape[0]

    ## Extract the number of kpoints
    kpointlist = root.find(".//varray[@name='kpointlist']")
    kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
    kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
    kpointlist_matrix = kpointlist_array.reshape(-1, 3)
    kpoints_number = kpointlist_matrix.shape[0]

    ## Extract eigen, occupancy number
    for kpoints_index in range(1, kpoints_number + 1):
        xpath_expr = f".//set[@comment='kpoint {kpoints_index}']"
        eigen_column = np.empty(0)
        occu_column  = np.empty(0)
        for eigen_occ_element in root.find(xpath_expr):
            eigen_values = list(map(float, eigen_occ_element.text.split()))
            eigen_column = np.append(eigen_column, eigen_values[0])
            occu_column = np.append(occu_column, eigen_values[1])
        if kpoints_index == 1 :
            eigen_matrix = eigen_column.reshape(-1, 1)
            occu_matrix = occu_column.reshape(-1, 1)
        else:
            eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
            occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))
    # eigen_sum = np.sum(eigen_matrix, axis=1)
    # occu_sum  = np.sum(occu_matrix, axis=1)

    ## Extract energy, total DOS, and integrated DOS
    # lists initialization
    energy_dos_list     = np.empty(0)
    total_dos_list      = np.empty(0)
    integrated_dos_list = np.empty(0)
    path_dos = ".//total/array/set/set[@comment='spin 1']/r"
    for element_dos in root.findall(path_dos):
        dos_values = list(map(float, element_dos.text.split()))
        energy_var = dos_values[0]
        energy_dos_list = np.append(energy_dos_list, energy_var)
        # total_dos_var = dos_values[1]
        # total_dos_list = np.append(total_dos_list, total_dos_var)
        # integrated_dos_var = dos_values[2]
        # integrated_dos_list = np.append(integrated_dos_list, integrated_dos_var)
    shift = efermi
    energy_dos_shift = energy_dos_list - shift

    ## Extract energy, s-PDOS, p_y-PDOS, p_z-PDOS, p_x-PDOS, d_xy-PDOS, d_yz-PDOS, d_z2-PDOS, d_xz-PDOS, x2-y2-PDOS
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
        d_xz_pdos_column    = np.empty(0)
        x2_y2_pdos_column   = np.empty(0)
        for pdos_element in root.findall(path_ions):
            pdos_values = list(map(float, pdos_element.text.split()))
            # Columns of energy
            energy_pdos_column = np.append(energy_pdos_column, pdos_values[0])
            # Columns of s-PDOS
            s_pdos_column = np.append(s_pdos_column, pdos_values[1])
            # Columns of p_y-PDOS
            p_y_pdos_column = np.append(p_y_pdos_column, pdos_values[2])
            # Columns of p_z-PDOS
            p_z_pdos_column = np.append(p_z_pdos_column, pdos_values[3])
            # Columns of p_x-PDOS
            p_x_pdos_column = np.append(p_x_pdos_column, pdos_values[4])
            # Columns of d_xy-PDOS
            d_xy_pdos_column = np.append(d_xy_pdos_column, pdos_values[5])
            # Columns of d_yz-PDOS
            d_yz_pdos_column = np.append(d_yz_pdos_column, pdos_values[6])
            # Columns of d_z2-PDOS
            d_z2_pdos_column = np.append(d_z2_pdos_column, pdos_values[7])
            # Columns of d_xz-PDOS
            d_xz_pdos_column = np.append(d_xz_pdos_column, pdos_values[8])
            # Columns of x2-y2-PDOS
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
            d_xz_pdos_matrix = d_xz_pdos_column.reshape(-1, 1)
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
            d_xz_pdos_matrix = np.hstack((d_xz_pdos_matrix, d_xz_pdos_column.reshape(-1, 1)))
            x2_y2_pdos_matrix = np.hstack((x2_y2_pdos_matrix, x2_y2_pdos_column.reshape(-1, 1)))
    energy_pdos_sum = energy_pdos_matrix[:,0]
    s_pdos_sum = np.sum(s_pdos_matrix, axis=1)
    p_y_pdos_sum = np.sum(p_y_pdos_matrix, axis=1)
    p_z_pdos_sum = np.sum(p_z_pdos_matrix, axis=1)
    p_x_pdos_sum = np.sum(p_x_pdos_matrix, axis=1)
    d_xy_pdos_sum = np.sum(d_xy_pdos_matrix, axis=1)
    d_yz_pdos_sum = np.sum(d_yz_pdos_matrix, axis=1)
    d_z2_pdos_sum = np.sum(d_z2_pdos_matrix, axis=1)
    d_xz_pdos_sum = np.sum(d_xz_pdos_matrix, axis=1)
    x2_y2_pdos_sum = np.sum(x2_y2_pdos_matrix, axis=1)
    energy_pdos_shift = energy_pdos_sum - shift
    total_dos_list = s_pdos_sum + p_y_pdos_sum + p_z_pdos_sum + p_x_pdos_sum + d_xy_pdos_sum + d_yz_pdos_sum + d_z2_pdos_sum + d_xz_pdos_sum + x2_y2_pdos_sum
    integrated_dos_list = np.trapz(total_dos_list, x=energy_dos_shift)

    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
            energy_dos_shift, total_dos_list, integrated_dos_list,                      # 5 ~ 7
            energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
            d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_xz_pdos_sum,                 # 13 ~ 16
            x2_y2_pdos_sum)

# Total PDoS Plotting
def plot_total_pdos(matter, x_range = None, y_top = None, supplement = None, pdos_type = None, pdos_data = None, color_family="blue"):
    # Help information
    help_info = "Usage: plot_pdos" + \
                "Use extract_pdos to extract the DoS data."

    if matter in ["help", "Help"]:
        print(help_info)

    # Figure setting
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Colors calling
    fermi_color = color_sampling("Orange")
    colors = color_sampling(color_family)

    # Data plotting range
    # y_axis_top = max(dos_data[6]); y_limit = y_axis_top * 0.6
    # y_axis_top = max(max(total_dos_list), max(integrated_dos_list))
    y_limit = y_top

    # Data plotting
    if pdos_type in ["All", "all"]:
        plt.plot(pdos_data[5], pdos_data[6], c=colors[1], label="Total PDoS", zorder=3)
        plt.plot(pdos_data[5], pdos_data[7], c=colors[2], label="Integrated DoS", zorder=2)
    if pdos_type in ["Total", "total"]:
        plt.plot(pdos_data[8], pdos_data[6], c=colors[1], label="Total DoS", zorder=2)
    if pdos_type in ["Integrated", "integrated"]:
        plt.plot(pdos_data[8], pdos_data[7], c=colors[2], label="Integrated DoS", zorder=2)

    plt.plot(pdos_data[8], pdos_data[9],  c=colors[3], label=r"$s$ PDoS",  zorder=4)
    plt.plot(pdos_data[8], pdos_data[12], c=colors[4], label=r"$p_x$ PDoS",zorder=5)
    plt.plot(pdos_data[8], pdos_data[10], c=colors[5], label=r"$p_y$ PDoS",zorder=5)
    plt.plot(pdos_data[8], pdos_data[11], c=colors[6], label=r"$p_z$ PDoS",zorder=5)

    # Plot Fermi energy as a vertical line
    efermi_pdos = pdos_data[0]
    shift = efermi_pdos
    plt.axvline(x = efermi_pdos-shift, linestyle="--", c=fermi_color[1], alpha=0.95, label=r"Fermi energy", zorder=1)
    fermi_energy_text = f"Fermi energy\n{efermi_pdos:.3f} (eV)"
    plt.text(efermi_pdos-shift-x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")

    # Title and labels
    # plt.title(f"Projected electronic density of state for {matter} ({supplement})")
    plt.title(f"PDoS for {matter} ({supplement})")
    plt.ylabel(r"Density of States", fontsize = 1.0* 12)
    plt.xlabel(r"Energy (eV)", fontsize = 1.0* 12)

    plt.ylim(0, y_top)
    plt.xlim(x_range*(-1), x_range)
    plt.legend(loc="upper right")
    # plt.show()

###  PDoS Plotting for each element
def pdos_sol_element(matter, x_range, y_top, supplement, pdos_total, element, pdos_element, color_family="blue"):

    # Figure setting
    fig_setting = canvas_setting(12, 6)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # Colors calling
    fermi_color = color_sampling("Orange")
    annotate_color = color_sampling("Grey")
    colors = color_sampling(color_family)

    # Data process
    elements = [matter, element]
    pdos_results = [pdos_total, pdos_element]
    label_positions = {0: (1, 0), 1: (0, 0)}

    fig.suptitle(f"Projected electronic density of state for {matter} ({supplement})", fontsize =1.0*18)

    for i in range(2):
        ax = axes_element[i]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        pdos_data = pdos_results[i]
        element = elements[i]

        efermi_pdos = pdos_data[0]

        if i == 0:
            ax.set_title("Total PDoS", fontsize = 1.0 * 16)
            line_width = 1.0
        else:
            ax.set_title(f"PDoS for {element}", fontsize = 1.0 * 16)
            line_width = 0.75
        ax.set_ylabel(r"Density of states", fontsize =1.0* 12)
        ax.set_xlabel(r"Energy (eV)", fontsize =1.0* 12)

        ax.plot(pdos_data[5], pdos_data[6],  c=colors[1], lw=line_width, label=r"Total DOS",  zorder=2)
        ax.plot(pdos_data[8], pdos_data[9],  c=colors[3], lw=line_width, label=r"$s$ PDoS",   zorder=3)
        ax.plot(pdos_data[8], pdos_data[12], c=colors[4], lw=line_width, label=r"$p_x$ PDoS", zorder=4)
        ax.plot(pdos_data[8], pdos_data[10], c=colors[5], lw=line_width, label=r"$p_y$ PDoS", zorder=4)
        ax.plot(pdos_data[8], pdos_data[11], c=colors[6], lw=line_width, label=r"$p_z$ PDoS", zorder=4)

        shift = efermi_pdos

        if i == 0:
            y_limit = y_top
        elif i == 1:
            # y_limit = y_top * 0.5
            y_limit = y_top

        ax.axvline(x = efermi_pdos-shift, linestyle="--", color=fermi_color[1], alpha=0.95, label=r"Fermi energy", zorder=1)
        ax.set_ylim(0, y_limit)
        ax.set_xlim(-x_range, x_range)

        fermi_energy_text = f"Fermi energy\n{efermi_pdos:.3f} (eV)"
        ax.text(efermi_pdos-shift-x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color=fermi_color[0], rotation=0, va = "top", ha="right")

        x_label, y_label = label_positions[i]
        if i == 0:
            ax.legend(loc="upper right")
        elif i == 1:
            ax.legend(loc="upper right")
        relative_offset = 0.025  # adjust this value as needed
        if x_label == 0:
            ha = "left"; x_label_offset = relative_offset
        else:
            ha = "right"; x_label_offset = -relative_offset
        if y_label == 0:
            va = "bottom"; y_label_offset = relative_offset
        else:
            va = "top"; y_label_offset = -relative_offset
        ax.annotate(f"({chr(97 + i)})",
                    xy = (x_label + x_label_offset, y_label + y_label_offset),
                    xycoords = "axes fraction",
                    fontsize = 1.0 * 18,
                    ha = ha, va = va,
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"}, zorder=5)

    plt.tight_layout()

def pdos_duo_element(matter, x_range, y_top, supplement, pdos_total, element_1, pdos_1, element_2, pdos_2, color_family="blue"):

    # Figure setting
    fig_setting = canvas_setting(12, 10)
    params = fig_setting[2]; plt.rcParams.update(params)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])
    fig = plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    # Colors calling
    fermi_color = color_sampling("Orange")
    annotate_color = color_sampling("Grey")
    colors = color_sampling(color_family)

    # Subfigures setting
    axes_element = [ax0, ax1, ax2]
    elements = [matter, element_1, element_2]
    pdos_results = [pdos_total, pdos_1, pdos_2]
    label_positions = {0: (0, 0), 1: (1, 1), 2:(0, 1)}

    fig.suptitle(f"Projected electronic density of state for {matter} ({supplement})", fontsize =1.0*18)

    for i in range(3):
        ax = axes_element[i]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        pdos_data = pdos_results[i]
        element = elements[i]

        efermi_pdos = pdos_data[0]

        if i == 0:
            ax.set_title("Total PDoS", fontsize = 1.0 * 16)
            line_width = 1.00
        else:
            ax.set_title(f"PDoS for {element}", fontsize = 1.0 * 16)
            line_width = 0.75
        ax.set_ylabel(r"Density of states", fontsize = 1.0 * 12)
        ax.set_xlabel(r"Energy (eV)", fontsize = 1.0 * 12)

        ax.plot(pdos_data[5], pdos_data[6],  c=colors[1], lw=line_width, label=r"Total DOS",  zorder=2)
        ax.plot(pdos_data[8], pdos_data[9],  c=colors[3], lw=line_width, label=r"$s$ PDoS",   zorder=3)
        ax.plot(pdos_data[8], pdos_data[12], c=colors[4], lw=line_width, label=r"$p_x$ PDoS", zorder=4)
        ax.plot(pdos_data[8], pdos_data[10], c=colors[5], lw=line_width, label=r"$p_y$ PDoS", zorder=4)
        ax.plot(pdos_data[8], pdos_data[11], c=colors[6], lw=line_width, label=r"$p_z$ PDoS", zorder=4)

        shift = efermi_pdos

        if i == 0:
            y_limit = y_top
        elif i == 1:
            y_limit = y_top * 0.5
        elif i == 2:
            y_limit = y_top * 0.5

        ax.axvline(x = efermi_pdos-shift, linestyle="--", color=fermi_color[1], alpha=0.95, label=r"Fermi energy", zorder=1)
        ax.set_ylim(0, y_limit)
        ax.set_xlim(-x_range, x_range)
        fermi_energy_text = f"Fermi energy\n{efermi_pdos:.3f} (eV)"
        if i == 1:
            ax.text(efermi_pdos-shift+x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color=fermi_color[0], rotation=0, va = "top", ha="left")
        else:
            ax.text(efermi_pdos-shift-x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color=fermi_color[0], rotation=0, va = "top", ha="right")

        x_label, y_label = label_positions[i]
        if i == 0:
            x_relative_offset = 0.025 * 0.5
            y_relative_offset = 0.025
            ax.legend(loc="upper right")
        elif i==1:
            x_relative_offset = 0.025
            y_relative_offset = 0.025
            ax.legend(loc="upper left")
        elif i==2:
            x_relative_offset = 0.025
            y_relative_offset = 0.025
            ax.legend(loc="upper right")

        if x_label == 0:
            ha = "left"; x_label_offset = x_relative_offset
        else:
            ha = "right"; x_label_offset = -x_relative_offset
        if y_label == 0:
            va = "bottom"; y_label_offset = y_relative_offset
        else:
            va = "top"; y_label_offset = -y_relative_offset

        ax.annotate(f"({chr(97 + i)})",
                    xy = (x_label + x_label_offset, y_label + y_label_offset),
                    xycoords = "axes fraction",
                    fontsize = 1.0 * 18,
                    ha = ha, va = va,
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"}, zorder=5)

    plt.tight_layout()

def pdos_tri_element(matter, x_range, y_top, supplement, pdos_total, element_1, pdos_1, element_2, pdos_2, element_3, pdos_3, color_family="blue"):

    # Figure settings
    fig_setting = canvas_setting(12, 10)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

    # Colors calling
    fermi_color = color_sampling("Orange")
    annotate_color = color_sampling("Grey")
    colors = color_sampling(color_family)

    # Data process
    elements = [matter, element_1, element_2, element_3]
    pdos_results = [pdos_total, pdos_1, pdos_2, pdos_3]

    label_positions = {0: (1, 0), 1: (0, 0), 2:(1, 1), 3:(0, 1)}

    fig.suptitle(f"Projected electronic density of state for {matter} ({supplement})", fontsize =1.0*18)

    for i in range(4):
        ax = axes_element[i]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        pdos_data = pdos_results[i]
        element = elements[i]

        efermi_pdos = pdos_data[0]

        if i == 0:
            ax.set_title("Total PDoS", fontsize = 1.0 * 16)
            line_width = 1.00
        else:
            ax.set_title(f"PDoS for {element}", fontsize = 1.0 * 16)
            line_width = 0.75
        ax.set_ylabel(r"Density of states", fontsize = 1.0 * 12)
        ax.set_xlabel(r"Energy (eV)", fontsize = 1.0 * 12)

        ax.plot(pdos_data[5], pdos_data[6],  c=colors[1], lw=line_width, label=r"Total DOS",  zorder=2)
        ax.plot(pdos_data[8], pdos_data[9],  c=colors[3], lw=line_width, label=r"$s$ PDoS",   zorder=3)
        ax.plot(pdos_data[8], pdos_data[12], c=colors[4], lw=line_width, label=r"$p_x$ PDoS", zorder=4)
        ax.plot(pdos_data[8], pdos_data[10], c=colors[5], lw=line_width, label=r"$p_y$ PDoS", zorder=4)
        ax.plot(pdos_data[8], pdos_data[11], c=colors[6], lw=line_width, label=r"$p_z$ PDoS", zorder=4)

        shift = efermi_pdos

        if i == 0:
            y_limit = y_top
        elif i == 1:
            y_limit = round(y_top * 0.3)
        elif i == 2:
            y_limit = round(y_top * 0.3)
        elif i == 3:
            y_limit = round(y_top * 0.3)

        ax.axvline(x = efermi_pdos-shift, linestyle="--", color=fermi_color[1], alpha=0.95, label=r"Fermi energy", zorder=1)
        ax.set_ylim(0, y_limit)
        ax.set_xlim(-x_range, x_range)
        fermi_energy_text = f"Fermi energy\n{efermi_pdos:.3f} (eV)"
        if i == 2:
            ax.text(efermi_pdos-shift+x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color=fermi_color[0], rotation=0, va = "top", ha="left")
            ax.legend(loc="upper left")
        else:
            ax.text(efermi_pdos-shift-x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color=fermi_color[0], rotation=0, va = "top", ha="right")
            ax.legend(loc="upper right")

        x_label, y_label = label_positions[i]
        relative_offset = 0.025
        if x_label == 0:
            ha = "left"; x_label_offset = relative_offset
        else:
            ha = "right"; x_label_offset = -relative_offset
        if y_label == 0:
            va = "bottom"; y_label_offset = relative_offset
        else:
            va = "top"; y_label_offset = -relative_offset

        ax.annotate(f"({chr(97 + i)})",
                    xy = (x_label + x_label_offset, y_label + y_label_offset),
                    xycoords = "axes fraction",
                    fontsize = 1.0 * 18,
                    ha = ha, va = va,
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"}, zorder=5)

    plt.tight_layout()

# General usage for PDoS plotting
helo_info = "help information"
def plot_pdos_element(*args):
    if args[0] == "help":
        print("helo_info")
        return
    if len(args) in [5,6]:
        return plot_total_pdos(args[0], args[1], args[2], args[3], args[4])
    if len(args) in [7,8]:
        return pdos_sol_element(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
    if len(args) in [9,10]:
        return pdos_duo_element(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])
    if len(args) in [11,12]:
        return pdos_tri_element(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10])
