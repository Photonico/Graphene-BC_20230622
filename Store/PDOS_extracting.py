#### Declarations of process functions for PDOS with vectorized programming
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

### Necessary packages invoking
import xml.etree.ElementTree as ET
import numpy as np

### File analysis
def analyze_dpos(file_path):
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

### Total PDOS: univseral elements and layers
def pdos_total_extracting(file_path):
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

    path_dos_spin_1  = ".//total/array/set/set[@comment='spin 1']/r"
    path_dos_spin_2  = ".//total/array/set/set[@comment='spin 2']/r"
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

### Elements analysis for PDoS calculation
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

### PDOS for elements
def pdos_element_extracting(file_path, element):
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

### PDOS for customized range
def pdos_segment(file_path, start, end):
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
