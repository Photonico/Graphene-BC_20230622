#### Declarations of process functions for DOS with vectorized programming
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

### Necessary packages invoking
import xml.etree.ElementTree as ET
import numpy as np

### Extract DOS from vasprun.xml
def extract_dos(file_path):
    # Analysis vasprun.xml file
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
    for kpoints_index in range(1, kpoints_number+1):
        xpath_expr = f".//set[@comment='kpoint {kpoints_index}']"
        eigen_column = np.empty(0)
        occu_column  = np.empty(0)
        for eigen_occ_element in root.find(xpath_expr):
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
    # eigen_sum = np.sum(eigen_matrix, axis=1)
    # occu_sum  = np.sum(occu_matrix, axis=1)

    ## Extract energy, total DOS, and integrated DOS
    # lists initialization
    energy_dos_list     = np.empty(0)
    total_dos_list      = np.empty(0)
    integrated_dos_list = np.empty(0)
    path_dos = f".//total/array/set/set[@comment='spin 1']/r"
    for element_dos in root.findall(path_dos):
        values_dos = list(map(float, element_dos.text.split()))
        energy_dos_list = np.append(energy_dos_list, values_dos[0])
        total_dos_list = np.append(total_dos_list, values_dos[1])
        integrated_dos_list = np.append(integrated_dos_list, values_dos[2])
    shift = efermi
    energy_dos_shift = energy_dos_list - shift

    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,
            energy_dos_shift, total_dos_list, integrated_dos_list)
