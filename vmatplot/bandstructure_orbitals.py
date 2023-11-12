#### Bandstructure for orbitals
# pylint: disable = C0103, C0114, C0116, C0301, C0302, C0321, R0913, R0914, R0915, W0612, W0105

import xml.etree.ElementTree as ET
import os

from vmatplot.algorithms import transpose_matrix
from vmatplot.commons import get_atoms_count

def extract_weights_kpoints(directory, spin_label, start_label=None, end_label=None):
    """
    Extracts the projected weight of eigenvalues for different orbitals (s, p, d) for specified spin electrons from a VASP calculation.

    This function parses the 'vasprun.xml' file to extract the projected weight of eigenvalues
    for each orbital type (s, p, and d orbitals) at each k-point for specified spin electrons. 
    The weights are summed over a range of atoms if specified.

    Args:
    directory (str): The directory path containing the VASP output files, specifically 'vasprun.xml'.
    spin_label (str): The label of spin ('spin1' for spin-up, 'spin2' for spin-down).
    start_label (int, optional): The starting index of atoms to be included in the sum. Defaults to the first atom.
    end_label (int, optional): The ending index of atoms to be included in the sum. Defaults to the last atom.

    Returns:
    tuple of lists: Contains multiple lists, each representing the weight of eigenvalues for a specific orbital type
    across all k-points. The order is s, py, pz, px, dxy, dyz, dz2, dx2y2, total d, and total p orbitals.
    """
    # Construct the path to the vasprun.xml file and parse it
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Atoms count
    atom_count = get_atoms_count(directory)

    # Initialize matrices to store the weight of eigenvalues for each orbital
    weights_kpoints_s = []
    weights_kpoints_py, weights_kpoints_pz, weights_kpoints_px = [], [], []
    weights_kpoints_dxy, weights_kpoints_dyz, weights_kpoints_dz2, weights_kpoints_dx2y2 = [], [], [], []
    weights_kpoints_d, weights_kpoints_p = [], []

    # Find the projected weight of eigenvalues section in the XML tree
    projected_section = root.find(".//projected/array")
    if projected_section is not None:
        # Find all k-point <set> elements within the projected section
        kpoint_sets = projected_section.findall(f".//set[@comment='{spin_label}']/set")
        for kpoint_set in kpoint_sets:
            weights_s, weights_py, weights_pz, weights_px = [], [], [], []
            weights_dxy, weights_dyz, weights_dz2, weights_dx2y2 = [], [], [], []
            for band_set in kpoint_set.findall(".//set"):
                r_elements = band_set.findall("./r")
                if r_elements:
                    # Extract and sum the weights for each orbital
                    if start_label is None:
                        start = 0
                    else: start = start_label
                    if end_label is None:
                        end = atom_count
                    else: end = end_label
                    weights_s.append(sum(float(r.text.split()[0]) for r in r_elements[start:end]))
                    weights_py.append(sum(float(r.text.split()[1]) for r in r_elements[start:end]))
                    weights_pz.append(sum(float(r.text.split()[2]) for r in r_elements[start:end]))
                    weights_px.append(sum(float(r.text.split()[3]) for r in r_elements[start:end]))
                    weights_dxy.append(sum(float(r.text.split()[4]) for r in r_elements[start:end]))
                    weights_dyz.append(sum(float(r.text.split()[5]) for r in r_elements[start:end]))
                    weights_dz2.append(sum(float(r.text.split()[6]) for r in r_elements[start:end]))
                    weights_dx2y2.append(sum(float(r.text.split()[7]) for r in r_elements[start:end]))
            # Sum of p and d orbitals for each k-point
            weights_d_kpoint = [sum(x) for x in zip(weights_dxy, weights_dyz, weights_dz2, weights_dx2y2)]
            weights_p_kpoint = [sum(x) for x in zip(weights_py, weights_pz, weights_px)]
            # Append weights for each orbital type
            weights_kpoints_s.append(weights_s)
            weights_kpoints_py.append(weights_py)
            weights_kpoints_pz.append(weights_pz)
            weights_kpoints_px.append(weights_px)
            weights_kpoints_dxy.append(weights_dxy)
            weights_kpoints_dyz.append(weights_dyz)
            weights_kpoints_dz2.append(weights_dz2)
            weights_kpoints_dx2y2.append(weights_dx2y2)
            weights_kpoints_d.append(weights_d_kpoint)
            weights_kpoints_p.append(weights_p_kpoint)
    else:
        print("Projected weight section not found in the XML file.")
    return (weights_kpoints_s,                                                                      # 0
            weights_kpoints_py, weights_kpoints_pz, weights_kpoints_px,                             # 1, 2, 3
            weights_kpoints_dxy, weights_kpoints_dyz, weights_kpoints_dz2, weights_kpoints_dx2y2,   # 4, 5, 6, 7
            weights_kpoints_d,                                                                      # -2
            weights_kpoints_p)                                                                      # -1

def extract_weights_kpoints_nonpolarized(directory, start_label=None, end_label=None):
    return extract_weights_kpoints(directory, "spin1", start_label, end_label)

def extract_weights_kpoints_spinUp(directory, start_label=None, end_label=None):
    return extract_weights_kpoints(directory, "spin1", start_label, end_label)

def extract_weights_kpoints_spinDown(directory, start_label=None, end_label=None):
    return extract_weights_kpoints(directory, "spin2", start_label, end_label)

def extract_weights_bands(directory, spin_label, start_label=None, end_label=None):
    """
    Extracts and transposes the weight of eigenvalues for different orbitals across bands.

    This function is designed to work with VASP calculation outputs. It extracts the projected weight of eigenvalues
    for different orbitals (s, p, d) across bands for specified spin states (spin-up or spin-down). The function
    allows for the selection of a specific range of atoms by specifying start and end labels.

    Args:
    - directory (str): The directory path containing the 'vasprun.xml' file from a VASP calculation.
    - spin_label (str): Specifies the spin state. Use "spin1" for spin-up and "spin2" for spin-down.
    - start_label (int, optional): The starting index of atoms to consider for weight extraction. Defaults to None, which considers the first atom.
    - end_label (int, optional): The ending index of atoms to consider for weight extraction. Defaults to None, which considers up to the last atom.

    Returns:
    - tuple of lists: Each list within the tuple represents the transposed weight of eigenvalues for a specific orbital type across all bands. The order is:
        0: s orbital
        1: py orbital
        2: pz orbital
        3: px orbital
        4: dxy orbital
        5: dyz orbital
        6: dz2 orbital
        7: d(x2-y2) orbital
        -2: Total weight for all d orbitals
        -1: Total weight for all p orbitals
    
    Example Usage:
    # Extracting weights for spin-up electrons across all atoms
    weights_for_bands = extract_weights_bands("/path/to/directory", "spin1")
    s_orbital_weights = weights_for_bands[0]  # Weights for s orbital across bands
    """
    weights_kpoints = extract_weights_kpoints(directory, spin_label, start_label, end_label)
    weights_bands_s = transpose_matrix(weights_kpoints[0])
    weights_bands_py = transpose_matrix(weights_kpoints[1])
    weights_bands_pz = transpose_matrix(weights_kpoints[2])
    weights_bands_px = transpose_matrix(weights_kpoints[3])
    weights_bands_dxy = transpose_matrix(weights_kpoints[4])
    weights_bands_dyz = transpose_matrix(weights_kpoints[5])
    weights_bands_dz2 = transpose_matrix(weights_kpoints[6])
    weights_bands_dx2y2 = transpose_matrix(weights_kpoints[7])
    weights_bands_d = transpose_matrix(weights_kpoints[-2])
    weights_bands_p = transpose_matrix(weights_kpoints[-1])
    return (weights_bands_s,                                                                                # 0
            weights_bands_py, weights_bands_pz, weights_bands_px,                                           # 1, 2, 3
            weights_bands_dxy, weights_bands_dyz, weights_bands_dz2, weights_bands_dx2y2,                   # 4, 5, 6, 7
            weights_bands_d,                                                                                # -2
            weights_bands_p                                                                                 # -1
            )

def extract_weights_bands_nonpolarized(directory, start_label=None, end_label=None):
    return extract_weights_bands(directory, "spin1", start_label, end_label)

def extract_weights_bands_spinUp(directory, start_label=None, end_label=None):
    return extract_weights_bands(directory, "spin1", start_label, end_label)

def extract_weights_bands_spinDown(directory, start_label=None, end_label=None):
    return extract_weights_bands(directory, "spin2", start_label, end_label)
