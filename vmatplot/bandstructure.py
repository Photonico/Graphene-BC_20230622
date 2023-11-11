#### Bandstructure
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

import xml.etree.ElementTree as ET
import os

import numpy as np

from vmatplot.algorithms import transpose_matrix

def extract_high_symlines(directory):
    """
    Extracts the high symmetry lines from the KPOINTS file in a VASP calculation directory.

    Args:
    directory (str): The directory path that contains the VASP KPOINTS file.
    
    Returns:
    tuple: A tuple containing the kpoints format, number of high symmetry lines, 
           a set of high symmetry points, and a list of limit points for each line.
    
    This function opens the KPOINTS file and reads the high symmetry lines specified within it.
    It checks for the expected format and extracts the high symmetry points and their limits.
    """
    # Open and read the KPOINTS file
    with open(os.path.join(directory, "KPOINTS"), "r", encoding="utf-8") as file:
        KPOINTS = file.readlines()
    # Check if the KPOINTS file is in line-mode
    if KPOINTS[2][0] not in ("l", "L"):
        raise ValueError(f"Expected 'L' on the third line of KPOINTS file, got: {KPOINTS[2]}")
    # Determine the format of the kpoints (cartesian or reciprocal)
    kpoints_format = "cartesian" if KPOINTS[3][0] in ["c", "C"] else "reciprocal"
    # Initialize a set to store unique high symmetry points
    high_symmetry_points = set()
    # Read the high symmetry points from the KPOINTS file
    for i in range(4, len(KPOINTS)):
        tokens = KPOINTS[i].strip().split()
        if tokens and tokens[-1].isalpha():
            high_symmetry_points.add(tokens[-1])
    # The number of unique high symmetry lines
    lines = len(high_symmetry_points)
    # The set of high symmetry points
    sets = high_symmetry_points
    # Extract non-empty lines from the KPOINTS file
    non_empty_lines = [line.split() for line in KPOINTS[4:] if line.strip()]
    # Extract the start and end points for each high symmetry line
    limits = []
    for i in range(0, len(non_empty_lines), 2):
        start = non_empty_lines[i]
        end = non_empty_lines[i+1]
        limits.append([start, end])
    # Return the kpoints format, number of lines, set of high symmetry points, and their limits
    return kpoints_format, lines, sets, limits


def extract_fermi_outcar(directory):
    """
    Extracts the Fermi energy level from the OUTCAR file of a VASP calculation.

    Args:
    directory (str): The directory path that contains the VASP OUTCAR file.

    Returns:
    float: The Fermi energy level extracted from the OUTCAR file.

    This function reads through the OUTCAR file line by line until it finds the line that
    contains the Fermi energy information. It then extracts and returns this value as a float.
    If the Fermi energy is not found, the function will return None.
    """
    # Open and read the OUTCAR file
    with open(os.path.join(directory, "OUTCAR"), "r", encoding="utf-8") as file:
        # Iterate over each line in the file
        for line in file:
            # Check if the line contains the Fermi energy information
            if "Fermi energy" in line:
                # Extract the Fermi energy value
                efermi = line.split()[2]
                # Return the Fermi energy as a float
                return float(efermi)
    # If the function reaches this point, the Fermi energy was not found
    # It's a good practice to handle this case, e.g., by logging an error or raising an exception
    # For now, we'll just return None
    return None

def extract_fermi(directory):
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for i in root.iter("i"):
        if "name" in i.attrib:
            if i.attrib["name"] == "efermi":
                fermi_energy = float(i.text)
                return fermi_energy
    raise ValueError("Fermi energy not found in vasprun.xml")

def extract_kpoints_eigenval(directory):
    """
    Extracts k-point coordinates from a VASP EIGENVAL file.

    Args:
    directory (str): The directory path that contains the VASP EIGENVAL file.

    Returns:
    numpy.ndarray: An array of k-point coordinates.

    The function reads the EIGENVAL file, which contains the eigenvalues for each k-point and band.
    It extracts the k-point coordinates from this file and returns them as a NumPy array.
    """
    # Open the EIGENVAL file
    with open(os.path.join(directory, "EIGENVAL"), "r", encoding="utf-8") as file:
        lines = file.readlines()
    # Initialize the list for k-points
    kpoints_list = []
    # Get the total number of bands and k-points from the sixth line of the file
    try:
        num_bands = int(lines[5].split()[2])
        num_kpoints = int(lines[5].split()[1])
    except IndexError as exc:
        # If the expected format is not found, raise an error
        raise ValueError("The EIGENVAL file does not have the expected format.") from exc
    # Calculate the number of lines in each k-point block (including the k-point line itself)
    block_size = num_bands + 1
    # Iterate over the EIGENVAL file to extract k-point coordinates
    # The k-point blocks start from line 7 (index 6) and are spaced by the block size
    for i in range(6, 6 + num_kpoints * block_size, block_size):
        # Extract the k-point coordinates from the first line of each block
        kpoint_line = lines[i].strip().split()
        # Check if there are enough elements in the line to represent a k-point
        if len(kpoint_line) < 4:
            continue  # Skip lines that don't have enough elements
        # Take the first three values as k-point coordinates (ignoring the weight)
        kpoint_coords = [float(kpoint_line[j]) for j in range(3)]
        kpoints_list.append(kpoint_coords)
    # Convert the k-point list to a NumPy array for efficiency
    kpoints_array = np.array(kpoints_list)
    return kpoints_array

def extract_highsym(directory):
    """
    Extracts the list of k-point coordinates from the vasprun.xml file of a VASP calculation.

    Args:
    directory (str): The directory path that contains the VASP vasprun.xml file.
    
    Returns:
    list: A list of k-point coordinates where each k-point is represented as a list of its coordinates.
    
    The function parses the vasprun.xml file to find the k-point coordinates used in the calculation.
    It looks for the 'varray' XML element with the attribute name set to 'kpointlist', which contains
    the k-point data. Each k-point is then extracted and appended to a list, which is returned.
    """
    # Construct the full path to the vasprun.xml file
    xml_file = os.path.join(directory, "vasprun.xml")
    # Parse the XML file
    tree = ET.parse(xml_file)
    # Get the root of the XML tree
    root = tree.getroot()
    # Initialize a list to store the k-point coordinates
    kpoints = []
    # Find all 'v' elements within 'varray' elements that have a 'name' attribute equal to 'kpointlist'
    # These elements contain the k-point coordinates
    for kpoint in root.findall(".//varray[@name='kpointlist']/v"):
        # Split the text content of the 'v' element to get the individual coordinate strings
        # Convert each coordinate string to a float and create a list of coordinates
        coords = [float(x) for x in kpoint.text.split()]
        # Append the list of coordinates to the kpoints list
        kpoints.append(coords)
    # Return the list of k-point coordinates
    return kpoints

def extract_kpath(directory):
    """
    Calculates the cumulative distances along a path through k-points in reciprocal space.

    Args:
    directory (str): The directory path that contains the VASP vasprun.xml file.

    Returns:
    list: A list of cumulative distances for the path through the k-points.

    This function uses the list of k-point coordinates extracted from the vasprun.xml file
    and computes the Euclidean distance between successive k-points. These distances are
    then summed cumulatively to provide a measure of the total path length traversed up to
    each k-point in the list.
    """
    # Extract the list of k-point coordinates
    kpoints = extract_highsym(directory)
    # Initialize the list for cumulative distances with the starting point (0 distance)
    cumulative_distances = [0]
    # Iterate over the list of k-points to calculate the path distances
    for i in range(1, len(kpoints)):
        # Calculate the Euclidean distance between successive k-points
        distance = np.linalg.norm(np.array(kpoints[i]) - np.array(kpoints[i-1]))
        # Add the distance from the previous total to get the new cumulative distance
        cumulative_distances.append(cumulative_distances[-1] + distance)
    # Return the list of cumulative distances
    return cumulative_distances

def extract_weight(directory):
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    weight_list = []
    for weight in root.findall(".//varray[@name='weights']/v"): # <varray name="weights" >
        weight_list.append(float(weight.text))
    return weight_list

def extract_kpoints_count(directory):
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Find the kpoints varray
    kpoints_varray = root.find(".//kpoints/varray[@name='kpointlist']")
    # Check if the varray exists
    if kpoints_varray is not None:
        # The number of kpoints is the number of <v> tags within the varray
        num_kpoints = len(kpoints_varray.findall("./v"))
        return num_kpoints
    else:
        print("The kpointlist section does not exist in the provided XML file.")
        return None

def extract_bands_count(directory):
    eigen_lines = extract_eigenvalues_bands(directory)
    return len(eigen_lines)

def kpoints_coordinate(directory):
    kpoints_file = os.path.join(directory, "KPOINTS")
    high_symmetry_points = {}

    with open(kpoints_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        # Assume high symmetry points start from the fifth line in the KPOINTS file
        for line in lines[4:]:      # High symmetry points coordinates usually start from line 5
            if line.strip():        # Ignore empty lines
                parts = line.split()
                if len(parts) == 4: # A line with coordinates should have four parts
                    # Assume the coordinates and label are separated by spaces, with the label being the last part
                    coords = tuple(map(float, parts[:3]))   # Convert the first three parts to float coordinates
                    label = parts[3]                        # The last part is the label of the high symmetry point
                    high_symmetry_points[label] = coords
    return high_symmetry_points

def kpoints_index(directory):
    # Retrieve the coordinates of the high symmetry points
    high_symmetry_points = kpoints_coordinate(directory)
    # Retrieve the list of kpoints
    kpoints_list = extract_highsym(directory)
    # Initialize a dictionary to store the indices of the high symmetry points
    high_symmetry_indices = {}
    # For each high symmetry point, find the closest kpoint
    for label, coord in high_symmetry_points.items():
        # Initialize a minimum distance to a very large number so any actual distance will be smaller
        min_distance = float('inf')
        min_index = None
        # Iterate over the kpoint list to find the kpoint closest to the current high symmetry point coordinates
        for index, kpoint in enumerate(kpoints_list):
            # Calculate the Euclidean distance
            distance = sum((c - k) ** 2 for c, k in zip(coord, kpoint)) ** 0.5
            # If this distance is the smallest so far, update the minimum distance and index
            if distance < min_distance:
                min_distance = distance
                min_index = index
        # Store the index of the closest kpoint
        high_symmetry_indices[label] = min_index
    return high_symmetry_indices

def kpoints_path(directory):
    """
    This function calculates the path distances for high symmetry points in the Brillouin zone.
    
    Args:
    directory (str): The directory path that contains the VASP output files.
    
    Returns:
    dict: A dictionary mapping high symmetry points to their cumulative path distances.
    
    The function works by first finding the indices of the high symmetry points in the k-point list.
    Then, it calculates the cumulative path distance for each k-point. Finally, it creates a dictionary
    that maps each high symmetry point label to its corresponding path distance.
    """
    # Get the indices of high symmetry points in the k-point list
    high_symmetry_indices = kpoints_index(directory)
    # Calculate the cumulative path distances for the k-points
    path = extract_kpath(directory)
    # Initialize a dictionary to store the high symmetry points and their path distances
    high_symmetry_paths = {}
    # Iterate over the high symmetry points and their indices
    for label, index in high_symmetry_indices.items():
        # Map the high symmetry point label to its path distance
        high_symmetry_paths[label] = path[index]
    # Return the dictionary of high symmetry points and their path distances
    return high_symmetry_paths

def extract_eigenvalues_kpoints(directory):
    """
    Extracts the eigenvalues for each k-point from a VASP vasprun.xml file.

    Args:
    directory (str): The directory path that contains the VASP vasprun.xml file.

    Returns:
    list of lists: A matrix where each sublist contains the eigenvalues for a specific k-point.

    This function parses the vasprun.xml file to extract the electronic energy levels (eigenvalues)
    at each k-point in the reciprocal lattice for the material being studied. These eigenvalues are
    crucial for analyzing the material's electronic structure, such as plotting band structures.
    """
    # Construct the path to the vasprun.xml file and parse it
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Initialize a list to store the eigenvalues for each k-point
    eigenvalues_matrix = []
    # Find the eigenvalues section in the XML tree
    eigenvalues_section = root.find(".//eigenvalues")
    if eigenvalues_section is not None:
        # Find all k-point <set> elements within the eigenvalues section
        kpoint_sets = eigenvalues_section.findall(".//set/set/set")
        if kpoint_sets:
            # Iterate over each k-point set to extract eigenvalues
            for kpoint_set in kpoint_sets:
                kpoint_eigenvalues = []
                # Iterate over each band's eigenvalue within the current k-point set
                for r in kpoint_set.findall("./r"):
                    # The energy eigenvalue is the first number in the <r> tag's text
                    energy = float(r.text.split()[0])
                    kpoint_eigenvalues.append(energy)
                # Append the list of eigenvalues for this k-point to the matrix
                eigenvalues_matrix.append(kpoint_eigenvalues)
        else:
            # Handle the case where no k-point <set> elements are found
            print("No k-point <set> elements found in the eigenvalues section.")
    else:
        # Handle the case where the eigenvalues section is missing
        print("Eigenvalues section not found in the XML file.")
    # Return the matrix of eigenvalues
    return eigenvalues_matrix

def extract_eigenvalues_bands(directory):
    """
    Extracts and transposes the eigenvalues matrix from a VASP calculation.

    This function first extracts the eigenvalues for each k-point using the
    extract_eigenvalues_kpoints function. It then transposes the resulting matrix
    so that each row corresponds to a band and each column corresponds to a k-point.

    Args:
    directory (str): The directory path that contains the VASP output files.

    Returns:
    list of lists: A transposed matrix of eigenvalues where each row contains the
    eigenvalues for a specific band across all k-points.
    """
    # Extract the eigenvalues for each k-point
    eigenvalues_matrix = extract_eigenvalues_kpoints(directory)
    # Transpose the matrix so that bands are rows and k-points are columns
    transposed_eigenvalues_matrix = transpose_matrix(eigenvalues_matrix)
    # Return the transposed matrix of eigenvalues
    return transposed_eigenvalues_matrix

def extract_eigenvalues_conductionBands(directory):
    """
    Extracts the conduction band eigenvalues from a VASP calculation.

    This function analyzes the eigenvalues for each band at every k-point and
    identifies those bands as conduction bands whose minimum eigenvalue is
    greater than the Fermi energy. It is assumed that the Fermi energy has
    been calculated correctly and is representative of the material's electronic
    structure.

    Args:
    directory (str): The directory path that contains the VASP output files, 
    specifically the 'vasprun.xml' file.

    Returns:
    list of lists: A matrix where each sublist contains the eigenvalues of a 
    conduction band at different k-points. Each sublist corresponds to a 
    conduction band.
    """
    eigenvalues_matrix = extract_eigenvalues_bands(directory)
    conduction_bands = []
    fermi_energy = extract_fermi(directory)
    for eigenvalues_bands in eigenvalues_matrix:
        if np.min(eigenvalues_bands) > fermi_energy:
            conduction_bands.append(eigenvalues_bands)
    return conduction_bands

def extract_eigenvalues_valenceBands(directory):
    """
    Extracts the valence band eigenvalues from a VASP calculation.

    This function goes through the eigenvalues for each band at every k-point and
    identifies those bands as valence bands whose maximum eigenvalue is less than
    the Fermi energy. The Fermi energy should be accurately determined to ensure
    correct identification of valence bands.

    Args:
    directory (str): The directory path that contains the VASP output files, 
    specifically the 'vasprun.xml' file.

    Returns:
    list of lists: A matrix where each sublist contains the eigenvalues of a 
    valence band at different k-points. Each sublist corresponds to a 
    valence band.
    """
    eigenvalues_matrix = extract_eigenvalues_bands(directory)
    valence_bands = []
    fermi_energy = extract_fermi(directory)
    for eigenvalues_bands in eigenvalues_matrix:
        if np.max(eigenvalues_bands) < fermi_energy:
            valence_bands.append(eigenvalues_bands)
    return valence_bands

def extract_eigenvalues_kpoints_spinUp(directory):
    """
    Extracts the projected eigenvalues for different orbitals (s, p, d) for spin-up electrons from a VASP calculation.

    This function parses the 'vasprun.xml' file from a VASP calculation to extract the projected eigenvalues
    for each orbital type (s, p, and d orbitals) at each k-point for spin-up electrons. The eigenvalues are
    organized into separate lists for each orbital type and each k-point.

    Args:
    directory (str): The directory path that contains the VASP output files, specifically 'vasprun.xml'.

    Returns:
    tuple of lists: Contains multiple lists, each representing the eigenvalues for a specific orbital type
    across all k-points. The order is s, py, pz, px, dxy, dyz, dz2, dx2y2, total d, and total p orbitals.
    """
    # Construct the path to the vasprun.xml file and parse it
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    ## Initialize matrices to store the eigenvalues every orbitals
    # s orbital
    eigenvalues_kpoints_s = []
    # p orbitals
    eigenvalues_kpoints_py = []
    eigenvalues_kpoints_pz = []
    eigenvalues_kpoints_px = []
    # d orbitals
    eigenvalues_kpoints_dxy = []
    eigenvalues_kpoints_dyz = []
    eigenvalues_kpoints_dz2 = []
    eigenvalues_kpoints_dx2y2 = []
    # summary
    eigenvalues_kpoints_d = []
    eigenvalues_kpoints_p = []

    # Find the projected eigenvalues section in the XML tree
    projected_section = root.find(".//projected/array")
    if projected_section is not None:
        # Find all k-point <set> elements within the projected section
        kpoint_sets = projected_section.findall(".//set[@comment='spin1']/set")
        for kpoint_set in kpoint_sets:
            eigenvalues_s_kpoint = []
            eigenvalues_py_kpoint = []
            eigenvalues_pz_kpoint = []
            eigenvalues_px_kpoint = []
            eigenvalues_dxy_kpoint = []
            eigenvalues_dyz_kpoint = []
            eigenvalues_dz2_kpoint = []
            eigenvalues_dx2y2_kpoint = []
            for band_set in kpoint_set.findall(".//set"):
                r_elements = band_set.findall("./r")
                if r_elements:
                    # Extract the value[0] as s orbital
                    s_value = float(r_elements[0].text.split()[0])
                    eigenvalues_s_kpoint.append(s_value)
                    # Extract the value[1] as py orbital
                    py_value = float(r_elements[0].text.split()[1])
                    eigenvalues_py_kpoint.append(py_value)
                    # Extract the value[2] as pz orbital
                    pz_value = float(r_elements[0].text.split()[2])
                    eigenvalues_pz_kpoint.append(pz_value)
                    # Extract the value[3] as px orbital
                    px_value = float(r_elements[0].text.split()[3])
                    eigenvalues_px_kpoint.append(px_value)
                    # Extract the value[4] as dxy orbital
                    dxy_value = float(r_elements[0].text.split()[4])
                    eigenvalues_dxy_kpoint.append(dxy_value)
                    # Extract the value[5] as dyz orbital
                    dyz_value = float(r_elements[0].text.split()[5])
                    eigenvalues_dyz_kpoint.append(dyz_value)
                    # Extract the value[6] as dz^2 orbital
                    dz2_value = float(r_elements[0].text.split()[6])
                    eigenvalues_dz2_kpoint.append(dz2_value)
                    # Extract the value[7] as d(x2-y2) orbital
                    dx2y2_value = float(r_elements[0].text.split()[7])
                    eigenvalues_dx2y2_kpoint.append(dx2y2_value)
                    # Sum of p and d orbitals
                    eigenvalues_d_kpoint = [sum(x) for x in zip(eigenvalues_dxy_kpoint, eigenvalues_dyz_kpoint, eigenvalues_dz2_kpoint, eigenvalues_dx2y2_kpoint)]
                    eigenvalues_p_kpoint = [sum(x) for x in zip(eigenvalues_py_kpoint, eigenvalues_pz_kpoint, eigenvalues_px_kpoint)]
            eigenvalues_kpoints_s.append(eigenvalues_s_kpoint)
            eigenvalues_kpoints_py.append(eigenvalues_py_kpoint)
            eigenvalues_kpoints_pz.append(eigenvalues_pz_kpoint)
            eigenvalues_kpoints_px.append(eigenvalues_px_kpoint)
            eigenvalues_kpoints_dxy.append(eigenvalues_dxy_kpoint)
            eigenvalues_kpoints_dyz.append(eigenvalues_dyz_kpoint)
            eigenvalues_kpoints_dz2.append(eigenvalues_dz2_kpoint)
            eigenvalues_kpoints_dx2y2.append(eigenvalues_dx2y2_kpoint)
            eigenvalues_kpoints_d.append(eigenvalues_d_kpoint)
            eigenvalues_kpoints_p.append(eigenvalues_p_kpoint)
    else:
        # Handle the case where the projected section is missing
        print("Projected eigenvalues section not found in the XML file.")
    # Return the matrices of eigenvalues
    return (eigenvalues_kpoints_s,                                                                                  # 0
            eigenvalues_kpoints_py, eigenvalues_kpoints_pz, eigenvalues_kpoints_px,                                 # 1, 2, 3
            eigenvalues_kpoints_dxy, eigenvalues_kpoints_dyz, eigenvalues_kpoints_dz2, eigenvalues_kpoints_dx2y2,   # 4, 5, 6, 7
            eigenvalues_kpoints_d,                                                                                  # -2
            eigenvalues_kpoints_p                                                                                   # -1
            )

def extract_eigenvalues_bands_spinUp(directory):
    """
    Extracts and transposes the eigenvalues matrix for spin-up electrons from a VASP calculation for different orbitals.

    This function first calls 'extract_eigenvalues_kpoints_spinUp' to extract the eigenvalues for each orbital type 
    (s, p, and d orbitals) at each k-point for spin-up electrons. It then transposes the resulting matrices so that 
    each row corresponds to a band and each column corresponds to a k-point. This is useful for band structure 
    analysis where eigenvalues are typically plotted as bands across k-points.

    Args:
    directory (str): The directory path that contains the VASP output files.

    Returns:
    tuple of lists: Contains multiple lists, each representing the transposed eigenvalues for a specific orbital type
    across all bands. The order is s, py, pz, px, dxy, dyz, dz2, dx2y2, total d, and total p orbitals.
    """
    eigenvalues_kpoints = extract_eigenvalues_kpoints_spinUp(directory)
    eigenvalues_bands_s = transpose_matrix(eigenvalues_kpoints[0])
    eigenvalues_bands_py = transpose_matrix(eigenvalues_kpoints[1])
    eigenvalues_bands_pz = transpose_matrix(eigenvalues_kpoints[2])
    eigenvalues_bands_px = transpose_matrix(eigenvalues_kpoints[3])
    eigenvalues_bands_dxy = transpose_matrix(eigenvalues_kpoints[4])
    eigenvalues_bands_dyz = transpose_matrix(eigenvalues_kpoints[5])
    eigenvalues_bands_dz2 = transpose_matrix(eigenvalues_kpoints[6])
    eigenvalues_bands_dx2y2 = transpose_matrix(eigenvalues_kpoints[7])
    eigenvalues_bands_d = transpose_matrix(eigenvalues_kpoints[-2])
    eigenvalues_bands_p = transpose_matrix(eigenvalues_kpoints[-1])
    return (eigenvalues_bands_s,                                                                                # 0
            eigenvalues_bands_py, eigenvalues_bands_pz, eigenvalues_bands_px,                                   # 1, 2, 3
            eigenvalues_bands_dxy, eigenvalues_bands_dyz, eigenvalues_bands_dz2, eigenvalues_bands_dx2y2,       # 4, 5, 6, 7
            eigenvalues_bands_d,                                                                                # -2
            eigenvalues_bands_p                                                                                 # -1
            )

def extract_eigenvalues_kpoints_spinDown(directory):
    """
    Extracts the projected eigenvalues for different orbitals (s, p, d) for spin-down electrons from a VASP calculation.

    This function parses the 'vasprun.xml' file from a VASP calculation to extract the projected eigenvalues
    for each orbital type (s, p, and d orbitals) at each k-point for spin-down electrons. The eigenvalues are
    organized into separate lists for each orbital type and each k-point.

    Args:
    directory (str): The directory path that contains the VASP output files, specifically 'vasprun.xml'.

    Returns:
    tuple of lists: Contains multiple lists, each representing the eigenvalues for a specific orbital type
    across all k-points. The order is s, py, pz, px, dxy, dyz, dz2, dx2y2, total d, and total p orbitals.
    """
    # Construct the path to the vasprun.xml file and parse it
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    ## Initialize matrices to store the eigenvalues every orbitals
    # s orbital
    eigenvalues_kpoints_s = []
    # p orbitals
    eigenvalues_kpoints_py = []
    eigenvalues_kpoints_pz = []
    eigenvalues_kpoints_px = []
    # d orbitals
    eigenvalues_kpoints_dxy = []
    eigenvalues_kpoints_dyz = []
    eigenvalues_kpoints_dz2 = []
    eigenvalues_kpoints_dx2y2 = []
    # summary
    eigenvalues_kpoints_d = []
    eigenvalues_kpoints_p = []

    # Find the projected eigenvalues section in the XML tree
    projected_section = root.find(".//projected/array")
    if projected_section is not None:
        # Find all k-point <set> elements within the projected section
        kpoint_sets = projected_section.findall(".//set[@comment='spin1']/set")
        for kpoint_set in kpoint_sets:
            eigenvalues_s_kpoint = []
            eigenvalues_py_kpoint = []
            eigenvalues_pz_kpoint = []
            eigenvalues_px_kpoint = []
            eigenvalues_dxy_kpoint = []
            eigenvalues_dyz_kpoint = []
            eigenvalues_dz2_kpoint = []
            eigenvalues_dx2y2_kpoint = []
            for band_set in kpoint_set.findall(".//set"):
                r_elements = band_set.findall("./r")
                if r_elements:
                    # Extract the value[0] as s orbital
                    s_value = float(r_elements[-1].text.split()[0])
                    eigenvalues_s_kpoint.append(s_value)
                    # Extract the value[1] as py orbital
                    py_value = float(r_elements[-1].text.split()[1])
                    eigenvalues_py_kpoint.append(py_value)
                    # Extract the value[2] as pz orbital
                    pz_value = float(r_elements[-1].text.split()[2])
                    eigenvalues_pz_kpoint.append(pz_value)
                    # Extract the value[3] as px orbital
                    px_value = float(r_elements[-1].text.split()[3])
                    eigenvalues_px_kpoint.append(px_value)
                    # Extract the value[4] as dxy orbital
                    dxy_value = float(r_elements[-1].text.split()[4])
                    eigenvalues_dxy_kpoint.append(dxy_value)
                    # Extract the value[5] as dyz orbital
                    dyz_value = float(r_elements[-1].text.split()[5])
                    eigenvalues_dyz_kpoint.append(dyz_value)
                    # Extract the value[6] as dz^2 orbital
                    dz2_value = float(r_elements[-1].text.split()[6])
                    eigenvalues_dz2_kpoint.append(dz2_value)
                    # Extract the value[7] as d(x2-y2) orbital
                    dx2y2_value = float(r_elements[-1].text.split()[7])
                    eigenvalues_dx2y2_kpoint.append(dx2y2_value)
                    # Sum of p and d orbitals
                    eigenvalues_d_kpoint = [sum(x) for x in zip(eigenvalues_dxy_kpoint, eigenvalues_dyz_kpoint, eigenvalues_dz2_kpoint, eigenvalues_dx2y2_kpoint)]
                    eigenvalues_p_kpoint = [sum(x) for x in zip(eigenvalues_py_kpoint, eigenvalues_pz_kpoint, eigenvalues_px_kpoint)]
            eigenvalues_kpoints_s.append(eigenvalues_s_kpoint)
            eigenvalues_kpoints_py.append(eigenvalues_py_kpoint)
            eigenvalues_kpoints_pz.append(eigenvalues_pz_kpoint)
            eigenvalues_kpoints_px.append(eigenvalues_px_kpoint)
            eigenvalues_kpoints_dxy.append(eigenvalues_dxy_kpoint)
            eigenvalues_kpoints_dyz.append(eigenvalues_dyz_kpoint)
            eigenvalues_kpoints_dz2.append(eigenvalues_dz2_kpoint)
            eigenvalues_kpoints_dx2y2.append(eigenvalues_dx2y2_kpoint)
            eigenvalues_kpoints_d.append(eigenvalues_d_kpoint)
            eigenvalues_kpoints_p.append(eigenvalues_p_kpoint)
    else:
        # Handle the case where the projected section is missing
        print("Projected eigenvalues section not found in the XML file.")
    # Return the matrices of eigenvalues
    return (eigenvalues_kpoints_s,                                                                                  # 0
            eigenvalues_kpoints_py, eigenvalues_kpoints_pz, eigenvalues_kpoints_px,                                 # 1, 2, 3
            eigenvalues_kpoints_dxy, eigenvalues_kpoints_dyz, eigenvalues_kpoints_dz2, eigenvalues_kpoints_dx2y2,   # 4, 5, 6, 7
            eigenvalues_kpoints_d,                                                                                  # -2
            eigenvalues_kpoints_p                                                                                   # -1
            )

def extract_eigenvalues_bands_spinDown(directory):
    """
    Extracts and transposes the eigenvalues matrix for spin-down electrons from a VASP calculation for different orbitals.

    This function first calls 'extract_eigenvalues_kpoints_spinDown' to extract the eigenvalues for each orbital type 
    (s, p, and d orbitals) at each k-point for spin-down electrons. It then transposes the resulting matrices so that 
    each row corresponds to a band and each column corresponds to a k-point. This is useful for band structure 
    analysis where eigenvalues are typically plotted as bands across k-points.

    Args:
    directory (str): The directory path that contains the VASP output files.

    Returns:
    tuple of lists: Contains multiple lists, each representing the transposed eigenvalues for a specific orbital type
    across all bands. The order is s, py, pz, px, dxy, dyz, dz2, dx2y2, total d, and total p orbitals.
    """
    eigenvalues_kpoints = extract_eigenvalues_kpoints_spinDown(directory)
    eigenvalues_bands_s = transpose_matrix(eigenvalues_kpoints[0])
    eigenvalues_bands_py = transpose_matrix(eigenvalues_kpoints[1])
    eigenvalues_bands_pz = transpose_matrix(eigenvalues_kpoints[2])
    eigenvalues_bands_px = transpose_matrix(eigenvalues_kpoints[3])
    eigenvalues_bands_dxy = transpose_matrix(eigenvalues_kpoints[4])
    eigenvalues_bands_dyz = transpose_matrix(eigenvalues_kpoints[5])
    eigenvalues_bands_dz2 = transpose_matrix(eigenvalues_kpoints[6])
    eigenvalues_bands_dx2y2 = transpose_matrix(eigenvalues_kpoints[7])
    eigenvalues_bands_d = transpose_matrix(eigenvalues_kpoints[-2])
    eigenvalues_bands_p = transpose_matrix(eigenvalues_kpoints[-1])
    return (eigenvalues_bands_s,                                                                                # 0
            eigenvalues_bands_py, eigenvalues_bands_pz, eigenvalues_bands_px,                                   # 1, 2, 3
            eigenvalues_bands_dxy, eigenvalues_bands_dyz, eigenvalues_bands_dz2, eigenvalues_bands_dx2y2,       # 4, 5, 6, 7
            eigenvalues_bands_d,                                                                                # -2
            eigenvalues_bands_p                                                                                 # -1
            )

def extract_eigenvalues_conductionBands_spinUp(directory):
    eigenvalues_set = extract_eigenvalues_bands_spinUp(directory)
    fermi_energy = extract_fermi(directory)
    conduction_spinUp_bands = []


# def extract_eigenvalues_conductionBands_spinDown(directory):

def extract_eigenvalues_kpoints_nonpolarized(directory):
    return extract_eigenvalues_kpoints_spinUp(directory)

def extract_eigenvalues_bands_nonpolarized(directory):
    return extract_eigenvalues_bands_spinUp(directory)

def rec_to_cart(klist_source, directory, crystal_type):
    # Define the reciprocal lattice vectors for different crystal types
    # These are the transformation matrices for converting reciprocal lattice
    # points to Cartesian coordinates.
    if crystal_type.lower() == "hcc":
        transform_matrix = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 2
    elif crystal_type.lower() == "bcc":
        transform_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2
    elif crystal_type.lower() == "hcp":
        # For HCP, the transformation depends on the ratio of the lattice constants a and c.
        # We need to read these from a file or define them here.
        # For example, let's assume we have them in a file named 'lattice_constants.txt'
        with open(os.path.join(directory, "lattice_constants.txt"), "r", encoding="utf-8") as file:
            a, c = map(float, file.readline().split())
        transform_matrix = np.array([[1, -1/np.sqrt(3), 0], [0, 2/np.sqrt(3), 0], [0, 0, a/c]])
    elif crystal_type.lower() == "sc":
        transform_matrix = np.identity(3)
    elif crystal_type.lower() == "graphene":
        a = 2.46  # Graphene's lattice constant in angstroms
        b1 = (2 * np.pi / a) * np.array([1 / np.sqrt(3), 1, 0])
        b2 = (2 * np.pi / a) * np.array([1 / np.sqrt(3), -1, 0])
        transform_matrix = np.array([b1, b2, [0, 0, 1]])
    else:
        raise ValueError(f"Unknown crystal type: {crystal_type}")
    # Convert the Klist from reciprocal to Cartesian coordinates
    cartesian_kpoints = np.dot(klist_source, transform_matrix.T)
    return cartesian_kpoints

def cart_to_rec(klist_source, directory, crystal_type):
    # Define the transformation matrices for converting Cartesian coordinates
    # to reciprocal lattice points for different crystal types.
    if crystal_type.lower() == "hcc":
        transform_matrix = np.linalg.inv(np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 2)
    elif crystal_type.lower() == "bcc":
        transform_matrix = np.linalg.inv(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2)
    elif crystal_type.lower() == "hcp":
        # For HCP, the transformation depends on the ratio of the lattice constants a and c.
        # We need to read these from a file or define them here.
        # For example, let's assume we have them in a file named 'lattice_constants.txt'
        with open(os.path.join(directory, "lattice_constants.txt"), "r", encoding="utf-8") as file:
            a, c = map(float, file.readline().split())
        transform_matrix = np.linalg.inv(np.array([[1, -1/np.sqrt(3), 0], [0, 2/np.sqrt(3), 0], [0, 0, a/c]]))
    elif crystal_type.lower() == "sc":
        transform_matrix = np.linalg.inv(np.identity(3))
    elif crystal_type.lower() == "graphene":
        a = 2.46  # Graphene's lattice constant in angstroms
        b1 = (2 * np.pi / a) * np.array([1 / np.sqrt(3), 1, 0])
        b2 = (2 * np.pi / a) * np.array([1 / np.sqrt(3), -1, 0])
        transform_matrix = np.linalg.inv(np.array([b1, b2, [0, 0, 1]]))
    else:
        raise ValueError(f"Unknown crystal type: {crystal_type}")
    # Convert the Kpoints from Cartesian to reciprocal coordinates
    reciprocal_kpoints = np.dot(klist_source, transform_matrix.T)
    return reciprocal_kpoints

def clean_kpoints(kpoints_list, tol=1e-10):
    kpoints_list[np.isclose(kpoints_list, 0, atol=tol)] = 0
    return kpoints_list

def create_matters_bs(matters_list):
    # bandstructure data type: "monocolor", "bands", "orbitals", "spin up", "spin up orbitals", "spin down", "spin down orbitals"
    """
    Prepares data for plotting band structures with various styles and options.

    This function processes a list of band structure specifications and extracts
    the necessary data for each specified material and plotting style. It supports
    multiple styles including monocolor, bands, orbitals, and spin-polarized bands.
    The function handles different data types and prepares them for visualization
    using the plot_bandstructure function.

    Args:
        matters_list (list of matter_list)
            A list where each sublist represents a specificband structure to be plotted. Each sublist containsthe following elements:
            - matter_list[0]: bstype (str)
                The plotting style. Supported styles include:
                    "monocolor", "bands", "orbitals", 
                    "spin up monocolor", "spin up bands", "spin up orbitals", 
                    "spin down monocolor", "spin down bands", "spin down orbitals".
            - matter_list[1]: label (str)
                The label for the data set. Use an empty string ("") if no label is needed.
            - matter_list[2]: directory (str)
                The directory path where the VASP calculation files are located.
            - matter_list[3]: color (str)
                The color family name for plotting. Specifies the color used in the plot.
            - matter_list[4]: alpha (float, optional)
                The opacity level of the plot line. Ranges from 0 (transparent) to 1 (opaque). If not provided, defaults to 1.0.

    Returns:
        list: A list of prepared data sets, each corresponding to an entry in matters_list.
              Each data set is a list containing the bstype, label, Fermi energy, k-path,
              band data (and/or orbital data), color, and alpha value. This list is ready
              to be used with the plot_bandstructure function for visualization.

    Example:
        input_list = [["bands", "Sample1", "/path/to/data1", "blue"],
                      ["monocolor", "Sample2", "/path/to/data2", "violet", 0.5]]
        prepared_data = create_matters_bs(input_list)
        # Now 'prepared_data' can be used with 'plot_bandstructure' for plotting.
    """
    matters = []
    if len(matters_list[0]) == 4:
        for matter_dir in matters_list:
            bstype, label, directory, color = matter_dir
            # Bandstructure plotting style: monocolor
            if bstype.lower() in ["monocolor"]:
                fermi_energy = extract_fermi(directory)
                kpath = extract_kpath(directory)
                bands = extract_eigenvalues_bands(directory)
                matters.append([bstype, label, fermi_energy, kpath, bands, color, 1.0])

            # Bandstructure plotting style: bands
            if bstype.lower() in ["bands"]:
                fermi_energy = extract_fermi(directory)
                kpath = extract_kpath(directory)
                conduction_bands = extract_eigenvalues_conductionBands(directory)
                valence_bands = extract_eigenvalues_valenceBands(directory)
                matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, 1.0])

            # Bandstructure plotting style: orbitals
            if bstype.lower() in ["orbitals"]:
                fermi_energy = extract_fermi(directory)
                kpath = extract_kpath(directory)
                orbitals_bands_set = extract_eigenvalues_bands_nonpolarized(directory)
                matters.append([bstype, label, fermi_energy, kpath, orbitals_bands_set, color, 1.0])

            # Bandstructure plotting style: spin up, spin up orbitals
            if bstype.lower() in ["spin up", "spin up monocolor", "spin up bands", "spin up orbitals"]:
                fermi_energy = extract_fermi(directory)
                kpath = extract_kpath(directory)
                orbitals_bands_set = extract_eigenvalues_bands_spinUp(directory)
                matters.append([bstype, label, fermi_energy, kpath, orbitals_bands_set, color, 1.0])

            # Bandstructure plotting style: spin down, spin down orbitals
            if bstype.lower() in ["spin down", "spin down monocolor", "spin down bands", "spin down orbitals"]:
                fermi_energy = extract_fermi(directory)
                kpath = extract_kpath(directory)
                orbitals_bands_set = extract_eigenvalues_bands_spinDown(directory)
                matters.append([bstype, label, fermi_energy, kpath, orbitals_bands_set, color, 1.0])

    elif len(matters_list[0]) == 5:
        for matter_dir in matters_list:
            bstype, label, directory, color, alpha = matter_dir

            # Bandstructure plotting style: monocolor
            if bstype.lower() in ["monocolor"]:
                fermi_energy = extract_fermi(directory)
                kpath = extract_kpath(directory)
                bands = extract_eigenvalues_bands(directory)
                matters.append([bstype, label, fermi_energy, kpath, bands, color, alpha])

            # Bandstructure plotting style: bands
            if bstype.lower() in ["bands"]:
                fermi_energy = extract_fermi(directory)
                kpath = extract_kpath(directory)
                conduction_bands = extract_eigenvalues_conductionBands(directory)
                valence_bands = extract_eigenvalues_valenceBands(directory)
                matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, alpha])

            # Bandstructure plotting style: orbitals
            if bstype.lower() in ["orbitals"]:
                fermi_energy = extract_fermi(directory)
                kpath = extract_kpath(directory)
                orbitals_bands_set = extract_eigenvalues_bands_nonpolarized(directory)
                matters.append([bstype, label, fermi_energy, kpath, orbitals_bands_set, color, alpha])

            # Bandstructure plotting style: spin up, spin up bands, spin up orbitals
            if bstype.lower() in ["spin up", "spin up monocolor", "spin up bands", "spin up orbitals"]:
                fermi_energy = extract_fermi(directory)
                kpath = extract_kpath(directory)
                orbitals_bands_set = extract_eigenvalues_bands_spinUp(directory)
                matters.append([bstype, label, fermi_energy, kpath, orbitals_bands_set, color, alpha])

            # Bandstructure plotting style: spin down, spin down bands, spin down orbitals
            if bstype.lower() in ["spin down", "spin down monocolor", "spin down bands", "spin down orbitals"]:
                fermi_energy = extract_fermi(directory)
                kpath = extract_kpath(directory)
                orbitals_bands_set = extract_eigenvalues_bands_spinDown(directory)
                matters.append([bstype, label, fermi_energy, kpath, orbitals_bands_set, color, alpha])

    return matters

# def create_matters_bsdos(matters_list):

# def create_matters_bspdos(matters_list):
