#### Bandstructure
# pylint: disable = C0103, C0114, C0116, C0301, C0302, C0321, R0913, R0914, R0915, W0612, W0105

import xml.etree.ElementTree as ET
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vmatplot.algorithms import transpose_matrix
from vmatplot.commons import extract_fermi
from vmatplot.output import canvas_setting, color_sampling
from vmatplot.DoS import extract_dos
from vmatplot.bandgap import extract_bandgap_outcar

def extract_high_sym(directory):
    """
    Extracts the high symmetry lines from the KPOINTS file in a VASP calculation directory.
    """
    # Open and read the KPOINTS file
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        kpoints_file = kpoints_opt_path
    elif os.path.exists(kpoints_file_path):
        kpoints_file = kpoints_file_path
    with open(kpoints_file, "r", encoding="utf-8") as file:
        KPOINTS = file.readlines()
    # Check if the KPOINTS file is in line-mode
    if KPOINTS[2][0] not in ("l", "L"):
        raise ValueError(f"Expected 'L' on the third line of KPOINTS file, got: {KPOINTS[2]}")
    # Initialize a set to store unique high symmetry points
    high_symmetry_points = set()
    # Read the high symmetry points from the KPOINTS file
    for i in range(4, len(KPOINTS)):
        tokens = KPOINTS[i].strip().split()
        if tokens and tokens[-1].isalpha():
            high_symmetry_points.add(tokens[-1])
    # The set of high symmetry points
    sets = high_symmetry_points
    return list(sets)

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
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        kpoints_file = kpoints_opt_path
    elif os.path.exists(kpoints_file_path):
        kpoints_file = kpoints_file_path

    with open(kpoints_file, "r", encoding="utf-8") as file:
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
    return kpoints_format, lines, list(sets), limits

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
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for kpoint in root.findall("./calculation/eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']/v"):
            # Split the text content of the 'v' element to get the individual coordinate strings
            # Convert each coordinate string to a float and create a list of coordinates
            coords = [float(x) for x in kpoint.text.split()]
            # Append the list of coordinates to the kpoints list
            kpoints.append(coords)
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
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

    The resulting cumulative distances serve as the x-axis values (k-points) in a bandstructure plot.
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
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")

    weight_list = []
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for weight in root.findall(".//eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='weights']/v"): # <varray name="weights" >
            weight_list.append(float(weight.text))
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        for weight in root.findall(".//varray[@name='weights']/v"): # <varray name="weights" >
            weight_list.append(float(weight.text))
    return weight_list

def extract_kpoints_count(directory):
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    # Find the kpoints varray
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        kpoints_varray = root.find("./calculation/eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']")
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
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
    eigen_lines = extract_eigenvalues_bands_nonpolarized(directory)
    return len(eigen_lines)

def kpoints_coordinate(directory):
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        kpoints_file = kpoints_opt_path
    elif os.path.exists(kpoints_file_path):
        kpoints_file = kpoints_file_path

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
        min_distance = float("inf")
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

def high_symmetry_coordinates(directory):
    """
    This function extracts the coordinates of high symmetry points from the KPOINTS file.

    Args:
    directory (str): The directory path that contains the VASP KPOINTS file.
    
    Returns:
    list: A list of coordinates for the high symmetry points in the Brillouin zone.
    """
    # Retrieve the coordinates of the high symmetry points
    high_symmetry_points = kpoints_coordinate(directory)
    # Extract the coordinates from the dictionary and store them in a list
    coordinates_list = list(high_symmetry_points.values())
    return coordinates_list

def high_symmetry_path(directory):
    """
    This function extracts the x-axis values (cumulative path distances) for the high symmetry points
    in a bandstructure plot.

    Args:
    directory (str): The directory path that contains the VASP output files.
    
    Returns:
    list: A list of x-axis values for the high symmetry points in the bandstructure plot.
    """
    # Get the indices of high symmetry points in the k-point list
    high_symmetry_indices = kpoints_index(directory)
    # Calculate the cumulative path distances for the k-points
    path = extract_kpath(directory)
    # Initialize a list to store the x-axis values for the high symmetry points
    high_sym_path = []
    # Iterate over the high symmetry points and their indices
    for index in high_symmetry_indices.values():
        # Append the corresponding path distance to the list
        high_sym_path.append(path[index])
    # Return the list of x-axis values
    return high_sym_path

def clean_kpoints(kpoints_list, tol=1e-10):
    kpoints_list[np.isclose(kpoints_list, 0, atol=tol)] = 0
    return kpoints_list

def extract_eigenvalues_kpoints(directory, spin_label):
    """
    Extracts the eigenvalues for each k-point from a VASP vasprun.xml file considering spin polarization.

    Args:
        directory (str): The directory path that contains the VASP vasprun.xml file.
        spin_label (str): The spin channel label ('spin1' or 'spin2').

    Returns:
        list of lists: A matrix where each sublist contains the eigenvalues for a specific k-point and spin channel.

    This function parses the vasprun.xml file to extract the electronic energy levels (eigenvalues)
    at each k-point in the reciprocal lattice for the material being studied, considering the specified spin channel.
    These eigenvalues are crucial for analyzing the material's electronic structure, such as plotting band structures.
    """
    xml_file = os.path.join(directory, "vasprun.xml")
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Initialize a list to store the eigenvalues for each k-point
    eigenvalues_matrix = []
    # Find the eigenvalues section in the XML tree
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        eigenvalues_section = root.find("./calculation/eigenvalues_kpoints_opt[@comment='kpoints_opt']/eigenvalues")
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        eigenvalues_section = root.find("./calculation/eigenvalues")
    if eigenvalues_section is not None:
        # Find all k-point <set> elements within the eigenvalues section
        # kpoint_sets = eigenvalues_section.findall(".//set/set/set")
        kpoint_sets = eigenvalues_section.findall(f".//set/set[@comment='{spin_label}']/set")
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

def extract_eigenvalues_kpoints_nonpolarized(directory):
    return extract_eigenvalues_kpoints(directory, "spin 1")

def extract_eigenvalues_kpoints_spinUp(directory):
    return extract_eigenvalues_kpoints(directory, "spin 1")

def extract_eigenvalues_kpoints_spinDown(directory):
    return extract_eigenvalues_kpoints(directory, "spin 2")

def extract_eigenvalues_bands(directory, spin_label):
    """
    Extracts and transposes the eigenvalues for each band from a VASP calculation.

    This function is designed to work with data from VASP (Vienna Ab initio Simulation Package) calculations. 
    It extracts the eigenvalues associated with each k-point for a given spin orientation (either 'spin1' or 'spin2'), 
    and then transposes the matrix so that each row represents a band and each column represents a k-point.

    Args:
        directory (str): The directory path that contains the VASP vasprun.xml file.
        spin_label (str): The spin label ('spin 1' or 'spin 2') for which the eigenvalues are to be extracted. 
                          'spin 1' typically refers to spin-up and 'spin 2' to spin-down in spin-polarized calculations.

    Returns:
        list of lists: A transposed matrix of eigenvalues where each row represents a band and each column represents a k-point. 
                       This format is useful for plotting band structures and analyzing the electronic properties of materials.

    Example:
        # Extract eigenvalues for 'spin 1' (spin-up) orientation
        directory = "/path/to/vasp/output"
        spin_label = "spin1"
        bands_matrix = extract_eigenvalues_bands(directory, spin_label)
        # 'bands_matrix' now contains the eigenvalues with bands as rows and k-points as columns
    """
    # Extract the eigenvalues for each k-point
    eigenvalues_matrix = extract_eigenvalues_kpoints(directory, spin_label)
    # Transpose the matrix so that bands are rows and k-points are columns
    transposed_eigenvalues_matrix = transpose_matrix(eigenvalues_matrix)
    # Return the transposed matrix of eigenvalues
    return transposed_eigenvalues_matrix

def extract_eigenvalues_bands_nonpolarized(directory):
    return extract_eigenvalues_bands(directory, "spin 1")

def extract_eigenvalues_bands_spinUp(directory):
    return extract_eigenvalues_bands(directory, "spin 1")

def extract_eigenvalues_bands_spinDown(directory):
    return extract_eigenvalues_bands(directory, "spin 2")

global_tolerance = 0.8

def extract_eigenvalues_conductionBands(directory, spin_label):
    eigenvalues_matrix = extract_eigenvalues_bands(directory, spin_label)
    conduction_bands = []
    TOLERANCE = global_tolerance
    current_LUMO = extract_bandgap_outcar(directory)[2]
    current_HOMO = extract_bandgap_outcar(directory)[1]
    for eigenvalues_bands in eigenvalues_matrix:
        if np.min(eigenvalues_bands) >= current_LUMO-TOLERANCE:
            conduction_bands.append(eigenvalues_bands)
    return conduction_bands

def extract_eigenvalues_valenceBands(directory, spin_label):
    eigenvalues_matrix = extract_eigenvalues_bands(directory, spin_label)
    valence_bands = []
    TOLERANCE = global_tolerance
    current_LUMO = extract_bandgap_outcar(directory)[2]
    current_HOMO = extract_bandgap_outcar(directory)[1]
    for eigenvalues_bands in eigenvalues_matrix:
        if np.max(eigenvalues_bands) <= current_HOMO+TOLERANCE:
            valence_bands.append(eigenvalues_bands)
    return valence_bands

def extract_eigenvalues_conductionBands_nonpolarized(directory):
    return extract_eigenvalues_conductionBands(directory, "spin 1")

def extract_eigenvalues_valenceBands_nonpolarized(directory):
    return extract_eigenvalues_valenceBands(directory, "spin 1")

def extract_eigenvalues_conductionBands_spinUp(directory):
    return extract_eigenvalues_conductionBands(directory, "spin 1")

def extract_eigenvalues_valenceBands_spinUp(directory):
    return extract_eigenvalues_valenceBands(directory, "spin 1")

def extract_eigenvalues_conductionBands_spinDown(directory):
    return extract_eigenvalues_conductionBands(directory, "spin 2")

def extract_eigenvalues_valenceBands_spinDown(directory):
    return extract_eigenvalues_valenceBands(directory, "spin 2")

def extract_high_sym_intersections(directory, spin_label):
    """
    Extracts the eigenvalues at high symmetry points for each band in a VASP bandstructure calculation,
    using the path value (x) in the bandstructure plot and the corresponding eigenvalue (y).

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are lists of tuples, where each tuple represents (path, eigenvalue) coordinates of intersection 
              points at that high symmetry point. 'path' is the x-coordinate, and 'eigenvalue' is the y-coordinate.
    """
    # Extract the eigenvalues (bands as rows, k-points as columns)
    eigenvalues_bands = extract_eigenvalues_bands(directory, spin_label)

    # Get the path distances for each k-point along the bandstructure path
    path = extract_kpath(directory)

    # Extract the high symmetry points and their positions in the bandstructure
    high_symmetry_indices = kpoints_index(directory)  # Get the indices of high symmetry points
    high_symmetry_labels = extract_high_sym(directory)  # Get high symmetry labels

    # Initialize a dictionary to store the intersection points
    intersections = {}

    # Iterate over each high symmetry point and its corresponding index
    for label, index in high_symmetry_indices.items():
        # Initialize a list to store (path, eigenvalue) coordinates at this high symmetry point
        intersections[label] = []
        for band in eigenvalues_bands:
            # x is the path value (cumulative distance along the bandstructure path)
            x = path[index]  # The x-axis value is the path (cumulative distance) at the high symmetry point
            # y is the eigenvalue at that point
            y = band[index]  # The y-axis value is the eigenvalue at that k-point
            # Append the (path, eigenvalue) coordinates to the list for this high symmetry point
            intersections[label].append((x, y))

    return intersections

def extract_high_sym_intersections_with_fermi(directory, spin_label):
    """
    Extracts the eigenvalues at high symmetry points for each band in a VASP bandstructure calculation,
    using the path value (x) in the bandstructure plot and the corresponding eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are lists of tuples, where each tuple represents (path, eigenvalue - Fermi) coordinates of intersection 
              points at that high symmetry point. 'path' is the x-coordinate, and 'eigenvalue' is the y-coordinate minus Fermi.
    """
    # Extract the eigenvalues (bands as rows, k-points as columns)
    eigenvalues_bands = extract_eigenvalues_bands(directory, spin_label)

    # Get the Fermi energy
    fermi_energy = extract_fermi(directory)

    # Get the path distances for each k-point along the bandstructure path
    path = extract_kpath(directory)

    # Extract the high symmetry points and their positions in the bandstructure
    high_symmetry_indices = kpoints_index(directory)    # Get the indices of high symmetry points
    high_symmetry_labels = extract_high_sym(directory)  # Get high symmetry labels

    # Initialize a dictionary to store the intersection points
    intersections = {}

    # Iterate over each high symmetry point and its corresponding index
    for label, index in high_symmetry_indices.items():
        # Initialize a list to store (path, eigenvalue - Fermi) coordinates at this high symmetry point
        intersections[label] = []
        for band in eigenvalues_bands:
            # x is the path value (cumulative distance along the bandstructure path)
            x = path[index]  # The x-axis value is the path (cumulative distance) at the high symmetry point
            # y is the eigenvalue at that point, minus the Fermi energy
            y = band[index] - fermi_energy  # The y-axis value is the eigenvalue at that k-point minus the Fermi level
            # Append the (path, eigenvalue - Fermi) coordinates to the list for this high symmetry point
            intersections[label].append((x, y))

    return intersections

def extract_high_sym_valence_intersections(directory, spin_label):
    """
    Extracts the valence band eigenvalues (y < 0) at high symmetry points for each band in a VASP 
    bandstructure calculation, using the path value (x) in the bandstructure plot and the corresponding 
    eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are lists of tuples, where each tuple represents (path, eigenvalue - Fermi) coordinates of valence 
              band intersection points (where eigenvalue - Fermi < 0).
    """
    intersections_with_fermi = extract_high_sym_intersections_with_fermi(directory, spin_label)
    valence_intersections = {}

    # Filter for valence band intersections (y < 0)
    for label, points in intersections_with_fermi.items():
        valence_intersections[label] = [point for point in points if point[1] < 0]

    return valence_intersections

def extract_high_sym_conduction_intersections(directory, spin_label):
    """
    Extracts the conduction band eigenvalues (y > 0) at high symmetry points for each band in a VASP 
    bandstructure calculation, using the path value (x) in the bandstructure plot and the corresponding 
    eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are lists of tuples, where each tuple represents (path, eigenvalue - Fermi) coordinates of conduction 
              band intersection points (where eigenvalue - Fermi > 0).
    """
    intersections_with_fermi = extract_high_sym_intersections_with_fermi(directory, spin_label)
    conduction_intersections = {}

    # Filter for conduction band intersections (y > 0)
    for label, points in intersections_with_fermi.items():
        conduction_intersections[label] = [point for point in points if point[1] > 0]

    return conduction_intersections

def extract_high_sym_min_conduction_intersections(directory, spin_label):
    """
    Extracts the lowest conduction band eigenvalue (minimum y > 0) at high symmetry points for each band 
    in a VASP bandstructure calculation, using the path value (x) in the bandstructure plot and the corresponding 
    eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are tuples representing (path, min_eigenvalue - Fermi) for the lowest conduction band intersection 
              point (where min_eigenvalue - Fermi > 0).
    """
    intersections_with_fermi = extract_high_sym_conduction_intersections(directory, spin_label)
    min_conduction_intersections = {}

    # Find the minimum conduction band eigenvalue (y > 0) for each high symmetry point
    for label, points in intersections_with_fermi.items():
        if points:  # If there are conduction intersections
            min_point = min(points, key=lambda point: point[1])  # Get the point with the minimum y
            min_conduction_intersections[label] = min_point

    return min_conduction_intersections

def extract_high_sym_max_valence_intersections(directory, spin_label):
    """
    Extracts the highest valence band eigenvalue (maximum y < 0) at high symmetry points for each band 
    in a VASP bandstructure calculation, using the path value (x) in the bandstructure plot and the corresponding 
    eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are tuples representing (path, max_eigenvalue - Fermi) for the highest valence band intersection 
              point (where max_eigenvalue - Fermi < 0).
    """
    intersections_with_fermi = extract_high_sym_valence_intersections(directory, spin_label)
    max_valence_intersections = {}

    # Find the maximum valence band eigenvalue (y < 0) for each high symmetry point
    for label, points in intersections_with_fermi.items():
        if points:  # If there are valence intersections
            max_point = max(points, key=lambda point: point[1])  # Get the point with the maximum y
            max_valence_intersections[label] = max_point

    return max_valence_intersections

def create_matters_bs(matters_list):
    matters = []
    for current_matter in matters_list:
        bstype, label, directory, *optional = current_matter
        if not optional:
            color = "orbital"
            lstyle = "solid"
            alpha = 1.0
        elif len(optional) == 1:
            color = optional[0]
            lstyle = "solid"
            alpha = 1.0
        elif len(optional) == 2:
            color = optional[0]
            lstyle =optional[1]
            alpha = 1.0
        else:
            color, lstyle, alpha = optional[0], optional[1], optional[2]
        # Bandstructure plotting style: monocolor
        if bstype.lower() in ["monocolor", "monocolor nonpolarized"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            bands = extract_eigenvalues_bands_nonpolarized(directory)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, lstyle, alpha])
        elif bstype.lower() in ["monocolor spin up", "spin up monocolor"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            bands = extract_eigenvalues_bands_spinUp(directory)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, lstyle, alpha])
        elif bstype.lower() in ["monocolor spin down", "spin down monocolor"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            bands = extract_eigenvalues_bands_spinDown(directory)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, lstyle, alpha])
        # Bandstructure plotting style: bands
        elif bstype.lower() in ["bands", "bands nonpolarized"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            conduction_bands = extract_eigenvalues_conductionBands_nonpolarized(directory)
            valence_bands = extract_eigenvalues_valenceBands_nonpolarized(directory)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, lstyle, alpha])
        elif bstype.lower() in ["bands spin up", "spin up bands"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            conduction_bands = extract_eigenvalues_conductionBands_spinUp(directory)
            valence_bands = extract_eigenvalues_valenceBands_spinUp(directory)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, lstyle, alpha])
        elif bstype.lower() in ["bands spin down", "spin down bands"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            conduction_bands = extract_eigenvalues_conductionBands_spinDown(directory)
            valence_bands = extract_eigenvalues_valenceBands_spinDown(directory)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, lstyle, alpha])
    return matters

def create_matters_bsDos(matters_list):
    matters = []
    for current_matter in matters_list:
        bstype, label, bs_dir, dos_dir, *optional = current_matter
        if not optional:
            color = "orbital"
            lstyle = "solid"
            alpha = 1.0
        elif len(optional) == 1:
            color = optional[0]
            lstyle = "solid"
            alpha = 1.0
        elif len(optional) == 2:
            color = optional[0]
            lstyle =optional[1]
            alpha = 1.0
        else:
            color, lstyle, alpha = optional[0], optional[1], optional[2]
        # Bandstructure plotting style: monocolor
        if bstype.lower() in ["monocolor", "monocolor nonpolarized"]:
            fermi_energy = extract_fermi(bs_dir)
            kpath = extract_kpath(bs_dir)
            bands = extract_eigenvalues_bands_nonpolarized(bs_dir)
            dos = extract_dos(dos_dir)
            matters.append([bstype, label, fermi_energy, kpath, bands, dos, color, lstyle, alpha])
        elif bstype.lower() in ["monocolor spin up", "spin up monocolor"]:
            fermi_energy = extract_fermi(bs_dir)
            kpath = extract_kpath(bs_dir)
            bands = extract_eigenvalues_bands_spinUp(bs_dir)
            dos = extract_dos(dos_dir)
            matters.append([bstype, label, fermi_energy, kpath, bands, dos, color, lstyle, alpha])
        elif bstype.lower() in ["monocolor spin down", "spin down monocolor"]:
            fermi_energy = extract_fermi(bs_dir)
            kpath = extract_kpath(bs_dir)
            bands = extract_eigenvalues_bands_spinDown(bs_dir)
            dos = extract_dos(dos_dir)
            matters.append([bstype, label, fermi_energy, kpath, bands, dos, color, lstyle, alpha])
        # Bandstructure plotting style: bands
        elif bstype.lower() in ["bands", "bands nonpolarized"]:
            fermi_energy = extract_fermi(bs_dir)
            kpath = extract_kpath(bs_dir)
            conduction_bands = extract_eigenvalues_conductionBands_nonpolarized(bs_dir)
            valence_bands = extract_eigenvalues_valenceBands_nonpolarized(bs_dir)
            dos = extract_dos(dos_dir)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, dos, color, lstyle, alpha])
        elif bstype.lower() in ["bands spin up", "spin up bands"]:
            fermi_energy = extract_fermi(bs_dir)
            kpath = extract_kpath(bs_dir)
            conduction_bands = extract_eigenvalues_conductionBands_spinUp(bs_dir)
            valence_bands = extract_eigenvalues_valenceBands_spinUp(bs_dir)
            dos = extract_dos(dos_dir)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, dos, color, lstyle, alpha])
        elif bstype.lower() in ["bands spin down", "spin down bands"]:
            fermi_energy = extract_fermi(bs_dir)
            kpath = extract_kpath(bs_dir)
            conduction_bands = extract_eigenvalues_conductionBands_spinDown(bs_dir)
            valence_bands = extract_eigenvalues_valenceBands_spinDown(bs_dir)
            dos = extract_dos(dos_dir)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, dos, color, lstyle, alpha])
    return matters

def plot_bandstructure(title, eigen_range=None, matters_list=None, legend_loc="False"):
    # Help information
    help_info = """
    Usage: plot_bandstructure
        arg[0]: title;
        arg[1]: the range of eigenvalues, from -arg[1] to arg[1];
        arg[2]: matters list;
        arg[3]: legend location;
    """
    if title in ["help", "Help"]:
        print(help_info)
        return

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")

    # Data calling and plotting
    matters = create_matters_bs(matters_list)
    for matter in matters:
        current_label = matter[1]
        if matter[0].lower() in ["monocolor"]:
            fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_band = [eigenvalue - fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    plt.plot(matter[3], current_band, c=color_sampling(matter[5])[1], linestyle=matter[6], alpha=matter[7], label=f"Bandstructure {current_label}", zorder=4)
                else:
                    plt.plot(matter[3], current_band, c=color_sampling(matter[5])[1], linestyle=matter[6], alpha=matter[7], zorder=4)
        elif matter[0] in ["bands"]:
            fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_conduction_band = [eigenvalue - fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    plt.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], alpha=matter[8], label=f"Conduction bands {current_label}", zorder=4)
                else:
                    plt.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], alpha=matter[8], zorder=4)
            for bands_index in range(0, len(matter[5])):
                current_valence_band = [eigenvalue - fermi for eigenvalue in matter[5][bands_index]]
                if bands_index == 0:
                    plt.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], alpha=matter[8], label=f"Valence bands {current_label}", zorder=4)
                else:
                    plt.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], alpha=matter[8], zorder=4)
        kpath_start = matter[3][0]
        kpath_end = matter[3][-1]
        fermi_last = matter[2]

    # Fermi energy as a horizon line
    plt.axhline(y = 0, color=fermi_color[0], alpha=1.00, linestyle="--", label="Fermi energy", zorder=2)
    efermi = fermi_last
    kpath_range = kpath_end-kpath_start
    # fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
    # plt.text(kpath_start+kpath_range*0.98, eigen_range*0.02, fermi_energy_text, fontsize=10, c=fermi_color[0], rotation=0, va = "bottom", ha="right", zorder=5)

    # Title
    plt.title(f"Bandstructure {title}")
    plt.ylabel("Energy (eV)")
    # plt.ylabel("$E-E_\text{F}$ (eV)")

    # y-axis range
    plt.ylim(eigen_range*(-1), eigen_range)
    # x-axis range
    plt.xlim(kpath_start, kpath_end)

    high_symmetry_paths = kpoints_path(matters_list[-1][2])
    high_symmetry_positions = list(high_symmetry_paths.values())
    # high_symmetry_positions = list(kpoints_path(matters_list[-1][2]).values())

    high_symmetry_positions.append(kpath_end)
    high_symmetry_labels = list(high_symmetry_paths.keys())
    # high_symmetry_labels = list(kpoints_path(matters_list[-1][2]).keys())

    high_symmetry_labels.append(high_symmetry_labels[0])
    plt.xticks(high_symmetry_positions, high_symmetry_labels)
    for k_loc in high_symmetry_positions[1:-1]:
        plt.axvline(x=k_loc, color=annotate_color[1], linestyle="--", zorder=1)

    # Legend
    if legend_loc is None:
        legend = plt.legend()
        legend.set_visible(False)
    else:
        legend = plt.legend(loc=legend_loc)

    plt.tight_layout()

def plot_bsDoS(title, eigen_range=None, dos_range=None, matters_list=None, legend_loc="False"):
    # Figure setting
    fig_setting = canvas_setting(12, 6)
    params = fig_setting[2]; plt.rcParams.update(params)

    fig = plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Colors calling
    bs_fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")

    # Data calling and plotting
    matters = create_matters_bsDos(matters_list)

    # Title
    fig.suptitle(f"Bandstructure and DoS {title}", fontsize=fig_setting[3][0], y=1.00)

    # ax1 Bandstructure
    ax1.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    ax1.set_title("Bandstructure", fontsize=fig_setting[3][1])

    for matter in matters:
        bs_current_label = matter[1]
        if matter[0].lower() in ["monocolor"]:
            bs_fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_band = [eigenvalue - bs_fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    ax1.plot(matter[3], current_band, c=color_sampling(matter[6])[1], linestyle=matter[7], alpha=matter[8], label=f"Bandstructure {bs_current_label}", zorder=4)
                else:
                    ax1.plot(matter[3], current_band, c=color_sampling(matter[6])[1], linestyle=matter[7], alpha=matter[8], zorder=4)
        elif matter[0].lower() in ["bands"]:
            bs_fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_conduction_band = [eigenvalue - bs_fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    ax1.plot(matter[3], current_conduction_band, c=color_sampling(matter[7])[2], linestyle=matter[8], alpha=matter[9], label=f"Conduction bands {bs_current_label}", zorder=4)
                else:
                    ax1.plot(matter[3], current_conduction_band, c=color_sampling(matter[7])[2], linestyle=matter[8], alpha=matter[9], zorder=4)
            for bands_index in range(0, len(matter[5])):
                current_valence_band = [eigenvalue - bs_fermi for eigenvalue in matter[5][bands_index]]
                if bands_index == 0:
                    ax1.plot(matter[3], current_valence_band, c=color_sampling(matter[7])[0], linestyle=matter[8], alpha=matter[9], label=f"Valence bands {bs_current_label}", zorder=4)
                else:
                    ax1.plot(matter[3], current_valence_band, c=color_sampling(matter[7])[0], linestyle=matter[8], alpha=matter[9], zorder=4)
        kpath_start = matter[3][0]
        kpath_end = matter[3][-1]
        bs_fermi_last = matter[2]

    # Fermi energy as a horizon line
    ax1.axhline(y = 0, color=bs_fermi_color[0], alpha=1.00, linestyle="--", label="Fermi energy", zorder=2)
    bs_efermi = bs_fermi_last
    kpath_range = kpath_end-kpath_start
    # bs_fermi_energy_text = f"Fermi energy\n{bs_efermi:.3f} (eV)"
    # ax1.text(kpath_start+kpath_range*0.98, eigen_range*0.02, bs_fermi_energy_text, fontsize=10, c=bs_fermi_color[0], rotation=0, va = "bottom", ha="right", zorder=5)

    # y-axis
    ax1.set_ylabel("Energy (eV)")
    ax1.set_ylim(eigen_range*(-1), eigen_range)
    # x-axis
    ax1.set_xlim(kpath_start, kpath_end)

    high_symmetry_paths = kpoints_path(matters_list[-1][2])
    high_symmetry_positions = list(high_symmetry_paths.values())
    # high_symmetry_positions = list(kpoints_path(matters_list[-1][2]).values())

    high_symmetry_positions.append(kpath_end)
    high_symmetry_labels = list(high_symmetry_paths.keys())
    # high_symmetry_labels = list(kpoints_path(matters_list[-1][2]).keys())

    high_symmetry_labels.append(high_symmetry_labels[0])

    ax1.set_xticks(high_symmetry_positions)
    ax1.set_xticklabels(high_symmetry_labels)

    for k_loc in high_symmetry_positions[1:-1]:
        ax1.axvline(x=k_loc, color=annotate_color[1], linestyle="--", zorder=1)

    # ax2 DoS
    ax2.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    ax2.set_title("DOS (a.u.)", fontsize=fig_setting[3][1])
    for matter in matters:
        DoS_current_label = matter[1]
        if matter[0].lower() in ["monocolor"]:
            dos_efermi = matter[5][0]
            plt.plot(matter[5][6], matter[5][5], c=color_sampling(matter[6])[1], label=f"Total DoS {DoS_current_label}", zorder = 2)

        elif matter[0].lower() in ["bands"]:
            dos_efermi = matter[6][0]
            # plt.plot(matter[6][6], matter[6][5], c=color_sampling(matter[7])[1], label=f"Total DoS {current_label}", zorder = 2)
            dos_data = matter[6][6]
            energy_data = matter[6][5]

            conduction_dos = [dos for dos, energy in zip(dos_data, energy_data) if energy > 0]
            conduction_energy = [energy for energy in energy_data if energy > 0]
            valence_dos = [dos for dos, energy in zip(dos_data, energy_data) if energy < 0]
            valence_energy = [energy for energy in energy_data if energy < 0]

            if conduction_dos and conduction_energy:
                ax2.plot(conduction_dos, conduction_energy, c=color_sampling(matter[7])[2])
            if valence_dos and valence_energy:
                ax2.plot(valence_dos, valence_energy, c=color_sampling(matter[7])[0])

    ax2.set_ylim(eigen_range*(-1), eigen_range)
    ax2.set_xlim(0, dos_range)

    ax2.set_xticks([])
    ax2.set_yticks([])

    shift = dos_efermi
    ax2.axhline(y = dos_efermi-shift, color=bs_fermi_color[0], alpha=1.00, linestyle="--", label="Fermi energy", zorder=2)

    if legend_loc not in [None, "False", False]:
        ax1.legend(loc=legend_loc)
        # ax2.legend(loc=legend_loc)
    plt.tight_layout()

def plot_bs_quadruple(suptitle, subtitles, tot_matters, eigen_range, legend_loc="upper right"):
    # figure Settings
    fig_setting = canvas_setting(16, 12)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

    # Colors calling
    annotate_color = color_sampling("Grey")
    fermi_color = color_sampling("Violet")

    # Title
    plt.suptitle(f"Bandstructures {suptitle}", fontsize=fig_setting[3][0], y=1.00)
    # plt.suptitle(f"{suptitle}", fontsize=fig_setting[3][0], y=1.00)

    # plot data
    for subplot_index in range(4):
        ax = axes_element[subplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        # ax.set_title(subtitles[subplot_index])

        matters = create_matters_bs(tot_matters[subplot_index])
        for matter in matters:
            current_label = matter[1]
            fermi = matter[2]
            if matter[0].lower() in ["monocolor"]:
                fermi = matter[2]
                for bands_index in range(0, len(matter[4])):
                    current_band = [eigenvalue - fermi for eigenvalue in matter[4][bands_index]]
                    if bands_index == 0:
                        ax.plot(matter[3], current_band, c=color_sampling(matter[5])[1], linestyle=matter[6], alpha=matter[7], label=f"Bandstructure {current_label}", zorder=4)
                    else:
                        ax.plot(matter[3], current_band, c=color_sampling(matter[5])[1], linestyle=matter[6], alpha=matter[7], zorder=4)
            elif matter[0] in ["bands"]:
                fermi = matter[2]
                for bands_index in range(0, len(matter[4])):
                    current_conduction_band = [eigenvalue - fermi for eigenvalue in matter[4][bands_index]]
                    if bands_index == 0:
                        ax.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], alpha=matter[8], label=f"Conduction bands {current_label}", zorder=4)
                    else:
                        ax.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], alpha=matter[8], zorder=4)
                for bands_index in range(0, len(matter[5])):
                    current_valence_band = [eigenvalue - fermi for eigenvalue in matter[5][bands_index]]
                    if bands_index == 0:
                        ax.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], alpha=matter[8], label=f"Valence bands {current_label}", zorder=4)
                    else:
                        ax.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], alpha=matter[8], zorder=4)
            kpath_start = matter[3][0]
            kpath_end = matter[3][-1]
            fermi_last = matter[2]

        # Fermi energy as a horizon line
        ax.axhline(y = 0, color=fermi_color[0], alpha=1.00, linestyle="--", zorder=2)

        # axis ranges
        ax.set_ylim(eigen_range*(-1), eigen_range)
        ax.set_xlim(kpath_start, kpath_end)

        # axis labels
        if subplot_index in [0,2]:
            ax.set_ylabel(r"Energy (eV)")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

        # High symmetry lines
        high_symmetry_paths = kpoints_path(tot_matters[subplot_index][-1][2])
        high_symmetry_positions = list(high_symmetry_paths.values())

        high_symmetry_positions.append(kpath_end)
        high_symmetry_labels = list(high_symmetry_paths.keys())

        high_symmetry_labels.append(high_symmetry_labels[0])
        ax.set_xticks(high_symmetry_positions, high_symmetry_labels)
        for k_loc in high_symmetry_positions[1:-1]:
            ax.axvline(x=k_loc, color=annotate_color[1], linestyle="--", zorder=1)

        # legends
        if legend_loc is None:
            legend = ax.legend()
            legend.set_visible(False)
        else:
            legend = ax.legend(loc=legend_loc)

        # subplots label
        orderlab_shift = 0.05
        x_loc = 0+orderlab_shift*0.75
        y_loc = 1-orderlab_shift
        ax.annotate(f"{subtitles[subplot_index]}",
                    xy=(x_loc, y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="left", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"},
                    zorder = 5)

    plt.tight_layout()

def plot_multi_bs(subfigures_amount, *args):
    help_info = "Usage: plot_multi_bs \n"
    # if subfigures_amount == 1:
    #     return plot_bs_single(*args)
    # elif subfigures_amount == 2:
    #     return plot_bs_double(*args)
    # elif subfigures_amount == 3:
    #     return plot_bs_triple(*args)
    # elif subfigures_amount == 4:
    #     return plot_bs_quadruple(*args)
    if subfigures_amount == 4:
        return plot_bs_quadruple(*args)
    # help information
    else:
        print(help_info)
