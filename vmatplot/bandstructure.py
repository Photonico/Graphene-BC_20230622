#### Bandstructure
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

import xml.etree.ElementTree as ET
import os
import numpy as np

from vmatplot.algorithms import transpose_matrix

def extract_high_symlines(directory):
    with open(os.path.join(directory, "KPOINTS"), "r", encoding="utf-8") as file:
        KPOINTS = file.readlines()
    file.close()
    if KPOINTS[2][0] not in ("l","L"):
        raise ValueError(f"Expected 'L' on the third line of KPOINTS file, got: {KPOINTS[2]}")
    # The format of KPOINTS
    kpoints_format = "cartesian" if KPOINTS[3][0] in ["c", "k"] else "reciprocal"
    # High symmetry points reading
    high_symmetry_points = set()
    for i in range(4, len(KPOINTS)):
        tokens = KPOINTS[i].strip().split()
        if tokens and tokens[-1].isalpha():
            high_symmetry_points.add(tokens[-1])
    lines = len(high_symmetry_points)
    sets = high_symmetry_points
    # print(f"The number of High Symmetry lines is {lines}")
    # Extracting non-empty lines
    non_empty_lines = []
    for line in KPOINTS[4:]:
        if line.strip():  # Check if the line is not empty
            non_empty_lines.append(line.split())
    # Extracting limits
    limits = []
    for i in range(0, len(non_empty_lines), 2):
        start = non_empty_lines[i]
        end = non_empty_lines[i+1]
        limits.append([start, end])
    return kpoints_format, lines, sets, limits

def extract_fermi_outcar(directory):
    with open(os.path.join(directory, "OUTCAR"), "r", encoding="utf-8") as file:
        for _, line in enumerate(file):
            if "Fermi energy" in line:
                efermi = line.split()[2]
                # print(f"The Fermi energy is: {efermi} eV")
                return float(efermi)

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

def extract_kpointlist_eigenval(directory):
    # Open the EIGENVAL file
    with open(os.path.join(directory, "EIGENVAL"), "r", encoding="utf-8") as file:
        lines = file.readlines()
    # Initialize the list for k-points
    kpoins_list = []
    # Get the total number of bands and k-points
    try:
        num_bands = int(lines[5].split()[2])
        num_kpoints = int(lines[5].split()[1])
    except IndexError as exc:
        raise ValueError("The EIGENVAL file does not have the expected format.") from exc
    # Calculate the number of lines in each k-point block (including the k-point line itself)
    block_size = num_bands + 1
    # Iterate over the EIGENVAL file to extract k-point coordinates
    for i in range(6, 6 + num_kpoints * block_size, block_size):
        # Extract the k-point coordinates
        kpoint_line = lines[i].strip().split()
        if len(kpoint_line) < 4:  # Check if there are enough elements in the line
            continue  # Skip lines that don't have enough elements
        kpoint_coords = [float(kpoint_line[j]) for j in range(3)]  # Take the first three values as k-point coordinates
        kpoins_list .append(kpoint_coords)
    # Convert the k-point list to a NumPy array for efficiency
    kpoints_array = np.array(kpoins_list)
    return kpoints_array

def extract_kpointlist(directory):
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # extract kpoint list
    kpoints = []
    for kpoint in root.findall(".//varray[@name='kpointlist']/v"):
        coords = [float(x) for x in kpoint.text.split()]
        kpoints.append(coords)
    return kpoints

def extract_kpath_distances(directory):
    kpoints = extract_kpointlist(directory)
    # convert kpoint list to energy list
    cumulative_distances = [0]
    for i in range(1, len(kpoints)):
        distance = np.linalg.norm(np.array(kpoints[i]) - np.array(kpoints[i-1]))
        cumulative_distances.append(cumulative_distances[-1] + distance)
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
    eigen_lines = extract_eigenvalues_lines(directory)
    return len(eigen_lines)

def extract_eigenvalues_kpoints(directory):
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Initialize the set of eigenvalues
    eigenvalues_matrix = []
    # Navigate to the eigenvalues section
    eigenvalues_section = root.find(".//eigenvalues")
    if eigenvalues_section is not None:
        # Find all k-point <set> elements (assuming they are nested within spin <set> elements)
        kpoint_sets = eigenvalues_section.findall(".//set/set/set")
        if kpoint_sets:
            # Loop over each k-point set
            for kpoint_set in kpoint_sets:
                kpoint_eigenvalues = []
                # Loop over each band at the current k-point
                for r in kpoint_set.findall("./r"):
                    # Extract the energy value, which is the first number in the <r> tag's text
                    energy = float(r.text.split()[0])
                    kpoint_eigenvalues.append(energy)
                # Append the list of eigenvalues for the current k-point to the matrix
                eigenvalues_matrix.append(kpoint_eigenvalues)
        else:
            print("No k-point <set> elements found in the eigenvalues section.")
    else:
        print("Eigenvalues section not found in the XML file.")
    return eigenvalues_matrix

def extract_eigenvalues_lines(directory):
    eigenvalues_matrix = extract_eigenvalues_kpoints(directory)
    transposed_eigenvalues_matrix = transpose_matrix(eigenvalues_matrix)
    return transposed_eigenvalues_matrix

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
