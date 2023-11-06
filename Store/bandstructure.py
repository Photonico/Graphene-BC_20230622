#### Bandstructure
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt

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

def extract_fermi(directory):
    with open(os.path.join(directory, "OUTCAR"), "r", encoding="utf-8") as file:
        for _, line in enumerate(file):
            if "Fermi energy" in line:
                efermi = line.split()[2]
                print(f"The Fermi energy is: {efermi} eV")
                return float(efermi)

def extract_klist(directory):
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

#%% Testing space

bswork = "/home/lu/Repos/Graphene-BC 2023/3_Bandstructure_PBE/G_Graphene-B4C3_Top"

kpoints = extract_high_symlines(bswork)
fermi = extract_fermi(bswork)
klist = extract_klist(bswork)
hs = extract_high_symlines(bswork)
print(hs)

#%% Bandstructure workspace

# def plot_bandstructure(bswork):

# plot_bandstructure(bswork)
#%%
