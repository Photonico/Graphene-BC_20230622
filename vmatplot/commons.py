#### Common codes
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import xml.etree.ElementTree as ET
import os

from typing import Tuple, Union
import numpy as np

def identify_algorithm(directory):
    """
    Identify and print the algorithm used in all INCAR files within the specified folder,
    ignoring spaces and checking if settings are commented out.
    :param directory: Path to the target folder
    """
    file_name = "INCAR"
    file_path = os.path.join(directory, file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        # Extract all active lines considering lines that start with space followed by "#"
        active_lines = [line.upper().replace(" ", "") for line in file if not line.lstrip().startswith('#')]

        # Initialize flags for detected algorithms
        hse06_flag = any("LHFCALC=.TRUE." in line for line in active_lines)
        hf_screen_flag = any("HFSCREEN" in line for line in active_lines)
        gga_pe_flag = any("GGA=PE" in line for line in active_lines)

        # Determine the algorithm based on detected flags
        if all(not flag for flag in [gga_pe_flag, hse06_flag, hf_screen_flag]):
            # Equals to `if not gga_pe_flag and not hse06_flag and not hf_screen_flag`
            # Equals to `if not (gga_pe_flag or hse06_flag or hf_screen_flag)`
            # Default to GGA-PBE if no specific flags are detected
            algorithm = "GGA-PBE"
            # print("0")
        elif gga_pe_flag and not (hse06_flag and hf_screen_flag):
            algorithm = "GGA-PBE"
            # print("1")
        elif hse06_flag and hf_screen_flag:
            algorithm = "HSE06"
            # print("2")
        else:
            algorithm = "Uncertain or other algorithms"

    return algorithm

def analyze_vasprun(directory_path):
    ## Help information
    if directory_path == "help":
        print("Please use this function on the project folder.")
        return []

    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    # Check if the vasprun.xml file exists in the given directory
    if not os.path.isfile(file_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")
        return

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

def get_atoms_count(directory):
    """
    Extracts the total number of atoms from a VASP vasprun.xml file.

    Args:
    directory (str): The directory path that contains the VASP vasprun.xml file.

    Returns:
    int: The total number of atoms in the calculation.
    """
    # Construct the path to the vasprun.xml file and parse it
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the atominfo section and extract the total number of atoms
    atominfo_section = root.find(".//atominfo/atoms")
    if atominfo_section is not None:
        return int(atominfo_section.text)
    else:
        print("Atominfo section not found in the XML file.")
        return None

def get_elements(directory_path):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    # Check if the vasprun.xml file exists in the given directory
    if not os.path.isfile(file_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")
        return

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

def extract_fermi(directory):
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    ## Extract eigen, occupancy number
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for dos in root.findall("./calculation/dos"):
            comment = dos.get("comment")
            if comment == "kpoints_opt":
                for i in dos.findall("i"):
                    if "name" in i.attrib:
                        if i.attrib["name"] == "efermi":
                            fermi_energy = float(i.text)
                            return fermi_energy
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        for i in root.iter("i"):
            if "name" in i.attrib:
                if i.attrib["name"] == "efermi":
                    fermi_energy = float(i.text)
                    return fermi_energy
    raise ValueError("Fermi energy not found in vasprun.xml")

def extract_part(ind_values, dep_values, start_value=None, end_value=None):
    # Ensure ind_values and dep_values are numpy arrays
    ind_values = np.asarray(ind_values)
    dep_values = np.asarray(dep_values)
    # Condition handling:
    # If both start_value and end_value are None, return the original data without processing.
    if start_value is None and end_value is None:
        return (ind_values, dep_values)
    # If only start_value is provided (not None), slice from start_value to the end.
    elif start_value is not None and end_value is None:
        condition = ind_values >= start_value
    # If only end_value is provided (not None), slice from the beginning to end_value.
    elif start_value is None and end_value is not None:
        condition = ind_values <= end_value
    # If both start_value and end_value are provided, slice between start_value and end_value.
    else:
        condition = (ind_values >= start_value) & (ind_values <= end_value)
    # Filtering data based on the condition
    ind_values_filtered = ind_values[condition]
    dep_values_filtered = dep_values[condition]
    return (ind_values_filtered, dep_values_filtered)

def check_range_type(data: Union[Tuple, Tuple[Tuple, ...], int, float]) -> str:
    """
    Determine the type of the provided range data.
    Args:
        data (Union[Tuple, Tuple[Tuple, ...], int, float]): The range data to check.
        It can be a simple number, a tuple with one or more elements, or nested tuples.
    Returns:
        str: A description of whether the input is a "Simple end",
             "Double ends", "Simple range", "Simple range with a rate", or "Double ranges".
    """
    # Check if the input is a single number (not a tuple)
    if isinstance(data, (int, float)):
        return "Simple end"
    # Check if the input is a tuple with one element
    elif isinstance(data, tuple) and len(data) == 1:
        if isinstance(data[0], tuple):
            return "Simple range"
        else:
            return "Simple end"
    # Check if the input is a tuple with two elements
    elif isinstance(data, tuple) and len(data) == 2:
        if isinstance(data[0], tuple) and isinstance(data[1], tuple):
            return "Double ranges"
        elif isinstance(data[0], tuple) and isinstance(data[1], (int, float)):
            return "Simple range with a rate"
        elif isinstance(data[0], (int, float)) and isinstance(data[1], (int, float)):
            return "Double ends"
    return "Unknown type"

def process_boundary(boundary, default=(None, None)):
    # Enhanced to handle single values as well as tuples/lists
    # If boundary is None or empty, return the default
    if not boundary:
        return default
    # If boundary is a single value (not a container), treat it as the end value
    if isinstance(boundary, (int, float)):
        return (None, boundary)
    # If boundary is a container with a single item, unpack it
    if isinstance(boundary, (list, tuple)) and len(boundary) == 1:
        return (None, boundary[0])
    # If boundary is a container with two items, return them as start and end
    elif isinstance(boundary, (list, tuple)) and len(boundary) == 2:
        return (boundary[0], boundary[1])
    # In case boundary doesn't match any expected pattern, return default
    else:
        return default

def process_boundaries_rescaling(boundary):
    boundary_type = check_range_type(boundary)
    scale_flag = False
    source_start, source_end, scaled_start, scaled_end = None, None, None, None

    if boundary_type == "Simple end":
        source_range = (None, boundary)
        source_start, source_end = process_boundary(source_range)
    elif boundary_type == "Simple range":
        source_range = boundary[0]
        source_start, source_end = process_boundary(source_range)
    elif boundary_type == "Double ends":
        scale_flag = True
        source_range = (None, boundary[0])
        scaled_range = (None, boundary[1])
        source_start, source_end = process_boundary(source_range)
        scaled_start, scaled_end = process_boundary(scaled_range)
    elif boundary_type == "Simple range with a rate":
        scale_flag = True
        source_range = boundary[0]
        scaled_range = tuple(bounds * boundary[1] for bounds in boundary[0])
        source_start, source_end = process_boundary(source_range)
        scaled_start, scaled_end = process_boundary(scaled_range)
    elif boundary_type == "Double ranges":
        scale_flag = True
        source_range = boundary[0]
        scaled_range = boundary[1]
        source_start, source_end = process_boundary(source_range)
        scaled_start, scaled_end = process_boundary(scaled_range)
    return scale_flag, source_start, source_end, scaled_start, scaled_end
