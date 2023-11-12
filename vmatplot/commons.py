#### Common codes
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import xml.etree.ElementTree as ET
import os

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
    for i in root.iter("i"):
        if "name" in i.attrib:
            if i.attrib["name"] == "efermi":
                fermi_energy = float(i.text)
                return fermi_energy
    raise ValueError("Fermi energy not found in vasprun.xml")
