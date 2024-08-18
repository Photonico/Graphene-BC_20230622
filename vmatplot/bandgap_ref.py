# Extract bandgap from bandstructure calculation
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import os
from pymatgen.io.vasp import Vasprun

def extract_bandgap_ref(directory):
    # Construct the full path to the vasprun.xml file
    vasprun_path = os.path.join(directory, "vasprun.xml")

    # Check if the file exists
    if not os.path.isfile(vasprun_path):
        print(f"vasprun.xml file not found in directory: {directory}")
        return None, None

    # Load the vasprun.xml file
    try:
        vasprun = Vasprun(vasprun_path, parse_projected_eigen=True)
    except Exception as e:
        print(f"Error reading vasprun.xml: {e}")
        return None, None

    # Get the band structure
    try:
        band_structure = vasprun.get_band_structure()
    except Exception as e:
        print(f"Error extracting band structure: {e}")
        return None, None

    # Get the band gap
    try:
        band_gap_details = band_structure.get_band_gap()
        material_is_metallic = band_structure.is_metal()
    except Exception as e:
        print(f"Error extracting band gap: {e}")
        return None, None

    if material_is_metallic:
        print("The material is metallic.")
        print(f"Band gap: {band_gap_details['energy']} eV")
    else:
        print(f"Band gap: {band_gap_details['energy']} eV")
        print(f"Direct gap: {band_gap_details['direct']}")
        print(f"Transition: {band_gap_details['transition']}")

    return band_gap_details, material_is_metallic

# Example usage
directory_path = "path/to/your/calculation/directory"
band_gap_info, is_metal = extract_bandgap_ref(directory_path)
if band_gap_info is not None:
    if is_metal:
        print("The material is metallic.")
    else:
        print(f"Band gap: {band_gap_info['energy']} eV, Direct: {band_gap_info['direct']}, Transition: {band_gap_info['transition']}")
else:
    print("Failed to extract band gap information.")
