# Extract bandgap from bandstructure calculation
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import os
import xml.etree.ElementTree as ET
from pymatgen.io.vasp import Vasprun

def extract_bandgap_ref(project_folder):
    vasprun_path = os.path.join(project_folder, "vasprun.xml")

    if not os.path.exists(vasprun_path):
        raise FileNotFoundError(f"{vasprun_path} does not exist.")

    vasprun = Vasprun(vasprun_path)

    band_structure = vasprun.get_band_structure()
    bandgap = band_structure.get_band_gap()["energy"]

    vbm = band_structure.get_vbm()["energy"]  # Highest Occupied Molecular Orbital (HOMO)
    cbm = band_structure.get_cbm()["energy"]  # Lowest Unoccupied Molecular Orbital (LUMO)

    return bandgap, vbm, cbm


def extract_bandgap_vasprun(project_folder):
    # Define the path to the vasprun.xml file
    vasprun_path = os.path.join(project_folder, "vasprun.xml")

    # Check if the file exists
    if not os.path.exists(vasprun_path):
        raise FileNotFoundError(f"{vasprun_path} does not exist.")

    # Parse the vasprun.xml file
    tree = ET.parse(vasprun_path)
    root = tree.getroot()

    # Locate Fermi energy (efermi) if available
    efermi_tag = root.find(".//i[@name='efermi']")
    efermi = float(efermi_tag.text) if efermi_tag is not None else 0.0

    # Find eigenvalues
    eigenvalues_spin_1 = []
    eigenvalues_spin_2 = []

    for eigenvalues_set in root.findall(".//eigenvalues/array/set/set"):
        for spin_index, spin in enumerate(eigenvalues_set.findall("./set/r")):
            # Extract the eigenvalue (first entry)
            energy = float(spin.text.split()[0])
            if spin_index == 0:
                eigenvalues_spin_1.append(energy)
            else:
                eigenvalues_spin_2.append(energy)

    # Sort eigenvalues
    eigenvalues_spin_1 = sorted(eigenvalues_spin_1)
    eigenvalues_spin_2 = sorted(eigenvalues_spin_2) if eigenvalues_spin_2 else eigenvalues_spin_1

    # Find HOMO and LUMO for spin channel 1
    homo_spin_1 = max([e for e in eigenvalues_spin_1 if e <= efermi])
    lumo_candidates_spin_1 = [e for e in eigenvalues_spin_1 if e > efermi]
    lumo_spin_1 = min(lumo_candidates_spin_1) if lumo_candidates_spin_1 else None

    # Handle spin channel 2 if it exists
    if eigenvalues_spin_2:
        homo_spin_2 = max([e for e in eigenvalues_spin_2 if e <= efermi])
        lumo_candidates_spin_2 = [e for e in eigenvalues_spin_2 if e > efermi]
        lumo_spin_2 = min(lumo_candidates_spin_2) if lumo_candidates_spin_2 else None

        # Combine spin channels for overall HOMO and LUMO
        homo = max(homo_spin_1, homo_spin_2)
        lumo = min(lumo_spin_1, lumo_spin_2) if lumo_spin_1 and lumo_spin_2 else lumo_spin_1 or lumo_spin_2
    else:
        homo = homo_spin_1
        lumo = lumo_spin_1

    # If no LUMO is found, set bandgap to zero (metallic case)
    if lumo is None:
        bandgap = 0.0
        lumo = homo  # Set LUMO to the same value as HOMO for metallic systems
    else:
        bandgap = lumo - homo

    return bandgap, homo, lumo

def extract_bandgap_outcar(directory):
    # Check if the user asked for help
    if directory == "help":
        print("Please use this function to OUTCAR of the bandstructure calculation.")
        return "Help provided."

    outcar_path = os.path.join(directory, "OUTCAR")
    if not os.path.isfile(outcar_path):
        print("OUTCAR file not found in the provided directory.")
        return None

    with open(outcar_path, "r", encoding="utf-8") as outcar_file:
        content = outcar_file.readlines()
    # Get HOMO, LUMO, and NKPT
    homo = None
    lumo = None
    nkpt = None

    for line in content:
        if "NELECT" in line:
            homo = int(float(line.split()[2]) / 2)
            lumo = homo + 1
        if "NKPTS" in line:
            nkpt = int(line.split()[3])

    # Extract energies for HOMO and LUMO
    homo_energies = [float(line.split()[1]) for i, line in enumerate(content) if f"     {homo}     " in line]
    lumo_energies = [float(line.split()[1]) for i, line in enumerate(content) if f"     {lumo}     " in line]

    # Get maximum HOMO and minimum LUMO energy considering nkpt values
    homo = sorted(homo_energies[:nkpt])[-1]
    lumo = sorted(lumo_energies[:nkpt])[0]
    bandgap =  lumo - homo

    return bandgap, homo, lumo
