# Bandstructure

import os

def extract_high_symlines(directory):
    file = open(os.path.join(directory, "KPOINTS"), "r")
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

extract_high_symlines("3_Bandstructure_PBE/A_BC3/KPOINTS")

def fermi_energy_extracting(directory):
    file = open(os.path.join(directory, "OUTCAR"), "r")
    OUTCAR = file.readlines()
    file.close()
    for i in range(0, len(OUTCAR)):
        if("Fermi energy" in OUTCAR[i]):
            efermi = OUTCAR[i].split()[2]
            print(f"The Fermi energy is: {efermi} eV")
            return efermi

print(fermi_energy_extracting("3_Bandstructure_PBE/A_BC3"))
