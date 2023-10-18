#### Lattice free energy

# Created by Lu Niu LukeNiu@outlook.com

import os
import xml.etree.ElementTree as ET

dest_dir_base = "Lattice_a" # Base name for destination directories
result_file = "lattice.dat"

dirs_to_walk = [dir for dir in os.listdir() if dir.startswith(dest_dir_base)]
results = []

for dest_dir in dirs_to_walk:
    file_name_xml = os.path.join(dest_dir, "vasprun.xml")
    file_name_poscar = os.path.join(dest_dir, "POSCAR")
    if os.path.isfile(file_name_xml) and os.path.isfile(file_name_poscar):
        
        # Extract e_fr_energy from vasprun.xml
        tree = ET.parse(file_name_xml)
        root = tree.getroot()
        e_fr_energy =float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)

        # Extract a_var from POSCAR
        with open(file_name_poscar, "r") as poscar_file:
            first_line = poscar_file.readline()
            a_var = float(first_line.split()[-1])  # assuming a_var is the last part on the first line

        # Add the result to the list
        results.append((a_var, e_fr_energy))

# Sort the results by a_var (the first element of the tuple)
results.sort(key=lambda x: x[0])

# Now write the sorted results to the file
with open(result_file, "w") as f:
    f.write(f"Lattice\t Free energy\n")
    for a_var, e_fr_energy in results:
        f.write(f"{a_var:.3f}\t{e_fr_energy}\n")
