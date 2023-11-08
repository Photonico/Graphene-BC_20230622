#### Kpoints free energy

# Created by Lu Niu LukeNiu@outlook.com

import os
import xml.etree.ElementTree as ET

dest_dir_base = "Kpoints_"  # Base name for destination directories
result_file = "Kpoints.dat"

dirs_to_walk = [dir for dir in os.listdir() if dir.startswith(dest_dir_base)]
results = []

for dest_dir in dirs_to_walk:
    file_name_xml = os.path.join(dest_dir, "vasprun.xml")
    file_name_kpoints = os.path.join(dest_dir, "KPOINTS")
    if os.path.isfile(file_name_xml) and os.path.isfile(file_name_kpoints):

        # Extract e_fr_energy from vasprun.xml
        tree = ET.parse(file_name_xml)
        root = tree.getroot()
        e_fr_energy =float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)

        # Extract k_var from 
        with open(file_name_kpoints, "r") as kpoints_file:
            first_line = kpoints_file.readline()
            k_var = int(first_line.split()[-1]) 

        # Add the result to the list
        results.append((k_var, e_fr_energy))

# Sort the results by k_var (the first element of the tuple)
results.sort(key=lambda x: x[0])

# Now write the sorted results to the file
with open(result_file, "w") as f:
    f.write(f"Kpoints\t Free energy\n")
    for k_var, e_fr_energy in results:
        f.write(f"{k_var}\t{e_fr_energy}\n")
