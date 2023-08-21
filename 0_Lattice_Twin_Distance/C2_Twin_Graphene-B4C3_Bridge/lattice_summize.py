#### Lattice free energy

# Created by Lu Niu LukeNiu@outlook.com

import os
import xml.etree.ElementTree as ET

def compute_average(data_lines):
    """Define the function to compute the average of the last value in each line."""
    total = 0
    for line in data_lines:
        values = line.split()       # Split the line into individual values
        total += float(values[-1])  # Add the last value to the total
        print(line)
    return total / len(data_lines)  # Return the average

dest_dir_base = "Lattice_" # Base name for destination directories
result_file = "lattice.dat"

distance_bound = 15

dirs_to_walk = [dir for dir in os.listdir() if dir.startswith(dest_dir_base)]
results = []

for dest_dir in dirs_to_walk:
    file_name_xml = os.path.join(dest_dir, "vasprun.xml")
    file_name_poscar = os.path.join(dest_dir, "POSCAR")
    file_name_contcar = os.path.join(dest_dir, "CONTCAR")
    
    if os.path.isfile(file_name_xml) and os.path.isfile(file_name_poscar):
        try:
            # Extract e_fr_energy from vasprun.xml
            tree = ET.parse(file_name_xml)
            root = tree.getroot()
            e_fr_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)

            # Extract a_var and z_var from POSCAR
            with open(file_name_poscar, "r") as poscar_file:
                first_line = poscar_file.readline()
                a_var = float(first_line.split("lattice parameter")[1].split()[0])
            with open(file_name_contcar, "r") as contcar_file:
                first_line = contcar_file.readline()
                lines = contcar_file.readlines()
                bottom_set = lines[7:14]    # Extract lines for the first set of values
                bottom_avg = compute_average(bottom_set)
                top_set = lines[14:21]      # Extract lines for the second set of values
                top_avg = compute_average(top_set)
                z_var = (top_avg - bottom_avg)*distance_bound

            # Add the result to the list
            results.append((a_var, z_var, e_fr_energy))

        except ET.ParseError:
            print(f"Error parsing XML in directory: {dest_dir}")
            continue

# Sort the results by a_var (the first element of the tuple)
results.sort(key=lambda x: x[0])

# Now write the sorted results to the file
with open(result_file, "w") as f:
    f.write(f"Lattice\t Distance\t Free energy\n")
    for a_var, z_var, e_fr_energy in results:
        f.write(f"{a_var:.6f}\t{z_var:.6f}\t{e_fr_energy}\n")
