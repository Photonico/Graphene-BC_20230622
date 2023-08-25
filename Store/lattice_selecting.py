#### Extract selected lattice and free energy

# Created by Lu Niu LukeNiu@outlook.com

import os
import xml.etree.ElementTree as ET

distance_bound = 15

def compute_average(data_lines):
    """Define the function to compute the average of the last value in each line."""
    total = 0
    for line in data_lines:
        values = line.split()       # Split the line into individual values
        total += float(values[-1])  # Add the last value to the total
        # print(line)
    return total / len(data_lines)  # Return the average

def lattice_select(project_folder):
    xml_path = os.path.join(project_folder, "vasprun.xml")
    poscar_path = os.path.join(project_folder, "POSCAR")
    if os.path.isfile(xml_path) and os.path.isfile(poscar_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            e_fr_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)
            with open(poscar_path, "r") as poscar_file:
                first_line = poscar_file.readline()
                a_var = float(first_line.split()[-1])
            return a_var, e_fr_energy
        except Exception as e:
            print("Error parsing files:", e)
            return None

def lattice_select_distance(project_folder, layer_number, total_atom_number):
    xml_path = os.path.join(project_folder, "vasprun.xml")
    poscar_path = os.path.join(project_folder, "POSCAR")
    contcar_path = os.path.join(project_folder, "CONTCAR")
    atom_number = int(total_atom_number / 2)

    if os.path.isfile(xml_path) and os.path.isfile(poscar_path) and os.path.isfile(contcar_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            e_fr_energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)
            with open(poscar_path, "r") as poscar_file:
                first_line = poscar_file.readline()
                lattice_type = first_line.split()[0]
                a_var = float(first_line.split("lattice parameter")[1].split()[0])
                lines = poscar_file.readlines()
                fifth_line = lines[3]
                distance_bound  = float(fifth_line.split()[-1])
            if layer_number == 2:
                with open(contcar_path, "r") as contcar_file:
                    first_line = contcar_file.readline()
                    lines = contcar_file.readlines()
                    bottom_set = lines[7: 7 + atom_number]                      # Extract lines for the first set of values
                    bottom_avg = compute_average(bottom_set)
                    top_set = lines[7 + atom_number: 7 + 2*atom_number]         # Extract lines for the second set of values
                    top_avg = compute_average(top_set)
                    z_var = (top_avg - bottom_avg) * distance_bound  
                return lattice_type, a_var, z_var, e_fr_energy
            elif layer_number == 1:
                return lattice_type, a_var, distance_bound, e_fr_energy
            else:
                print("Error: Unsupported layer_number provided!")
                return None

        except Exception as e:
            print("Error parsing files:", e)
            return None
