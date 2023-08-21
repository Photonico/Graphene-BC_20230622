#### Extract selected lattice and free energy

# Created by Lu Niu LukeNiu@outlook.com

import os
import xml.etree.ElementTree as ET

def select_lattice(project_folder):
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
