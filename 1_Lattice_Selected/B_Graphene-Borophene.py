#### Lattice calculate via VASP

# How to use: `nohup python3 lattice_calculate.py &`
# Check the running: `ps aux | grep python3`
# Created by Lu Niu LukeNiu@outlook.com

import numpy as np
import os
import subprocess
import shutil
import time

source_dir      = "Lattice_source" 
distance_bound  = 15.0

for lattice_type in ["Top", "Bridge", "Hollow_1", "Hollow_2"]:

    if lattice_type == "Top":
        lattice_code = "B1"
        lattice_shift = 0
        a_var = 1; z_var = 2

    elif lattice_type == "Bridge":
        lattice_code = "B2"
        lattice_shift = 1/12
        a_var = 1; z_var = 2

    elif lattice_type == "Hollow_1":
        lattice_code = "B3"
        lattice_shift = 1/9
        a_var = 1; z_var = 2

    elif lattice_type == "Hollow_2":
        lattice_code = "B4"
        lattice_shift = 1/6
        a_var = 1; z_var = 2

    dirs_to_walk = []; dest_dir = f"Twin_{lattice_code}_Graphene-Borophene_{lattice_type}_var_{a_var:.3f}_dis_{z_var:.3f}"
    os.makedirs(dest_dir, exist_ok = True)
    for file_name in ["INCAR", "KPOINTS", "POTCAR", "vasp_nci.sh", "vasp_usyd.sh"]:
        shutil.copy(os.path.join(source_dir, file_name), dest_dir)
    a_1 = np.sqrt(3) * 0.5 * a_var
    a_2 = 0.5 * a_var
    with open(os.path.join(dest_dir, "POSCAR"), "w") as f:
        f.write(f"""Graphene-Borophene {lattice_type} with lattice parameter {a_var:.3f} and distance {z_var:.3f}
        1.0
        {a_1:.16f}    {a_2:.16f}    0.0000000000000000
       -{a_1:.16f}    {a_2:.16f}    0.0000000000000000
        0.0000000000000000    0.0000000000000000   {distance_bound:.16f}
        B   C
        8   8
        Direct
        0.6688868879072842  0.0000000000000000  0.1000000000000014
        0.3311121120927183  0.0000000000000000  0.1000000000000014
        0.0000000000000000  0.3311121120927183  0.1000000000000014
        0.0000000000000000  0.6688868879072842  0.1000000000000014
        0.3311121120927183  0.3311121120927183  0.1000000000000014
        0.6688868879072842  0.6688868879072842  0.1000000000000014
        0.6666659999999993  0.3333330000000032  0.0992218869917920
        0.3333330000000032  0.6666659999999993  0.1007781130082108
        {0.1666659999999993+lattice_shift:.16f}  {0.3333330000000032-lattice_shift:.16f}  {0.1000000000000000+z_var/distance_bound:.16f}
        {0.3333330000000032+lattice_shift:.16f}  {0.1666659999999993-lattice_shift:.16f}  {0.1000000000000000+z_var/distance_bound:.16f}
        {0.1666659999999993+lattice_shift:.16f}  {0.8333330000000032-lattice_shift:.16f}  {0.1000000000000000+z_var/distance_bound:.16f}
        {0.3333330000000032+lattice_shift:.16f}  {0.6666659999999993-lattice_shift:.16f}  {0.1000000000000000+z_var/distance_bound:.16f}
        {0.6666659999999993+lattice_shift:.16f}  {0.3333330000000032-lattice_shift:.16f}  {0.1000000000000000+z_var/distance_bound:.16f}
        {0.8333330000000032+lattice_shift:.16f}  {0.1666659999999993-lattice_shift:.16f}  {0.1000000000000000+z_var/distance_bound:.16f}
        {0.6666659999999993+lattice_shift:.16f}  {0.8333330000000032-lattice_shift:.16f}  {0.1000000000000000+z_var/distance_bound:.16f}
        {0.8333330000000032+lattice_shift:.16f}  {0.6666659999999993-lattice_shift:.16f}  {0.1000000000000000+z_var/distance_bound:.16f}""")
    dirs_to_walk.append(dest_dir)

    time.sleep(2)

    for dest_dir in dirs_to_walk:
        if "vasp_nci.sh" in os.listdir(dest_dir):
            print(dest_dir)
            subprocess.run(["bash", "-c", f"cd {dest_dir}; qsub vasp_nci.sh"])
            time.sleep(4)
