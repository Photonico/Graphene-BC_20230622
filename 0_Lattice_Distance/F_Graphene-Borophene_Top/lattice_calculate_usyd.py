#### Lattice calculate via VASP

# How to use: `nohup python3 lattice_calculate.py &`
# Check the running: `ps aux | grep python3`
# Created by Lu Niu LukeNiu@outlook.com

import numpy as np
import os
import subprocess
import shutil
import time

lattice_type = "Top"
lattice_shift = 0

source_dir = "../lattice_source"
dirs_to_walk    = []

distance_bound = 15.0

a_start = 4.9
a_end = 5.1
a_step = 0.05

z_start = 3.3
z_end = 3.5
z_step = 0.05

# a_start = 4.9; a_end = 4.9; a_step = 0.1; z_start = 3.3; z_end = 3.3; z_step = 0.1; 

for z_var in np.arange(z_start, z_end+z_step, z_step):
    for a_var in np.arange(a_start, a_end+a_step, a_step):
        dest_dir = f"Lattice_a{a_var:.3f}_d{z_var:.3f}"
        os.makedirs(dest_dir, exist_ok = True)
        for file_name in ["INCAR", "KPOINTS", "POTCAR", "vasp_gadi.sh", "vasp_cluster.sh"]:
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
    if "vasp_cluster.sh" in os.listdir(dest_dir):
        print(dest_dir)
        subprocess.run(["csh", "-c", f"cd {dest_dir}; qsub vasp_cluster.sh"])
        time.sleep(4)
