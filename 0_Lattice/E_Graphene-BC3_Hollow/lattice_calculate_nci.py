#### Lattice calculate via VASP

# How to use: `nohup python3 lattice_calculate.py &`
# Check the running: `ps aux | grep python3`
# Created by Lu Niu LukeNiu@outlook.com

import numpy as np
import os
import subprocess
import shutil
import time

lattice_type = "Hollow"
lattice_shift = 1/6

source_dir = "../lattice_source"
dirs_to_walk    = []

distance_bound = 15.0

z_var = 3.383

a_step = 0.002
a_start = 5.041
a_end = 5.048

# for a_var in np.arange(a_start, a_end, a_step):
for a_var in [5.005,5.015,5.075]:
    dest_dir = f"Lattice_a{a_var:.3f}_d{z_var:.3f}"
    os.makedirs(dest_dir, exist_ok = True)
    for file_name in ["INCAR", "KPOINTS", "POTCAR", "vasp_gadi.sh", "vasp_cluster.sh"]:
        shutil.copy(os.path.join(source_dir, file_name), dest_dir)
    a_1 = np.sqrt(3) * 0.5 * a_var
    a_2 = 0.5 * a_var
    with open(os.path.join(dest_dir, "POSCAR"), "w") as f:
        f.write(f"""Graphene-BC3 {lattice_type} with lattice parameter {a_var:.3f} and distance {z_var:.3f}
        1.0
        {a_1:.16f}    {a_2:.16f}    0.0000000000000000
       -{a_1:.16f}    {a_2:.16f}    0.0000000000000000
        0.0000000000000000    0.0000000000000000   {distance_bound:.16f}
        B   C
        2   14
        Direct
        0.3383303433200950  0.6691643220287062  0.1000000000000000
        0.6716552153790474  0.3358414604455504  0.1000000000000000
        0.1637394633814182  0.3199996222090675  0.1000000000000000
        0.3225021842022073  0.1612639661328572  0.1000000000000000
        0.1637643005245266  0.8437638408621737  0.1000000000000000
        0.8462386341522503  0.1612368417574501  0.1000000000000000
        0.6874985035840169  0.8437300034210438  0.1000000000000000
        0.8462673554564475  0.6849959431431500  0.1000000000000000
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
        subprocess.run(["bash", "-c", f"cd {dest_dir}; qsub vasp_gadi.sh"])
        time.sleep(4)
