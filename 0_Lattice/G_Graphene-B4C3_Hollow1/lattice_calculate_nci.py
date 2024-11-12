#### Lattice calculate via VASP

# How to use: `nohup python3 lattice_calculate.py &`
# Check the running: `ps aux | grep python3`
# Created by Lu Niu LukeNiu@outlook.com

import numpy as np
import os
import subprocess
import shutil
import time

lattice_type = "Hollow 1"
lattice_shift = -1/18

source_dir = "../lattice_source"
dirs_to_walk    = []

distance_bound = 15.0

z_var = 3.559

a_step = 0.002
a_start = 4.840
a_end = 4.858

# for a_var in np.arange(a_start, a_end, a_step):
for a_var in [4.805, 4.815, 4.825, 4.875]:
    dest_dir = f"Lattice_a{a_var:.3f}_d{z_var:.3f}"
    os.makedirs(dest_dir, exist_ok = True)
    for file_name in ["INCAR", "KPOINTS", "POTCAR", "vasp_gadi.sh", "vasp_cluster.sh"]:
        shutil.copy(os.path.join(source_dir, file_name), dest_dir)
    a_1 = np.sqrt(3) * 0.5 * a_var
    a_2 = 0.5 * a_var
    with open(os.path.join(dest_dir, "POSCAR"), "w") as f:
        f.write(f"""Graphene-B4C3 {lattice_type} with lattice parameter {a_var:.3f} and distance {z_var:.3f}
        1.0
        {a_1:.16f}    {a_2:.16f}    0.0000000000000000
       -{a_1:.16f}    {a_2:.16f}    0.0000000000000000
        0.0000000000000000    0.0000000000000000   {distance_bound:.16f}
        B   C
        4   11
        Direct
        0.0041536073678543  0.0026721015370299  0.1203413379052876
        0.3569533512947416  0.0028550305604824  0.0974889342262841
        0.0042385768318738  0.3555069917146056  0.0974925637235273
        0.6514229529441167  0.6500655703952489  0.0974768786615599
        0.3280864628224336  0.3267193752228721  0.1003050777105869
        0.6802777482657802  0.0027808524732649  0.1002756210064604
        0.0041653004731970  0.6788380780964971  0.1002655867662980
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
