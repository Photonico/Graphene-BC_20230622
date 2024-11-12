#### Lattice calculate via VASP

# How to use: `nohup python3 lattice_calculate.py &`
# Check the running: `ps aux | grep python3`
# Created by Lu Niu LukeNiu@outlook.com

import numpy as np
import os
import subprocess
import shutil
import time

source_dir = "lattice_source"
dest_dir_base   = "Lattice_a"       # Base name for destination directories
dirs_to_walk    = []

start = 2.400
end = 2.500
step = 0.005

arr = np.arange(start, end+step, step)
arr_ext = np.append(arr, 2.467)

for a_var in arr_ext:
    dest_dir = f"{dest_dir_base}{a_var:.3f}"
    os.makedirs(dest_dir, exist_ok = True)
    for file_name in ["INCAR", "KPOINTS", "vasp_gadi.sh", "POTCAR"]:
        shutil.copy(os.path.join(source_dir, file_name), dest_dir)
    a_1 = np.sqrt(3) * a_var * 0.5
    a_2 = a_var * 0.5
    with open(os.path.join(dest_dir, "POSCAR"), "w") as f:
        f.write(f"""Graphene with lattice parameter {a_var:.3f}
        1.0
        {a_1:.6f}    {a_2:.6f}    0.000000
       -{a_1:.6f}    {a_2:.6f}    0.000000
        0.000000    0.000000    15.00000
        C
        2
        Direct
        0.333333    0.666666    0.100000
        0.666666    0.333333    0.100000""")
    dirs_to_walk.append(dest_dir)

time.sleep(2)

for dest_dir in dirs_to_walk:
    if "vasp_gadi.sh" in os.listdir(dest_dir):
        print(dest_dir)
        subprocess.run(["bash", "-c", f"cd {dest_dir}; qsub vasp_gadi.sh"])
        time.sleep(4)
