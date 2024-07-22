#### Lattice calculate via VASP

# How to use: `nohup python3 lattice_calculate.py &`
# Check the running: `ps aux | grep python3`
# Created by Lu Niu LukeNiu@outlook.com

import numpy as np
import os
import subprocess
import shutil
import time

source_dir = "../lattice_source"
dest_dir_base   = "Lattice_var_"    # Base name for destination directories
dirs_to_walk    = []

start = 4.691
end = 4.699
step = 0.001

for a_var in np.arange(start, end+step, step):
    dest_dir = f"{dest_dir_base}{a_var:.3f}"
    os.makedirs(dest_dir, exist_ok = True)
    for file_name in ["INCAR", "KPOINTS", "vasp_cluster.sh", "POTCAR"]:
        shutil.copy(os.path.join(source_dir, file_name), dest_dir)
    a_1 = np.sqrt(3) * 0.5 * a_var
    a_2 = 0.5 * a_var
    with open(os.path.join(dest_dir, "POSCAR"), "w") as f:
        f.write(f"""B4C3 with lattice parameter {a_var:.3f}
        1.0
        {a_1:.6f}    {a_2:.6f}    0.000000
       -{a_1:.6f}    {a_2:.6f}    0.000000
        0.000000    0.000000    15.00000
        B   C
        4   3
        Direct
        0.012000    0.008000    0.113646
        0.335033    0.019000    0.100000
        0.018000    0.328933    0.100000
        0.664666    0.670006    0.100000
        0.329933    0.329833    0.100000
        0.665666    0.004000    0.100000
        0.004000    0.659666    0.100000""")
    dirs_to_walk.append(dest_dir)

time.sleep(2)

for dest_dir in dirs_to_walk:
    if "vasp_cluster.sh" in os.listdir(dest_dir):
        print(dest_dir)
        subprocess.run(["csh", "-c", f"cd {dest_dir}; qsub vasp_cluster.sh"])
        time.sleep(4)
