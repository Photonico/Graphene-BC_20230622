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

start = 5.171
end = 5.179
step = 0.001

for a_var in np.arange(start, end+step, step):
    dest_dir = f"{dest_dir_base}{a_var:.3f}"
    os.makedirs(dest_dir, exist_ok = True)
    for file_name in ["INCAR", "KPOINTS", "vasp_cluster.sh", "POTCAR"]:
        shutil.copy(os.path.join(source_dir, file_name), dest_dir)
    a_1 = np.sqrt(3) * 0.5 * a_var
    a_2 = 0.5 * a_var
    with open(os.path.join(dest_dir, "POSCAR"), "w") as f:
        f.write(f"""BC3 with lattice parameter {a_var:.3f}
        1.0
        {a_1:.6f}    {a_2:.6f}    0.000000
       -{a_1:.6f}    {a_2:.6f}    0.000000
        0.000000    0.000000    15.00000
        B  C
        2  6
        Direct
        0.3383556788790425  0.6691933804436119  0.2500000000000000
        0.6716854984940485  0.3357841194011583  0.2500000000000000
        0.1639341765360518  0.3209612090164141  0.2500000000000000
        0.3234438254853054  0.1619491780981250  0.2500000000000000
        0.1644514154583912  0.8435078271868911  0.2500000000000000
        0.8455567461896502  0.1614769405085851  0.2500000000000000
        0.6865073017608341  0.8430499579333741  0.2500000000000000
        0.8460613571966716  0.6840733874118388  0.2500000000000000 """)
    dirs_to_walk.append(dest_dir)

time.sleep(2)

for dest_dir in dirs_to_walk:
    if "vasp_cluster.sh" in os.listdir(dest_dir):
        print(dest_dir)
        subprocess.run(["csh", "-c", f"cd {dest_dir}; qsub vasp_cluster.sh"])
        time.sleep(4)
