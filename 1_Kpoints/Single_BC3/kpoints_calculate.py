#### Kpoints calculate via VASP

# How to use: `nohup python3 Kpoints_calculate.py &`
# Check the running: `ps aux | grep python3`
# Created by Lu Niu LukeNiu@outlook.com

import numpy as np
import os
import subprocess
import shutil
import textwrap
import time

source_dir      = "Kpoints_source"  # Source directory containing INCAR, KPOINTS, POTCAR, and vasp_job.sh
dest_dir_base   = "Kpoints_var_"    # Base name for destination directories
dirs_to_walk    = []

start = 1
end = 20
step = 1

for k_var in np.arange(start, end+step, step):
    dest_dir = f"{dest_dir_base}{k_var}"
    os.makedirs(dest_dir, exist_ok = True)
    for file_name in ["INCAR", "POSCAR", "vasp_job.sh", "POTCAR"]:
        shutil.copy(os.path.join(source_dir, file_name), dest_dir)
    with open(os.path.join(dest_dir, "KPOINTS"), "w") as f:
        f.write(textwrap.dedent(f"""
            k-points {k_var}
            0
            Gamma
            {k_var}    {k_var}    1
            0    0    0

        """).strip())
    dirs_to_walk.append(dest_dir)

time.sleep(2)

# for dest_dir in dirs_to_walk:
#     if "vasp_job.sh" in os.listdir(dest_dir):
#         print(dest_dir)
#         subprocess.run(["csh", "-c", f"cd {dest_dir}; qsub vasp_job.sh"])
#         time.sleep(2)
