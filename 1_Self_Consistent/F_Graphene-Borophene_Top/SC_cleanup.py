#### SC cleanup

import os

#%%
# List of files to keep
files_to_keep = [
    "INCAR", "KPOINTS", "POSCAR", "POTCAR", "vasp_gadi.sh", "SC_cleanup.py"
]

def cleanup(files):
    """Clean VASP works"""
    # Files input
    # Start from the current directory and search files and directories recursively
    for dirpath, dirnames, filenames in os.walk(os.getcwd()):
        if "lattice_source" in dirnames:
            dirnames.remove("lattice_source")
        for filename in filenames:
            if filename not in files:
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                except (FileNotFoundError, PermissionError) as e:
                    print(f"Error deleting {file_path}: {e}")
    print("Cleanup complete.")

cleanup(files_to_keep)

# %%
