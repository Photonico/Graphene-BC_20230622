#### Read kpoints verse free energy from kpoints.dat
# pylint: disable = C0103, C0114, C0116

def read_kpoints(kpoints_path):
    # Initialize the lists for kpoints constant and free energy
    kpoints, free_energy = [], []

    with open(kpoints_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for line in lines:
            split_line = line.strip().split()
            kpoints.append(float(split_line[0]))
            free_energy.append(float(split_line[1]))

    return kpoints, free_energy
