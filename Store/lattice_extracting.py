#### Read lattice verse free energy from lattice.dat
# pylint: disable = C0103, C0114, C0116

def read_lattice(lattice_path):
    # Initialize the lists for lattice constant and free energy
    lattice, free_energy = [], []

    with open(lattice_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for line in lines:
            split_line = line.strip().split()
            lattice.append(float(split_line[0]))
            free_energy.append(float(split_line[1]))

    return lattice, free_energy

def read_lattice_distance(lattice_path):
    # Initialize the lists for lattice constant and free energy
    lattice, distance, free_energy = [], [], []

    with open(lattice_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for line in lines:
            split_line = line.strip().split()
            lattice.append(float(split_line[0]))
            distance.append(float(split_line[1]))
            free_energy.append(float(split_line[2]))

    return lattice, distance, free_energy
