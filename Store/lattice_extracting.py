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

def read_lattice_select(lattice_path, times, divisor):
    # Initialize the lists for lattice constant and free energy
    lattice, free_energy = [], []

    with open(lattice_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for line in lines:
            split_line = line.strip().split()
            lattice_value = float(split_line[0])

            # Check if the value after the decimal point is 0
            if lattice_value * times % divisor == 0:
                lattice.append(lattice_value)
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

def read_lattice_distance_select(lattice_path, times, divisor):
    # Initialize the lists for lattice constant and free energy
    lattice, distance, free_energy = [], [], []

    with open(lattice_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for line in lines:
            split_line = line.strip().split()
            lattice_value = float(split_line[0])
            if lattice_value * times % divisor == 0:
                lattice.append(lattice_value)
                distance.append(float(split_line[1]))
                free_energy.append(float(split_line[2]))

    return lattice, distance, free_energy
