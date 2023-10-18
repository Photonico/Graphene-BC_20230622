##### Data Process : Filter Segment
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915

def filter_segment(lattice_source, distance_source, free_energy_source, lattice_start, lattice_end):
    filtered_data = [(lattice, distance, energy) for lattice, distance, energy in zip(lattice_source, distance_source, free_energy_source) if lattice_start <= lattice <= lattice_end]

    if not filtered_data:
        return [], [], []

    return zip(*filtered_data)
