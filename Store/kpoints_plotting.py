#### Kpoints plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import ScalarFormatter
from Store.output import canvas_setting, color_sampling
from Store.kpoints import read_kpoints_free_energy

def plot_kpoints_free_energy(matter, source_data=None, direction="Total", kpoints_start=None, kpoints_end=None, color_family="blue"):
    help_info = "Usage: read_kpoints_free_energy(data_path)\n" + \
                "data_path: Path to the data file containing lattice and free energy values.\n"

    # Check if the user asked for help
    if (matter == "help") and (source_data is None):
        print(help_info)
        return

    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color calling
    colors = color_sampling(color_family)

    # Data input
    data_input = read_kpoints_free_energy(source_data)
    tot_kpoints = data_input[0]
    sep_kpoints = data_input[1]
    x_kpoints, y_kpoints, z_kpoints = [], [], []
    for coord in sep_kpoints:
        x_kpoints.append(coord[0])
        y_kpoints.append(coord[1])
        z_kpoints.append(coord[2])
    # lattice = data_input[2]
    energy = data_input[3]

    # Direction selecting
    if direction in ["X","x"]:
        kpoints_type = "X - Kpoints"
        kpoints = x_kpoints
    elif direction in ["Y","y"]:
        kpoints_type = "Y - Kpoints"
        kpoints = y_kpoints
    elif direction in ["Z","z"]:
        kpoints_type = "Z - Kpoints"
        kpoints = z_kpoints
    else:
        kpoints_type = "Total Kpoints"
        kpoints = tot_kpoints

    # Figure title
    plt.title(f"Free energy versus {kpoints_type} for {matter}")
    plt.xlabel(f"{kpoints_type}"); plt.ylabel(r"Energy (eV)")

    # Axis style
    plt.ticklabel_format(style="sci", axis="y", scilimits=(-3,3))
    if kpoints_end is None:
        kpoints_end = np.max(kpoints)
    if kpoints_start is None:
        kpoints_start = 1

    start_index = kpoints.index(kpoints_start)
    end_index = kpoints.index(kpoints_end)

    kpoints_plotting = kpoints[start_index:end_index+1]
    energy_plotting = energy[start_index:end_index+1]

    # Plotting
    plt.scatter(kpoints_plotting, energy_plotting, c=colors[1], zorder =1)
    plt.xticks(kpoints_plotting)
