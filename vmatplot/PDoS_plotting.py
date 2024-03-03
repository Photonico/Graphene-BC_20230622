#### Declarations of process functions for PDoS plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt

from vmatplot.output import canvas_setting, color_sampling
from vmatplot.commons import extract_fermi
from vmatplot.PDoS import extract_pdos

def create_matters_totpdos(matters_list):
    # matters = create_matters_dos(matters_list)
    # matters[0] = label
    # matters[1][0] = energy: x-axis
    # matters[1][6] = Total DoS
    # matters[1][7] = integral dos
    # matters[2] = color family
    # matters[3] = alpha
    # matters[4] = linestyle
    matters = []
    for matter_dir in matters_list:
        if len(matter_dir) == 2:
            label, directory = matter_dir
            color = "blue"
            alpha = 1.0
            linestyle = "solid"
        elif len(matter_dir) == 3:
            label, directory, color = matter_dir
            alpha = 1.0
            linestyle = "solid"
        elif len(matter_dir) == 4:
            label, directory, color, alpha = matter_dir
            linestyle = "solid"
        else:
            label, directory, color, alpha, linestyle = matter_dir
        dos_data = extract_pdos(directory)
        matters.append([label, dos_data, color, alpha, linestyle])
    return matters

def plot_totpdos(title, x_range = None, y_top = None, pdos_type = None, matters_list = None):
    # Help information
    help_info = "Usage: plot_dos" + \
                "Use extract_dos to extract the DoS data into a two-dimensional list firstly.\n"

    if title in ["help", "Help"]:
        print(help_info)
    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Color calling
    fermi_color = color_sampling("Violet")

    # Matter list
    matters = create_matters_totpdos(matters_list)
    for _, matter in enumerate(matters):
        plt.plot(matter[1][8], matter[1][9],  c=color_sampling(matter[2])[3], alpha=matter[3], linestyle=matter[4], label=f"$s$ PDoS for {matter[0]}",  zorder=4)
        plt.plot(matter[1][8], matter[1][12], c=color_sampling(matter[2])[4], alpha=matter[3], linestyle=matter[4], label=f"$p_x$ PDoS for {matter[0]}",zorder=3)
        plt.plot(matter[1][8], matter[1][10], c=color_sampling(matter[2])[5], alpha=matter[3], linestyle=matter[4], label=f"$p_y$ PDoS for {matter[0]}",zorder=2)
        plt.plot(matter[1][8], matter[1][11], c=color_sampling(matter[2])[6], alpha=matter[3], linestyle=matter[4], label=f"$p_z$ PDoS for {matter[0]}",zorder=1)
        if pdos_type in ["All", "all"]:
            # for _, matter in enumerate(matters):
            plt.plot(matter[1][8], matter[1][6], c=color_sampling(matter[2])[1], label=f"Total PDoS for {matter[0]}", alpha=matter[3], linestyle=matter[4], zorder = 6)
            plt.plot(matter[1][8], matter[1][7], c=color_sampling(matter[2])[2], label=f"Integrated PDoS for {matter[0]}", alpha=matter[3], linestyle=matter[4], zorder = 5)
            efermi = matter[1][0]  
        if pdos_type in ["Total", "total"]:
            # for _, matter in enumerate(matters):
            plt.plot(matter[1][8], matter[1][6], c=color_sampling(matter[2])[1], label=f"Total PDoS for {matter[0]}", alpha=matter[3], linestyle=matter[4], zorder = 5)
            efermi = matter[1][0]
        if pdos_type in ["Integrated", "integrated"]:
            # for _, matter in enumerate(matters):
            plt.plot(matter[1][8], matter[1][7], c=color_sampling(matter[2])[2], label=f"Integrated PDoS for {matter[0]}", alpha=matter[3], linestyle=matter[4], zorder = 5)
            efermi = matter[1][0]

    # Plot Fermi energy as a vertical line
    shift = efermi
    plt.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder = 1)
    fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
    plt.text(efermi-shift-x_range*0.02, y_top*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")

    # Title
    # plt.title(f"Electronic density of state for {title} ({supplement})")
    plt.title(f"{pdos_type} PDoS for {title} ")
    plt.ylabel(r"Density of States"); plt.xlabel(r"Energy (eV)")

    plt.ylim(0, y_top)
    plt.xlim(x_range*(-1), x_range)
    # plt.legend(loc="best")
    plt.legend(loc="upper right")

# def create_matters_elepdos(matters_list):

# def create_matters_segpdos(matters_list):

# plot_pdos_segment
