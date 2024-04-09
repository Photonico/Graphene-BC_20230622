#### Declarations of process functions for PDoS plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# import xml.etree.ElementTree as ET
# import os
# import numpy as np
import matplotlib.pyplot as plt

from vmatplot.output import canvas_setting, color_sampling
# from vmatplot.commons import extract_fermi
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
        # label
        if matter[0] not in ["", None]:
            current_label = f"({matter[0]})"
        else:
            current_label = ""
        if pdos_type in ["All", "all"]:
            # for _, matter in enumerate(matters):
            plt.plot(matter[1][8], matter[1][6], c=color_sampling(matter[2])[1], label=f"Total PDoS {current_label}", alpha=matter[3], linestyle=matter[4], zorder = 2)
            plt.plot(matter[1][8], matter[1][7], c=color_sampling(matter[2])[2], label=f"Integrated PDoS {current_label}", alpha=matter[3], linestyle=matter[4], zorder = 3)
            efermi = matter[1][0]
        if pdos_type in ["Total", "total"]:
            # for _, matter in enumerate(matters):
            plt.plot(matter[1][8], matter[1][6], c=color_sampling(matter[2])[1], label=f"Total PDoS {current_label}", alpha=matter[3], linestyle=matter[4], zorder = 2)
            efermi = matter[1][0]
        if pdos_type in ["Integrated", "integrated"]:
            # for _, matter in enumerate(matters):
            plt.plot(matter[1][8], matter[1][7], c=color_sampling(matter[2])[2], label=f"Integrated PDoS {current_label}", alpha=matter[3], linestyle=matter[4], zorder = 3)
            efermi = matter[1][0]
        plt.plot(matter[1][8], matter[1][9],  c=color_sampling(matter[2])[3], alpha=matter[3], linestyle=matter[4], label=f"$s$ PDoS {current_label}",  zorder=4)
        plt.plot(matter[1][8], matter[1][12], c=color_sampling(matter[2])[4], alpha=matter[3], linestyle=matter[4], label=f"$p_x$ PDoS {current_label}",zorder=5)
        plt.plot(matter[1][8], matter[1][10], c=color_sampling(matter[2])[5], alpha=matter[3], linestyle=matter[4], label=f"$p_y$ PDoS {current_label}",zorder=6)
        plt.plot(matter[1][8], matter[1][11], c=color_sampling(matter[2])[6], alpha=matter[3], linestyle=matter[4], label=f"$p_z$ PDoS {current_label}",zorder=7)

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

def plot_total_segment(title, matters_list):
    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Color calling
    fermi_color = color_sampling("Violet")

    # Ranges
    x_range = matters_list[0][1]
    y_top = matters_list[0][2]

    # Matter
    matter = matters_list
    num_elements = (len(matter[0]) - 3) // 2
    # Plot PDoS
    for index in range(num_elements):
        i_0 = 2 * index
        i_1 = 3 * index
        if matter[0][3+i_0] not in ["", None]:
            current_label = f"({matter[0][3+i_0]})"
        else:
            current_label = ""
        plt.plot(matter[0][4+i_0][8], matter[0][4+i_0][6],  c=color_sampling(matter[1][0+i_1])[1], alpha=matter[1][1+i_1], linestyle=matter[1][2+i_1], label=f"Total PDoS {current_label}", zorder=2)
        plt.plot(matter[0][4+i_0][8], matter[0][4+i_0][9],  c=color_sampling(matter[1][0+i_1])[3], alpha=matter[1][1+i_1], linestyle=matter[1][2+i_1], label=f"$s$ PDoS {current_label}", zorder=3)
        plt.plot(matter[0][4+i_0][8], matter[0][4+i_0][12], c=color_sampling(matter[1][0+i_1])[4], alpha=matter[1][1+i_1], linestyle=matter[1][2+i_1], label=f"$p_x$ PDoS {current_label}", zorder=4)
        plt.plot(matter[0][4+i_0][8], matter[0][4+i_0][10], c=color_sampling(matter[1][0+i_1])[5], alpha=matter[1][1+i_1], linestyle=matter[1][2+i_1], label=f"$p_y$ PDoS {current_label}", zorder=5)
        plt.plot(matter[0][4+i_0][8], matter[0][4+i_0][11], c=color_sampling(matter[1][0+i_1])[6], alpha=matter[1][1+i_1], linestyle=matter[1][2+i_1], label=f"$p_z$ PDoS {current_label}", zorder=6)
    efermi = matter[0][4][0]

    # Plot Fermi energy as a vertical line
    shift = efermi
    plt.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder = 1)
    fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
    plt.text(efermi-shift-x_range*0.02, y_top*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")

    # Title
    # plt.title(f"Electronic density of state for {title} ({supplement})")
    plt.title(f"Total PDoS for {title} ")
    plt.ylabel(r"Density of States"); plt.xlabel(r"Energy (eV)")

    plt.ylim(0, y_top)
    plt.xlim(x_range*(-1), x_range)
    # plt.legend(loc="best")
    plt.legend(loc="upper right")


def plot_sol_segment_pdos(title, matters_list):
    # Figure settings
    fig_setting = canvas_setting(8, 11)    # 2 * 5 + 1
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b"]

    # Materials information
    num_elements = len(matters_list[-1])//3
    matter = matters_list
    efermi = matter[0][4][0]

    # Ranges
    x_range = []
    y_top   = []
    for subplot_index in range(3):
        x_range.append(matter[subplot_index][1])
        y_top.append(matter[subplot_index][2])

    # Data process
    titles = []
    labels = [[], []]
    pdoses = [[], []]
    for subplot_index in range(2):
        titles.append(matter[subplot_index][0])
        for matter_index in range(num_elements):
            labels[matter_index].append(matter[subplot_index][3+2*matter_index])
            pdoses[matter_index].append(matter[subplot_index][4+2*matter_index])

    # Style parameters
    color = []
    alpha = []
    lines = []
    for matter_index in range(num_elements):
        color.append(matter[-1][0+3*matter_index])
        alpha.append(matter[-1][1+3*matter_index])
        lines.append(matter[-1][2+3*matter_index])

    fig.suptitle(f"PDoS for {title}", fontsize=fig_setting[3][0], y=0.99)

    for supplot_index in range(2):
        ax = axes_element[supplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # Data ploting
        ax.set_title(f"{titles[supplot_index]}", fontsize=fig_setting[3][1])
        for matter_index in range(num_elements):
            if labels[matter_index][supplot_index] not in [None, ""]:
                current_label = f"({labels[matter_index][supplot_index]})"
            else:
                current_label = ""
            current_pdos  = pdoses[matter_index][supplot_index]
            ax.plot(current_pdos[8], current_pdos[6],c=color_sampling(color[matter_index])[1],alpha=alpha[matter_index],ls=lines[matter_index],label=f"Total PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[9],c=color_sampling(color[matter_index])[3],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$s$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[12],c=color_sampling(color[matter_index])[4],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_x$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[10],c=color_sampling(color[matter_index])[5],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_y$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[11],c=color_sampling(color[matter_index])[6],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_z$ PDoS {current_label}",zorder=2)
        ax.set_xlim(-x_range[supplot_index],x_range[supplot_index])
        ax.set_ylim(0, y_top[supplot_index])
        ax.set_ylabel(r"Density of states")
        if supplot_index == 1:
            ax.set_xlabel(r"Energy (eV)")
        shift = efermi
        fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
        ax.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder=1)

        # Fermi energy
        ax.text(efermi-shift-x_range[supplot_index]*0.02, y_top[supplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")
        ax.legend(loc="upper right")

        # Subplots label
        orderlab_shift = 0.05
        x_loc = 0+orderlab_shift*0.75
        y_loc = 1-orderlab_shift

        ax.annotate(f"({order_labels[supplot_index]})",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="center", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})
    plt.tight_layout()

def plot_duo_segment_pdos(title, matters_list):
    # Figure settings
    fig_setting = canvas_setting(8, 16)    # 3 * 5 + 1
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(3, 1, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1], axs[2]]

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b","c"]

    # Materials information
    num_elements = len(matters_list[-1])//3
    matter = matters_list
    efermi = matter[0][4][0]

    # Ranges
    x_range = []
    y_top   = []
    for subplot_index in range(3):
        x_range.append(matter[subplot_index][1])
        y_top.append(matter[subplot_index][2])

    # Data process
    titles = []
    labels = [[], []]
    pdoses = [[], []]
    for subplot_index in range(3):
        titles.append(matter[subplot_index][0])
        for matter_index in range(num_elements):
            labels[matter_index].append(matter[subplot_index][3+2*matter_index])
            pdoses[matter_index].append(matter[subplot_index][4+2*matter_index])

    # Style parameters
    color = []
    alpha = []
    lines = []
    for matter_index in range(num_elements):
        color.append(matter[-1][0+3*matter_index])
        alpha.append(matter[-1][1+3*matter_index])
        lines.append(matter[-1][2+3*matter_index])

    fig.suptitle(f"PDoS for {title}", fontsize=fig_setting[3][0], y=0.99)

    for supplot_index in range(3):
        ax = axes_element[supplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # Data ploting
        ax.set_title(f"{titles[supplot_index]}", fontsize=fig_setting[3][1])
        for matter_index in range(num_elements):
            if labels[matter_index][supplot_index] != "":
                current_label = f"({labels[matter_index][supplot_index]})"
            if labels[matter_index][supplot_index] == "":
                current_label = ""
            current_pdos  = pdoses[matter_index][supplot_index]
            ax.plot(current_pdos[8], current_pdos[6],c=color_sampling(color[matter_index])[1],alpha=alpha[matter_index],ls=lines[matter_index],label=f"Total PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[9],c=color_sampling(color[matter_index])[3],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$s$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[12],c=color_sampling(color[matter_index])[4],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_x$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[10],c=color_sampling(color[matter_index])[5],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_y$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[11],c=color_sampling(color[matter_index])[6],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_z$ PDoS {current_label}",zorder=2)
        ax.set_xlim(-x_range[supplot_index],x_range[supplot_index])
        ax.set_ylim(0, y_top[supplot_index])
        ax.set_ylabel(r"Density of states")
        if supplot_index == 2:
            ax.set_xlabel(r"Energy (eV)")
        shift = efermi
        fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
        ax.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder=1)

        # Fermi energy
        ax.text(efermi-shift-x_range[supplot_index]*0.02, y_top[supplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")
        ax.legend(loc="upper right")

        # Subplots label
        orderlab_shift = 0.05
        x_loc = 0+orderlab_shift*0.75
        y_loc = 1-orderlab_shift

        ax.annotate(f"({order_labels[supplot_index]})",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="center", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})
    plt.tight_layout()

def plot_tri_segment_pdos(title, matters_list):

    # Figure settings
    fig_setting = canvas_setting(8, 21)    # 4 * 5 + 1
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(4, 1, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1], axs[2], axs[3]]

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b","c","d"]

    # Materials information
    num_elements = len(matters_list[-1])//3
    matter = matters_list
    efermi = matter[0][4][0]

    # Ranges
    x_range = []
    y_top   = []
    for subplot_index in range(4):
        x_range.append(matter[subplot_index][1])
        y_top.append(matter[subplot_index][2])

    # Data process
    titles = []
    labels = [[], []]
    pdoses = [[], []]
    for subplot_index in range(4):
        titles.append(matter[subplot_index][0])
        for matter_index in range(num_elements):
            labels[matter_index].append(matter[subplot_index][3+2*matter_index])
            pdoses[matter_index].append(matter[subplot_index][4+2*matter_index])

    # Style parameters
    color = []
    alpha = []
    lines = []
    for matter_index in range(num_elements):
        color.append(matter[-1][0+3*matter_index])
        alpha.append(matter[-1][1+3*matter_index])
        lines.append(matter[-1][2+3*matter_index])

    fig.suptitle(f"PDoS for {title}", fontsize=fig_setting[3][0], y=0.99)

    for supplot_index in range(4):
        ax = axes_element[supplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # Data ploting
        ax.set_title(f"{titles[supplot_index]}", fontsize=fig_setting[3][1])
        for matter_index in range(num_elements):
            if labels[matter_index][supplot_index] != "":
                current_label = f"({labels[matter_index][supplot_index]})"
            if labels[matter_index][supplot_index] == "":
                current_label = ""
            current_pdos  = pdoses[matter_index][supplot_index]
            ax.plot(current_pdos[8], current_pdos[6],c=color_sampling(color[matter_index])[1],alpha=alpha[matter_index],ls=lines[matter_index],label=f"Total PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[9],c=color_sampling(color[matter_index])[3],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$s$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[12],c=color_sampling(color[matter_index])[4],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_x$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[10],c=color_sampling(color[matter_index])[5],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_y$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[11],c=color_sampling(color[matter_index])[6],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_z$ PDoS {current_label}",zorder=2)
        ax.set_xlim(-x_range[supplot_index],x_range[supplot_index])
        ax.set_ylim(0, y_top[supplot_index])
        ax.set_ylabel(r"Density of states")
        if supplot_index == 3:
            ax.set_xlabel(r"Energy (eV)")
        shift = efermi
        fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
        ax.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder=1)

        # Fermi energy
        ax.text(efermi-shift-x_range[supplot_index]*0.02, y_top[supplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")
        ax.legend(loc="upper right")

        # Subplots label
        orderlab_shift = 0.05
        x_loc = 0+orderlab_shift*0.75
        y_loc = 1-orderlab_shift

        ax.annotate(f"({order_labels[supplot_index]})",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="center", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})
    plt.tight_layout()

def plot_tri_segment_pdos_block(title, matters_list):

    # Figure settings
    fig_setting = canvas_setting(16, 12)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b","c","d"]

    # Materials information
    num_elements = len(matters_list[-1])//3
    matter = matters_list
    efermi = matter[0][4][0]

    # Ranges
    x_range = []
    y_top   = []
    for subplot_index in range(4):
        x_range.append(matter[subplot_index][1])
        y_top.append(matter[subplot_index][2])

    # Data process
    titles = []
    labels = [[], []]
    pdoses = [[], []]
    for subplot_index in range(4):
        titles.append(matter[subplot_index][0])
        for matter_index in range(num_elements):
            labels[matter_index].append(matter[subplot_index][3+2*matter_index])
            pdoses[matter_index].append(matter[subplot_index][4+2*matter_index])

    # Style parameters
    color = []
    alpha = []
    lines = []
    for matter_index in range(num_elements):
        color.append(matter[-1][0+3*matter_index])
        alpha.append(matter[-1][1+3*matter_index])
        lines.append(matter[-1][2+3*matter_index])

    fig.suptitle(f"PDoS for {title}", fontsize=fig_setting[3][0])

    for supplot_index in range(4):
        ax = axes_element[supplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

        # Data ploting
        ax.set_title(f"{titles[supplot_index]}", fontsize=fig_setting[3][1])
        for matter_index in range(num_elements):
            if labels[matter_index][supplot_index] != "":
                current_label = f"({labels[matter_index][supplot_index]})"
            if labels[matter_index][supplot_index] == "":
                current_label = ""
            current_pdos  = pdoses[matter_index][supplot_index]
            ax.plot(current_pdos[8], current_pdos[6],c=color_sampling(color[matter_index])[1],alpha=alpha[matter_index],ls=lines[matter_index],label=f"Total PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[9],c=color_sampling(color[matter_index])[3],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$s$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[12],c=color_sampling(color[matter_index])[4],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_x$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[10],c=color_sampling(color[matter_index])[5],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_y$ PDoS {current_label}",zorder=2)
            ax.plot(current_pdos[8], current_pdos[11],c=color_sampling(color[matter_index])[6],alpha=alpha[matter_index],ls=lines[matter_index],label=f"$p_z$ PDoS {current_label}",zorder=2)
        ax.set_xlim(-x_range[supplot_index],x_range[supplot_index])
        ax.set_ylim(0, y_top[supplot_index])
        ax.set_ylabel(r"Density of states")
        if supplot_index == 3:
            ax.set_xlabel(r"Energy (eV)")
        shift = efermi
        fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
        ax.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=1.00, label="Fermi energy", zorder=1)
        if supplot_index == 2:
            ax.text(efermi-shift+x_range[supplot_index]*0.02, y_top[supplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="left")
            ax.legend(loc="upper left")
        else:
            ax.text(efermi-shift-x_range[supplot_index]*0.02, y_top[supplot_index]*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")
            ax.legend(loc="upper right")

        orderlab_shift = 0.05
        if supplot_index == 0:
            x_loc = 1-orderlab_shift*0.75
            y_loc = 0+orderlab_shift
        elif supplot_index == 1:
            x_loc = 0+orderlab_shift*0.75
            y_loc = 0+orderlab_shift
        elif supplot_index == 2:
            x_loc = 1-orderlab_shift*0.75
            y_loc = 1-orderlab_shift
        elif supplot_index == 3:
            x_loc = 0+orderlab_shift*0.75
            y_loc = 1-orderlab_shift

        ax.annotate(f"({order_labels[supplot_index]})",
                    xy=(x_loc,y_loc),
                    xycoords="axes fraction",
                    fontsize=1.0 * 16,
                    ha="center", va="center",
                    bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})
    plt.tight_layout()

plot_seg_helo_info = "help information"
plot_seg_usage = plot_seg_helo_info
def plot_segment_pdos(*args):
    if len(args) == 1:
        print(plot_seg_helo_info)
    if len(args) == 2:
        if len(args[1]) == 1:
            print("Format error")
            print(plot_seg_usage)
        if len(args[1]) == 2:
            return plot_total_segment(args[0], args[1])
        if len(args[1]) == 3:
            return plot_sol_segment_pdos(args[0], args[1])
        if len(args[1]) == 4:
            return plot_duo_segment_pdos(args[0], args[1])
        if len(args[1]) == 5:
            return plot_tri_segment_pdos(args[0], args[1])
