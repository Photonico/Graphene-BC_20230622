##### Data Process and Ploting for PDoS
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from Store.output import canvas_setting, color_sampling

###  PDoS Plotting for each element
def pdos_single_element_summary(matter, pdos_total, element, pdos_element, x_range, y_top, method):

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=256)
    axes_element = [axs[0], axs[1]]

    elements = [matter, element]
    pdos_results = [pdos_total, pdos_element]

    label_positions = {0: (1, 0), 1: (0, 0)}

    fig.suptitle(f"Projected electronic density of state for {matter} ({method})", fontsize =1.0*18)

    for i in range(2):
        ax = axes_element[i]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        pdos_data = pdos_results[i]
        element = elements[i]

        efermi_pdos = pdos_data[0]
        energy_dos_shift = pdos_data[5]
        total_dos_list = pdos_data[6]
        energy_pdos_shift = pdos_data[8]
        s_pdos_sum = pdos_data[9]
        p_y_pdos_sum = pdos_data[10]
        p_z_pdos_sum = pdos_data[11]
        p_x_pdos_sum = pdos_data[12]

        if i == 0:
            ax.set_title("Total PDoS", fontsize = 1.0 * 16)
        else:
            ax.set_title(f"PDoS for {element}", fontsize = 1.0 * 16)
        ax.set_ylabel(r"Density of states", fontsize =1.0* 12)
        ax.set_xlabel(r"Energy (eV)", fontsize =1.0* 12)

        energy_dos_array = np.array(energy_dos_shift)
        total_dos_array = np.array(total_dos_list)
        energy_pdos_array = np.array(energy_pdos_shift)
        s_pdos_array = np.array(s_pdos_sum)
        p_y_pdos_array = np.array(p_y_pdos_sum)
        p_z_pdos_array = np.array(p_z_pdos_sum)
        p_x_pdos_array = np.array(p_x_pdos_sum)

        ax.plot(energy_dos_array, total_dos_array, c="#8C64F0", label=r"Total DOS", zorder=2)
        ax.plot(energy_pdos_array,   s_pdos_array, c="#FF5064", label=r"$s$ PDoS",  zorder=1)
        ax.plot(energy_pdos_array, p_x_pdos_array, c="#1473D2", label=r"$p_x$ PDoS",zorder=1)
        ax.plot(energy_pdos_array, p_y_pdos_array, c="#37AA3C", label=r"$p_y$ PDoS",zorder=1)
        ax.plot(energy_pdos_array, p_z_pdos_array, c="#F096FF", label=r"$p_z$ PDoS",zorder=1)

        shift = efermi_pdos

        if i == 0:
            y_limit = y_top
        elif i == 1:
            y_limit = y_top * 0.5

        ax.axvline(x = efermi_pdos-shift, linestyle="--", color="#F5820F", alpha=0.95, label=r"Fermi energy")
        ax.set_ylim(0, y_limit)
        ax.set_xlim(-x_range, x_range)

        fermi_energy_text = f"Fermi energy\n{efermi_pdos:.3f} (eV)"
        ax.text(efermi_pdos-shift-x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color="#EB731E", rotation=0, va = "top", ha="right")

        x_label, y_label = label_positions[i]
        if i == 0:
            ax.legend(loc="upper right")
        elif i == 1:
            ax.legend(loc="upper right")
        relative_offset = 0.025  # adjust this value as needed
        if x_label == 0:
            ha = "left"; x_label_offset = relative_offset
        else:
            ha = "right"; x_label_offset = -relative_offset
        if y_label == 0:
            va = "bottom"; y_label_offset = relative_offset
        else:
            va = "top"; y_label_offset = -relative_offset
        ax.annotate(f"({chr(97 + i)})",
                    xy = (x_label + x_label_offset, y_label + y_label_offset),
                    xycoords = "axes fraction",
                    fontsize = 1.0 * 18,
                    ha = ha, va = va,
                    bbox = {"facecolor": "white", "alpha": 0.6, "edgecolor": "#AFAFAF", "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

def pdos_duo_element_summary(matter, pdos_total, element_1, pdos_1, element_2, pdos_2, x_range, y_top, method):

    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

    fig = plt.figure(figsize=(12,10), dpi=256)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    axes_element = [ax0, ax1, ax2]
    elements = [matter, element_1, element_2]
    pdos_results = [pdos_total, pdos_1, pdos_2]

    label_positions = {0: (0, 0), 1: (1, 1), 2:(0, 1)}

    fig.suptitle(f"Projected electronic density of state for {matter} ({method})", fontsize =1.0*18)

    for i in range(3):
        ax = axes_element[i]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        pdos_data = pdos_results[i]
        element = elements[i]

        efermi_pdos = pdos_data[0]
        energy_dos_shift = pdos_data[5]
        total_dos_list = pdos_data[6]
        energy_pdos_shift = pdos_data[8]
        s_pdos_sum = pdos_data[9]
        p_y_pdos_sum = pdos_data[10]
        p_z_pdos_sum = pdos_data[11]
        p_x_pdos_sum = pdos_data[12]

        if i == 0:
            ax.set_title("Total PDoS", fontsize = 1.0 * 16)
        else:
            ax.set_title(f"PDoS for {element}", fontsize = 1.0 * 16)
        ax.set_ylabel(r"Density of states", fontsize = 1.0 * 12)
        ax.set_xlabel(r"Energy (eV)", fontsize = 1.0 * 12)

        energy_dos_array = np.array(energy_dos_shift)
        total_dos_array = np.array(total_dos_list)
        energy_pdos_array = np.array(energy_pdos_shift)
        s_pdos_array = np.array(s_pdos_sum)
        p_y_pdos_array = np.array(p_y_pdos_sum)
        p_z_pdos_array = np.array(p_z_pdos_sum)
        p_x_pdos_array = np.array(p_x_pdos_sum)

        ax.plot(energy_dos_array, total_dos_array, c="#8C64F0", label=r"Total DOS", zorder=2)
        ax.plot(energy_pdos_array,   s_pdos_array, c="#FF5064", label=r"$s$ PDoS",  zorder=1)
        ax.plot(energy_pdos_array, p_x_pdos_array, c="#1473D2", label=r"$p_x$ PDoS",zorder=1)
        ax.plot(energy_pdos_array, p_y_pdos_array, c="#37AA3C", label=r"$p_y$ PDoS",zorder=1)
        ax.plot(energy_pdos_array, p_z_pdos_array, c="#F096FF", label=r"$p_z$ PDoS",zorder=1)

        shift = efermi_pdos

        if i == 0:
            y_limit = y_top
        elif i == 1:
            y_limit = y_top * 0.5
        elif i == 2:
            y_limit = y_top * 0.5

        ax.axvline(x = efermi_pdos-shift, linestyle="--", color="#F5820F", alpha=0.95, label=r"Fermi energy")
        ax.set_ylim(0, y_limit)
        ax.set_xlim(-x_range, x_range)
        fermi_energy_text = f"Fermi energy\n{efermi_pdos:.3f} (eV)"
        if i == 1:
            ax.text(efermi_pdos-shift+x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color="#EB731E", rotation=0, va = "top", ha="left")
        else:
            ax.text(efermi_pdos-shift-x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color="#EB731E", rotation=0, va = "top", ha="right")

        x_label, y_label = label_positions[i]
        if i == 0:
            x_relative_offset = 0.025 * 0.5
            y_relative_offset = 0.025
            ax.legend(loc="upper right")
        elif i==1:
            x_relative_offset = 0.025
            y_relative_offset = 0.025
            ax.legend(loc="upper left")
        elif i==2:
            x_relative_offset = 0.025
            y_relative_offset = 0.025
            ax.legend(loc="upper right")

        if x_label == 0:
            ha = "left"; x_label_offset = x_relative_offset
        else:
            ha = "right"; x_label_offset = -x_relative_offset
        if y_label == 0:
            va = "bottom"; y_label_offset = y_relative_offset
        else:
            va = "top"; y_label_offset = -y_relative_offset

        ax.annotate(f"({chr(97 + i)})",
                    xy = (x_label + x_label_offset, y_label + y_label_offset),
                    xycoords = "axes fraction",
                    fontsize = 1.0 * 18,
                    ha = ha, va = va,
                    bbox = {"facecolor": "white", "alpha": 0.6, "edgecolor": "#AFAFAF", "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

def pdos_tri_element_summary(matter, pdos_total, element_1, pdos_1, element_2, pdos_2, element_3, pdos_3, x_range, y_top, method):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=256)

    axes_element = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    elements = [matter, element_1, element_2, element_3]
    pdos_results = [pdos_total, pdos_1, pdos_2, pdos_3]

    label_positions = {0: (1, 0), 1: (0, 0), 2:(1, 1), 3:(0, 1)}

    fig.suptitle(f"Projected electronic density of state for {matter} ({method})", fontsize =1.0*18)

    for i in range(4):
        ax = axes_element[i]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        pdos_data = pdos_results[i]
        element = elements[i]

        efermi_pdos = pdos_data[0]
        energy_dos_shift = pdos_data[5]
        total_dos_list = pdos_data[6]
        energy_pdos_shift = pdos_data[8]
        s_pdos_sum = pdos_data[9]
        p_y_pdos_sum = pdos_data[10]
        p_z_pdos_sum = pdos_data[11]
        p_x_pdos_sum = pdos_data[12]

        if i == 0:
            ax.set_title("Total PDoS", fontsize = 1.0 * 16)
        else:
            ax.set_title(f"PDoS for {element}", fontsize = 1.0 * 16)
        ax.set_ylabel(r"Density of states", fontsize = 1.0 * 12)
        ax.set_xlabel(r"Energy (eV)", fontsize = 1.0 * 12)

        energy_dos_array = np.array(energy_dos_shift)
        total_dos_array = np.array(total_dos_list)
        energy_pdos_array = np.array(energy_pdos_shift)
        s_pdos_array = np.array(s_pdos_sum)
        p_y_pdos_array = np.array(p_y_pdos_sum)
        p_z_pdos_array = np.array(p_z_pdos_sum)
        p_x_pdos_array = np.array(p_x_pdos_sum)

        ax.plot(energy_dos_array, total_dos_array, c="#8C64F0", label=r"Total DOS", zorder=2)
        ax.plot(energy_pdos_array,   s_pdos_array, c="#FF5064", label=r"$s$ PDoS",  zorder=1)
        ax.plot(energy_pdos_array, p_x_pdos_array, c="#1473D2", label=r"$p_x$ PDoS",zorder=1)
        ax.plot(energy_pdos_array, p_y_pdos_array, c="#37AA3C", label=r"$p_y$ PDoS",zorder=1)
        ax.plot(energy_pdos_array, p_z_pdos_array, c="#F096FF", label=r"$p_z$ PDoS",zorder=1)

        shift = efermi_pdos

        if i == 0:
            y_limit = y_top
        elif i == 1:
            y_limit = round(y_top * 0.3)
        elif i == 2:
            y_limit = round(y_top * 0.3)
        elif i == 3:
            y_limit = round(y_top * 0.3)

        ax.axvline(x = efermi_pdos-shift, linestyle="--", color="#F5820F", alpha=0.95, label=r"Fermi energy")
        ax.set_ylim(0, y_limit)
        ax.set_xlim(-x_range, x_range)
        fermi_energy_text = f"Fermi energy\n{efermi_pdos:.3f} (eV)"
        if i == 2:
            ax.text(efermi_pdos-shift+x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color="#EB731E", rotation=0, va = "top", ha="left")
            ax.legend(loc="upper left")
        else:
            ax.text(efermi_pdos-shift-x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color="#EB731E", rotation=0, va = "top", ha="right")
            ax.legend(loc="upper right")

        x_label, y_label = label_positions[i]
        relative_offset = 0.025
        if x_label == 0:
            ha = "left"; x_label_offset = relative_offset
        else:
            ha = "right"; x_label_offset = -relative_offset
        if y_label == 0:
            va = "bottom"; y_label_offset = relative_offset
        else:
            va = "top"; y_label_offset = -relative_offset

        ax.annotate(f"({chr(97 + i)})",
                    xy = (x_label + x_label_offset, y_label + y_label_offset),
                    xycoords = "axes fraction",
                    fontsize = 1.0 * 18,
                    ha = ha, va = va,
                    bbox = {"facecolor": "white", "alpha": 0.6, "edgecolor": "#AFAFAF", "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

    # return list([matter, pdos_total, element_1, pdos_1, element_2, pdos_2, element_3, pdos_3, x_range, y_top, method])

## General usage
# def pdos_element_plotting(*args):
#     if args[0] == "help":
#         print("help information")
#         return
#     if len(args) == 5:
#         return plot_total_pdos(args[0], args[1], args[2], args[3], args[4])
#     if len(args) == 7:
#         return pdos_single_element_summary(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
#     if len(args) == 9:
#         return pdos_duo_element_summary(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])
#     if len(args) == 11:
#         return pdos_tri_element_summary(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10])
