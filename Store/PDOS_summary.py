##### Data Process and Ploting for PDOS
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

params = {"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
          "axes.titlesize": 18, "axes.labelsize": 12, "figure.facecolor": "w"}
plt.rcParams.update(params)

def pdos_single_summary(matter, pdos_total, element, pdos_element, x_range, y_top):

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), dpi=196)

    axes_element = [ax0, ax1]
    elements = [matter, element]
    pdos_results = [pdos_total, pdos_element]

    label_positions = {0: (1, 0), 1: (0, 0)}

    fig.suptitle(f"Projected electronic density of state for {matter}", fontsize =1.0*18)

    for i in range(2):
        ax = axes_element[i]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        pdos_result = pdos_results[i]
        element = elements[i]

        efermi_pdos = pdos_result[0]
        energy_dos_shift = pdos_result[5]
        total_dos_list = pdos_result[6]
        energy_pdos_shift = pdos_result[8]
        s_pdos_sum = pdos_result[9]
        p_y_pdos_sum = pdos_result[10]
        p_z_pdos_sum = pdos_result[11]
        p_x_pdos_sum = pdos_result[12]

        ax.set_title(f"PDOS for {element}", fontsize =1.0* 16)
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
        ax.plot(energy_pdos_array,   s_pdos_array, c="#FF5064", label=r"$s$ PDOS",  zorder=1)
        ax.plot(energy_pdos_array, p_x_pdos_array, c="#1473D2", label=r"$p_x$ PDOS",zorder=1)
        ax.plot(energy_pdos_array, p_y_pdos_array, c="#37AA3C", label=r"$p_y$ PDOS",zorder=1)
        ax.plot(energy_pdos_array, p_z_pdos_array, c="#F096FF", label=r"$p_z$ PDOS",zorder=1)

        shift = efermi_pdos

        if i == 0:
            y_limit = y_top
            ax.legend(loc="upper right")
        elif i == 1:
            y_limit = y_top * 0.5
            ax.legend(loc="upper right")

        ax.axvline(x = efermi_pdos-shift, linestyle="--", color="#F5820F", alpha=0.95, label=r"Fermi energy")
        ax.set_ylim(0, y_limit)
        ax.set_xlim(-x_range, x_range)

        fermi_energy_text = f"Fermi energy: {efermi_pdos:.3f} (eV)"
        ax.text(efermi_pdos-shift-x_range*0.02, y_limit*0.95, fermi_energy_text, fontsize =1.0*12, color="#EB731E", rotation=0, ha="right")

        x_label, y_label = label_positions[i]
        relative_offset = 0.02  # adjust this value as needed
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


def pdos_duo_summary(matter, pdos_total, element_1, pdos_1, element_2, pdos_2, x_range, y_top):

    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

    fig = plt.figure(figsize=(12,10), dpi=196)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    axes_element = [ax0, ax1, ax2]
    elements = [matter, element_1, element_2]
    pdos_results = [pdos_total, pdos_1, pdos_2]

    label_positions = {0: (0, 0), 1: (1, 1), 2:(0, 1)}

    fig.suptitle(f"Projected electronic density of state for {matter}", fontsize =1.0*18)

    for i in range(3):
        ax = axes_element[i]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        pdos_result = pdos_results[i]
        element = elements[i]

        efermi_pdos = pdos_result[0]
        energy_dos_shift = pdos_result[5]
        total_dos_list = pdos_result[6]
        energy_pdos_shift = pdos_result[8]
        s_pdos_sum = pdos_result[9]
        p_y_pdos_sum = pdos_result[10]
        p_z_pdos_sum = pdos_result[11]
        p_x_pdos_sum = pdos_result[12]

        ax.set_title(f"PDOS for {element}", fontsize =1.0* 16)
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
        ax.plot(energy_pdos_array,   s_pdos_array, c="#FF5064", label=r"$s$ PDOS",  zorder=1)
        ax.plot(energy_pdos_array, p_x_pdos_array, c="#1473D2", label=r"$p_x$ PDOS",zorder=1)
        ax.plot(energy_pdos_array, p_y_pdos_array, c="#37AA3C", label=r"$p_y$ PDOS",zorder=1)
        ax.plot(energy_pdos_array, p_z_pdos_array, c="#F096FF", label=r"$p_z$ PDOS",zorder=1)

        shift = efermi_pdos

        if i == 0:
            y_limit = y_top
            ax.legend(loc="upper right")
        elif i == 1:
            y_limit = y_top * 0.5
            ax.legend(loc="upper left")
        elif i == 2:
            y_limit = y_top * 0.5
            ax.legend(loc="upper right")

        ax.axvline(x = efermi_pdos-shift, linestyle="--", color="#F5820F", alpha=0.95, label=r"Fermi energy")
        ax.set_ylim(0, y_limit)
        ax.set_xlim(-x_range, x_range)

        fermi_energy_text = f"Fermi energy: {efermi_pdos:.3f} (eV)"
        if i == 1:
            ax.text(efermi_pdos-shift+x_range*0.02, y_limit*0.95, fermi_energy_text, fontsize =1.0*12, color="#EB731E", rotation=0, ha="left")
        else:
            ax.text(efermi_pdos-shift-x_range*0.02, y_limit*0.95, fermi_energy_text, fontsize =1.0*12, color="#EB731E", rotation=0, ha="right")

        x_label, y_label = label_positions[i]
        if i == 0:
            x_relative_offset = 0.02 * 0.5
            y_relative_offset = 0.02
        else:
            x_relative_offset = 0.02
            y_relative_offset = 0.02
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
