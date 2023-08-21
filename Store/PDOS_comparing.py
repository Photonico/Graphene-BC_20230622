##### Data Process and Comparing Ploting for PDOS
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915

import matplotlib.pyplot as plt
import numpy as np

params = {"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
          "axes.titlesize": 18, "axes.labelsize": 12, "figure.facecolor": "w"}
plt.rcParams.update(params)

# Comparing plotting for two elements
def pdos_duo_element(matter, element_1, pdos_1, element_2, pdos_2, y_top):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 6.4), dpi=196)
 
    axes = [ax1, ax2]
    elements = [element_1, element_2]
    pdos_results = [pdos_1, pdos_2]

    fig.suptitle(f"Projected electronic density of state for {matter}", fontsize =1.0*18)
    for i in range(2):
        ax = axes[i]
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

        x_range = 6
        shift = efermi_pdos
        y_limit = y_top

        ax.axvline(x = efermi_pdos-shift, linestyle="--", color="#F5820F", alpha=0.95, label=r"Fermi energy")
        ax.set_ylim(0, y_limit)
        ax.set_xlim(-x_range, x_range)

        if i == 0:
            ax.legend(loc="best")
            label_x = ax.get_position().x1 - 0.001  # x1: right
            label_y = ax.get_position().y0 + 0.001  # y0: bottom
            va = "bottom"; ha = "right"
        else:
            ax.legend(loc="best")
            label_x = ax.get_position().x0 + 0.001  # x0: left
            label_y = ax.get_position().y0 + 0.001  # y0: bottom
            va = "bottom"; ha = "left"
        fig.text(label_x, label_y, f"({chr(97 + i)})", fontsize =1.0*18, va=va, ha=ha,
                bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "#D7D7D7", "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

def pdos_tri_element(matter, pdos_sum, element_1, pdos_1, element_2, pdos_2, element_3, pdos_3, y_top_0, y_top_1, y_top_2, y_top_3):
    # fig, (ax0, ax1, ax2, ax3) = plt.subplots(2, 2, figsize=(14.4, 10.2), dpi=196)
    # axes = [ax0, ax1, ax2, ax3]
    fig, axs = plt.subplots(2, 2, figsize=(14.4, 10.2), dpi=196)
    axes = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

    elements = [matter, element_1, element_2, element_3]
    pdos_results = [pdos_sum, pdos_1, pdos_2, pdos_3]

    fig.suptitle(f"Projected electronic density of state for {matter}", fontsize =1.0*18)
    for i in range(4):
        ax = axes[i]
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

        energy_dos_array = np.array(energy_dos_shift)
        total_dos_array = np.array(total_dos_list)
        energy_pdos_array = np.array(energy_pdos_shift)
        s_pdos_array = np.array(s_pdos_sum)
        p_y_pdos_array = np.array(p_y_pdos_sum)
        p_z_pdos_array = np.array(p_z_pdos_sum)
        p_x_pdos_array = np.array(p_x_pdos_sum)

        x_range = 6
        shift = efermi_pdos

        if i == 0:
            ax.set_title(f"The sum of PDOS in {matter}", fontsize =1.0* 16)
            ax.set_ylabel(r"Density of states", fontsize =1.0* 12)
            ax.set_xlabel(r"Energy (eV)", fontsize =1.0* 12)
        else:
            ax.set_title(f"PDOS for {element} layer", fontsize =1.0* 16)
            ax.set_ylabel(r"Density of states", fontsize =1.0* 12)
            ax.set_xlabel(r"Energy (eV)", fontsize =1.0* 12)

        ax.plot(energy_dos_array, total_dos_array, c="#8C64F0", label=r"Total DOS", zorder=2)
        ax.plot(energy_pdos_array,   s_pdos_array, c="#FF5064", label=r"$s$ PDOS",  zorder=1)
        ax.plot(energy_pdos_array, p_x_pdos_array, c="#1473D2", label=r"$p_x$ PDOS",zorder=1)
        ax.plot(energy_pdos_array, p_y_pdos_array, c="#37AA3C", label=r"$p_y$ PDOS",zorder=1)
        ax.plot(energy_pdos_array, p_z_pdos_array, c="#F096FF", label=r"$p_z$ PDOS",zorder=1)

        ax.axvline(x = efermi_pdos-shift, linestyle="--", color="#F5820F", alpha=0.95, label=r"Fermi energy")

        if i == 0:
            y_limit = y_top_0
            ax.legend(loc="upper right")
            label_x = ax.get_position().x1 - 0.001  # x1: right
            label_y = ax.get_position().y0 + 0.01   # y0: bottom
            va = "bottom"; ha = "right"
        elif i == 1:
            y_limit = y_top_1
            ax.legend(loc="upper right")
            label_x = ax.get_position().x0 + 0.001  # x0: left
            label_y = ax.get_position().y0 + 0.01   # y0: bottom
            va = "bottom"; ha = "left"
        elif i == 2:
            y_limit = y_top_2
            ax.legend(loc="upper left")
            label_x = ax.get_position().x1 + 0.001
            label_y = ax.get_position().y1 - 0.02
            va = "top"; ha = "right"
        else:
            y_limit = y_top_3
            ax.legend(loc="upper right")
            label_x = ax.get_position().x0 + 0.001
            label_y = ax.get_position().y1 - 0.02
            va = "top"; ha = "left"
        fig.text(label_x, label_y, f"({chr(97 + i)})", fontsize =1.0*18, va=va, ha=ha,
            bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "#D7D7D7", "boxstyle": "round, pad=0.2"})

        ax.set_ylim(0, y_limit)
        ax.set_xlim(-x_range, x_range)

    plt.tight_layout()
