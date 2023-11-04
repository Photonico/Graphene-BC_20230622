##### Data Process and Ploting for PDOS
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915

import matplotlib.pyplot as plt
import numpy as np

params = {"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
          "axes.titlesize": 18, "axes.labelsize": 12, "figure.facecolor": "w"}
plt.rcParams.update(params)

# Total PDOS Plotting
def pdos_total_plotting(pdos_result, matter, y_top):
    # Extract data
    # pdos_result = pdos_single_extracting(pdos_file_path)
    efermi_pdos = pdos_result[0]
    energy_dos_shift = pdos_result[5]
    total_dos_list = pdos_result[6]
    energy_pdos_shift = pdos_result[8]
    s_pdos_sum = pdos_result[9]
    p_y_pdos_sum = pdos_result[10]
    p_z_pdos_sum = pdos_result[11]
    p_x_pdos_sum = pdos_result[12]

    # Set up the specified style parameters
    plt.figure(dpi=256, figsize=(9.6, 6.4))
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Title and labels
    # plt.title(f"Projected electronic density of state for {matter}")
    plt.title(f"PDOS for {matter}")
    plt.ylabel(r"Density of states", fontsize =1.0* 12)
    plt.xlabel(r"Energy (eV)", fontsize =1.0* 12)

    # Process the data
    energy_dos_array = np.array(energy_dos_shift)
    total_dos_array = np.array(total_dos_list)
    energy_pdos_array = np.array(energy_pdos_shift)
    s_pdos_array = np.array(s_pdos_sum)
    p_y_pdos_array = np.array(p_y_pdos_sum)
    p_z_pdos_array = np.array(p_z_pdos_sum)
    p_x_pdos_array = np.array(p_x_pdos_sum)

    # Plot data
    # max_value = np.max([total_dos_array, s_pdos_array, p_y_pdos_array, p_z_pdos_array, p_x_pdos_array])
    # y_axis_top = max_value
    # y_limit = y_axis_top * 1.125
    y_limit = y_top
    x_range = 6
    shift = efermi_pdos

    plt.plot(energy_dos_array, total_dos_array, c="#8C64F0", label=r"Total DOS", zorder=2)
    plt.plot(energy_pdos_array,   s_pdos_array, c="#FF5064", label=r"$s$ PDOS",  zorder=1)
    plt.plot(energy_pdos_array, p_x_pdos_array, c="#1473D2", label=r"$p_x$ PDOS",zorder=1)
    plt.plot(energy_pdos_array, p_y_pdos_array, c="#37AA3C", label=r"$p_y$ PDOS",zorder=1)
    plt.plot(energy_pdos_array, p_z_pdos_array, c="#F096FF", label=r"$p_z$ PDOS",zorder=1)

    plt.axvline(x = efermi_pdos-shift, linestyle="--", color="#F5820F", alpha=0.95, label=r"Fermi energy")

    # fermi_energy_text = f"Fermi energy: {efermi_pdos:.3f} (eV)"
    # plt.text(efermi_pdos-shift-x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color="#EB731E", rotation=0, ha="right")

    plt.ylim(0, y_limit)
    plt.xlim(-x_range, x_range)
    plt.legend(loc="best")


# PDOS Plotting for elements
def pdos_element_plotting(pdos_result, matter, y_top):
    # Extract data
    # pdos_result = pdos_single_extracting(pdos_file_path)
    efermi_pdos = pdos_result[0]
    energy_dos_shift = pdos_result[5]
    total_dos_list = pdos_result[6]
    energy_pdos_shift = pdos_result[8]
    s_pdos_sum = pdos_result[9]
    p_y_pdos_sum = pdos_result[10]
    p_z_pdos_sum = pdos_result[11]
    p_x_pdos_sum = pdos_result[12]

    # Set up the specified style parameters
    plt.figure(dpi=256, figsize=(9.6, 6.4))
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Title and labels
    # plt.title(f"Projected electronic density of state for {matter}")
    plt.title(f"PDOS for {matter}")
    plt.ylabel(r"Density of states", fontsize =1.0* 12)
    plt.xlabel(r"Energy (eV)", fontsize =1.0* 12)

    # Process the data
    energy_dos_array = np.array(energy_dos_shift)
    total_dos_array = np.array(total_dos_list)
    energy_pdos_array = np.array(energy_pdos_shift)
    s_pdos_array = np.array(s_pdos_sum)
    p_y_pdos_array = np.array(p_y_pdos_sum)
    p_z_pdos_array = np.array(p_z_pdos_sum)
    p_x_pdos_array = np.array(p_x_pdos_sum)

    # Plot data
    # max_value = np.max([total_dos_array, s_pdos_array, p_y_pdos_array, p_z_pdos_array, p_x_pdos_array])
    # y_axis_top = max_value
    # y_limit = y_axis_top * 1.125
    y_limit = y_top
    x_range = 6
    shift = efermi_pdos

    plt.plot(energy_dos_array, total_dos_array, c="#8C64F0", label=r"Total DOS", zorder=2)
    plt.plot(energy_pdos_array,   s_pdos_array, c="#FF5064", label=r"$s$ PDOS",  zorder=1)
    plt.plot(energy_pdos_array, p_x_pdos_array, c="#1473D2", label=r"$p_x$ PDOS",zorder=1)
    plt.plot(energy_pdos_array, p_y_pdos_array, c="#37AA3C", label=r"$p_y$ PDOS",zorder=1)
    plt.plot(energy_pdos_array, p_z_pdos_array, c="#F096FF", label=r"$p_z$ PDOS",zorder=1)

    plt.axvline(x = efermi_pdos-shift, linestyle="--", color="#F5820F", alpha=0.95, label=r"Fermi energy")

    # fermi_energy_text = f"Fermi energy: {efermi_pdos:.3f} (eV)"
    # plt.text(efermi_pdos-shift-x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color="#EB731E", rotation=0, ha="right")

    plt.ylim(0, y_limit)
    plt.xlim(-x_range, x_range)
    plt.legend(loc="best")

    # plt.show()
