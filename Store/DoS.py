#### DoS
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

from Store.output import canvas_setting, color_sampling

def extract_dos(file_path):
    # Analysis vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()

    ## Extract Fermi energy
    efermi_element = root.find(".//dos/i[@name='efermi']")
    efermi = float(efermi_element.text.strip())

    ## Extract the number of ions
    first_positions = root.find(".//varray[@name='positions'][1]")
    positions_concatenated_text = " ".join([position.text for position in first_positions.findall("v")])
    positions_array = np.fromstring(positions_concatenated_text, sep=" ")
    positions_matrix = positions_array.reshape(-1, 3)
    ions_number = positions_matrix.shape[0]

    ## Extract the number of kpoints
    kpointlist = root.find(".//varray[@name='kpointlist']")
    kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
    kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
    kpointlist_matrix = kpointlist_array.reshape(-1, 3)
    kpoints_number = kpointlist_matrix.shape[0]

    ## Extract eigen, occupancy number
    for kpoints_index in range(1, kpoints_number+1):
        xpath_expr = f".//set[@comment='kpoint {kpoints_index}']"
        eigen_column = np.empty(0)
        occu_column  = np.empty(0)
        for eigen_occ_element in root.find(xpath_expr):
            values_eigen = list(map(float, eigen_occ_element.text.split()))
            eigen_var = values_eigen[0]
            eigen_column = np.append(eigen_column, eigen_var)
            occu_var = values_eigen[1]
            occu_column = np.append(occu_column, occu_var)
        if kpoints_index == 1 :
            eigen_matrix = eigen_column.reshape(-1, 1)
            occu_matrix = occu_column.reshape(-1, 1)
        else:
            eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
            occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))

    ## Extract energy, total DOS, and integrated DOS
    # lists initialization
    energy_dos_list     = np.empty(0)
    total_dos_list      = np.empty(0)
    integrated_dos_list = np.empty(0)
    path_dos = ".//total/array/set/set[@comment='spin 1']/r"
    for element_dos in root.findall(path_dos):
        values_dos = list(map(float, element_dos.text.split()))
        energy_dos_list = np.append(energy_dos_list, values_dos[0])
        total_dos_list = np.append(total_dos_list, values_dos[1])
        integrated_dos_list = np.append(integrated_dos_list, values_dos[2])
    shift = efermi
    energy_dos_shift = energy_dos_list - shift

    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
            energy_dos_shift, total_dos_list, integrated_dos_list)                      # 5 ~ 7

# DoS Plotting
def plot_dos_sol(matter, x_range=None, y_top=None, supplement=None, dos_type=None, dos_data=None, color_family="blue"):
    # Help information
    help_info = "Usage: plot_dos_sol" + \
                "Use extract_dos to extract the DoS data."

    if matter in ["help", "Help"]:
        print(help_info)

    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Color calling
    fermi_color = color_sampling("Orange")

    # Data plotting range
    # y_axis_top = max(dos_data[6]); y_limit = y_axis_top * 0.6
    # y_axis_top = max(max(total_dos_list), max(integrated_dos_list))
    y_limit = y_top

    # Data plotting
    colors = color_sampling(color_family)
    if dos_type in ["All", "all"]:
        plt.plot(dos_data[5], dos_data[6], c=colors[1], label="Total DoS", zorder=3)
        plt.plot(dos_data[5], dos_data[7], c=colors[2], label="Integrated DoS", zorder=2)
    if dos_type in ["Total", "total"]:
        plt.plot(dos_data[5], dos_data[6], c=colors[1], label="Total DoS", zorder=2)
    if dos_type in ["Integrated", "integrated"]:
        plt.plot(dos_data[5], dos_data[7], c=colors[2], label="Integrated DoS", zorder=2)

    # Plot Fermi energy as a vertical line
    efermi = dos_data[0]
    shift = efermi
    plt.axvline(x = efermi-shift, linestyle="--", c=fermi_color[1], alpha=0.95, label="Fermi energy", zorder=1)
    fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
    plt.text(efermi-shift-x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")

    # Title
    plt.title(f"Electronic density of state for {matter} ({supplement})")
    plt.ylabel(r"Density of States", fontsize = 1.0* 12)
    plt.xlabel(r"Energy (eV)", fontsize = 1.0* 12)

    plt.ylim(0, y_limit)
    plt.xlim(x_range*(-1), x_range)
    plt.legend(loc="upper right")
    # plt.show()

# Universal DoS Plotting
def plot_dos(title, x_range = None, y_top = None, supplement = None, dos_type = None, matters = None):
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
    fermi_color = color_sampling("Orange")

    if all(term is not None for term in [x_range, y_top]):
        # Data plotting
        if dos_type in ["All", "all"]:
            for index, matter in enumerate(matters):
                plt.plot(matter[1][5], matter[1][6], c=color_sampling(matter[2])[1], label=f"Total DOS for {matter[0]}", zorder = 3)
                plt.plot(matter[1][5], matter[1][7], c=color_sampling(matter[2])[2], label=f"Integrated DOS for {matter[0]}", zorder = 2)
                efermi = matter[1][0]
        if dos_type in ["Total", "total"]:
            for index, matter in enumerate(matters):
                plt.plot(matter[1][5], matter[1][6], c=color_sampling(matter[2])[1], label=f"Total DOS for {matter[0]}", zorder = 2)
                efermi = matter[1][0]
        if dos_type in ["Integrated", "integrated"]:
            for index, matter in enumerate(matters):
                plt.plot(matter[1][5], matter[1][7], c=color_sampling(matter[2])[2], label=f"Integrated DOS for {matter[0]}", zorder = 2)
                efermi = matter[1][0]
        # Plot Fermi energy as a vertical line
        shift = efermi
        plt.axvline(x = efermi-shift, linestyle="--", c=fermi_color[1], alpha=0.95, label="Fermi energy", zorder = 1)
        fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
        plt.text(efermi-shift-x_range*0.02, y_top*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")

        # Title
        plt.title(f"Electronic density of state for {title} ({supplement})")
        plt.ylabel(r"Density of States"); plt.xlabel(r"Energy (eV)")

        plt.ylim(0, y_top)
        plt.xlim(x_range*(-1), x_range)
        # plt.legend(loc="best")
        plt.legend(loc="upper right")
