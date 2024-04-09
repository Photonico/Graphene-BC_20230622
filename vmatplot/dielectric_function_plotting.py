#### Dielectric function plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import matplotlib.pyplot as plt

from vmatplot.output import canvas_setting, color_sampling
from vmatplot.dielectric_function import extract_dielectric_function

def create_matters_dielectric_function(dielectric_list):
    # data = create_matters_dielectric_function(dielectric_list)
    # data[0] = current curve label
    # data[1] = dielectric data
    # data[2] = color family
    # data[3] = alpha
    # data[4] = linewidth
    data = []
    for dielectric_dir in dielectric_list:
        if len(dielectric_dir) == 2:
            label, directory = dielectric_dir
            color = "blue"
            alpha = 1.0
            linewidth = None
        elif len(dielectric_dir) == 3:
            label, directory, color = dielectric_dir
            alpha = 1.0
            linewidth = None
        elif len(dielectric_dir) == 4:
            label, directory, color, alpha = dielectric_dir
            linewidth = None
        else:
            label, directory, color, alpha,  linewidth = dielectric_dir
        dielectric_data = extract_dielectric_function(directory)
        data.append([label,dielectric_data,color,alpha,linewidth])
    return data

def plot_dielectric_function_XZ(title, dielectric_list, energy_max = None):
    # Coding reference: plot_sol_segment_pdos(title, matters_list)
    fig_setting = canvas_setting(8, 11)
    # fig_setting = canvas_setting(8, 21)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b"]

    # Materials information
    dataset = create_matters_dielectric_function(dielectric_list)
    subtitles = ["Inplane", "Outplane"]

    fig.suptitle(f"Dielectric function for {title}", fontsize=fig_setting[3][0])
    # Data plotting
    for supplot_index in range(2):
        ax = axes_element[supplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        ax.set_title(subtitles[supplot_index])

        for _, data in enumerate(dataset):
            # Labels
            if data[0] not in ["", None]:
                current_label = f"({data[0]})"
            else:
                current_label = ""
            # Inplane
            if supplot_index == 0:
                lines_real = ax.plot(data[1]["density_energy_real"], data[1]["density_xx_real"], color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"Real part {current_label}")
                lines_real[0].set_dashes([2, 0])
                lines_imag = ax.plot(data[1]["density_energy_imag"], data[1]["density_xx_imag"], color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"Imaginary part {current_label}")
                lines_imag[0].set_dashes([2, 1])
            # Outplane
            elif supplot_index == 1:
                lines_real = ax.plot(data[1]["density_energy_real"], data[1]["density_zz_real"], color=color_sampling(data[2])[2], alpha=data[3], lw=data[4], label=f"Real part {current_label}")
                lines_real[0].set_dashes([2, 0])
                lines_imag = ax.plot(data[1]["density_energy_imag"], data[1]["density_zz_imag"], color=color_sampling(data[2])[2], alpha=data[3], lw=data[4], label=f"Imaginary part {current_label}")
                lines_imag[0].set_dashes([2, 1])

        if energy_max is not None:
            ax.set_xlim(0, energy_max)
        ax.set_ylabel(r"Dielectric function")
        if supplot_index == 1:
            ax.set_xlabel(r"Photon energy (eV)")
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
