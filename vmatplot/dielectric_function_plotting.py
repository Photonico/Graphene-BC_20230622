#### Dielectric function plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import matplotlib.pyplot as plt
import os

from vmatplot.output import canvas_setting, color_sampling
from vmatplot.algorithms import process_boundary, extract_part
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

def plot_dielectric_function_XZ_col(title, dielectric_list=None, inplane_energy_boundary=(None, None), outplane_energy_boundary=(None, None)):
    # Help information
    help_info = "Usage: plot_dielectric_function_XZ" + \
                "The independent value includes \n" +\
                "\t title, \n" +\
                "\t dielectric function data list, \n" +\
                "\t Inplane photon energy range (Optional), \n" +\
                "\t Outplane photon energy range (Optional). \n"
    if title in ["help", "Help"]:
        print(help_info)
    # Figure settings
    fig_setting = canvas_setting(8, 11)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # Colors calling
    annotate_color = color_sampling("Grey")
    order_labels = ["a","b"]

    # Materials information
    dataset = create_matters_dielectric_function(dielectric_list)
    # data_length = len(dataset)
    subtitles = ["In-plane", "Out-of-plane"]

    # Suptitle
    fig.suptitle(f"Dielectric function {title}", fontsize=fig_setting[3][0], y=0.96)
    # fig.suptitle(f"Dielectric function {title}", fontsize=fig_setting[3][0], y=1.00)

    # Data boundary
    inplane_start, inplane_end = process_boundary(inplane_energy_boundary)
    outplane_start, outplane_end = process_boundary(outplane_energy_boundary)

    # Data plotting
    starts_inplane = []; ends_inplane = []; starts_outplane = []; ends_outplane = []
    for supplot_index in range(2):
        ax = axes_element[supplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        ax.set_title(subtitles[supplot_index])

        for index, data in enumerate(dataset):
            # Labels
            current_label = data[0]
            # Inplane
            if supplot_index == 0:
                inplane_energy_real, inplane_density_real = extract_part(data[1]["density_energy_real"],data[1]["density_xx_real"],inplane_start,inplane_end)
                inplane_energy_imag, inplane_density_imag = extract_part(data[1]["density_energy_imag"],data[1]["density_xx_imag"],inplane_start,inplane_end)

                # plasmon resonance line
                starts_inplane.append(inplane_energy_real[0]); ends_inplane.append(inplane_energy_real[-1])
                start_inplane = min(starts_inplane); end_inplane = max(ends_inplane)
                if index == len(dielectric_list)-1:
                    plas_line, = ax.plot([start_inplane,end_inplane],[0,0],color=color_sampling("grey")[1],linestyle="--")

                lines_real = ax.plot(inplane_energy_real, inplane_density_real, color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"Real part {current_label}")
                lines_real[0].set_dashes([2, 0])
                lines_imag = ax.plot(inplane_energy_imag, inplane_density_imag, color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"Imaginary part {current_label}")
                lines_imag[0].set_dashes([2, 1])

            # Outplane
            elif supplot_index == 1:
                outplane_energy_real, outplane_density_real = extract_part(data[1]["density_energy_real"],data[1]["density_zz_real"],outplane_start,outplane_end)
                outplane_energy_imag, outplane_density_imag = extract_part(data[1]["density_energy_imag"],data[1]["density_zz_imag"],outplane_start,outplane_end)

                # plasmon resonance line
                starts_outplane.append(outplane_energy_real[0]); ends_outplane.append(outplane_energy_real[-1])
                start_outplane = min(starts_outplane); end_outplane = max(ends_outplane)
                if index == len(dielectric_list)-1:
                    plas_line, = ax.plot([start_outplane,end_outplane],[0,0],color=color_sampling("grey")[1],linestyle="--")

                lines_real = ax.plot(outplane_energy_real, outplane_density_real, color=color_sampling(data[2])[2], alpha=data[3], lw=data[4], label=f"Real part {current_label}")
                lines_real[0].set_dashes([2, 0])
                lines_imag = ax.plot(outplane_energy_imag, outplane_density_imag, color=color_sampling(data[2])[2], alpha=data[3], lw=data[4], label=f"Imaginary part {current_label}")
                lines_imag[0].set_dashes([2, 1])

        # axis label
        if supplot_index == 1:
            ax.set_xlabel(r"Photon energy (eV)")
        ax.set_ylabel(r"Dielectric function")

        # Legend
        # handles, labels = ax.get_legend_handles_labels()
        # plas_index = handles.index(plas_line)
        # legend_order = [i for i in range(len(handles)) if i != plas_index] + [plas_index]
        # ax.legend([handles[idx] for idx in legend_order], [labels[idx] for idx in legend_order], loc="upper right")
        ax.legend(loc="upper right")

        # Subplots label
        # orderlab_shift = 0.05
        # x_loc = 0+orderlab_shift*0.75
        # y_loc = 1-orderlab_shift
        # ax.annotate(f"({order_labels[supplot_index]})",
        #             xy=(x_loc,y_loc),
        #             xycoords="axes fraction",
        #             fontsize=1.0 * 16,
        #             ha="center", va="center",
        #             bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

def plot_dielectric_function_XZ_row(title, dielectric_list=None, inplane_energy_boundary=(None, None), outplane_energy_boundary=(None, None)):
    # Help information
    help_info = "Usage: plot_dielectric_function_XZ" + \
                "The independent value includes \n" +\
                "\t title, \n" +\
                "\t dielectric function data list, \n" +\
                "\t Inplane photon energy range (Optional), \n" +\
                "\t Outplane photon energy range (Optional). \n"
    if title in ["help", "Help"]:
        print(help_info)
    # Figure settings
    fig_setting = canvas_setting(16, 6)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0], axs[1]]

    # Colors calling
    # annotate_color = color_sampling("Grey")
    # order_labels = ["a","b"]

    # Materials information
    dataset = create_matters_dielectric_function(dielectric_list)
    # data_length = len(dataset)
    subtitles = ["In-plane", "Out-of-plane"]

    # Suptitle
    # fig.suptitle(f"Dielectric function {title}", fontsize=fig_setting[3][0], y=0.90)
    fig.suptitle(f"Dielectric function {title}", fontsize=fig_setting[3][0], y=1.00)

    # Data boundary
    inplane_start, inplane_end = process_boundary(inplane_energy_boundary)
    outplane_start, outplane_end = process_boundary(outplane_energy_boundary)

    # Data plotting
    starts_inplane = []; ends_inplane = []; starts_outplane = []; ends_outplane = []
    for supplot_index in range(2):
        ax = axes_element[supplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        ax.set_title(subtitles[supplot_index])

        for index, data in enumerate(dataset):
            # Labels
            current_label = data[0]
            # Inplane
            if supplot_index == 0:
                inplane_energy_real, inplane_density_xx_real = extract_part(data[1]["density_energy_real"],data[1]["density_xx_real"],inplane_start,inplane_end)
                inplane_energy_imag, inplane_density_xx_imag = extract_part(data[1]["density_energy_imag"],data[1]["density_xx_imag"],inplane_start,inplane_end)

                # plasmon resonance line
                starts_inplane.append(inplane_energy_real[0]); ends_inplane.append(inplane_energy_real[-1])
                start_inplane = min(starts_inplane); end_inplane = max(ends_inplane)
                if index == len(dielectric_list)-1:
                    plas_line, = ax.plot([start_inplane,end_inplane],[0,0],color=color_sampling("grey")[1],linestyle="--")

                lines_real = ax.plot(inplane_energy_real, inplane_density_xx_real, color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"Real part {current_label}")
                lines_real[0].set_dashes([2, 0])
                lines_imag = ax.plot(inplane_energy_imag, inplane_density_xx_imag, color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"Imaginary part {current_label}")
                lines_imag[0].set_dashes([2, 1])
            # Outplane
            elif supplot_index == 1:
                outplane_energy_real, outplane_density_xx_real = extract_part(data[1]["density_energy_real"],data[1]["density_zz_real"],outplane_start,outplane_end)
                outplane_energy_imag, outplane_density_xx_imag = extract_part(data[1]["density_energy_imag"],data[1]["density_zz_imag"],outplane_start,outplane_end)

                # plasmon resonance line
                starts_outplane.append(outplane_energy_real[0]); ends_outplane.append(outplane_energy_real[-1])
                start_outplane = min(starts_outplane); end_outplane = max(ends_outplane)
                if index == len(dielectric_list)-1:
                    plas_line, = ax.plot([start_outplane,end_outplane],[0,0],color=color_sampling("grey")[1],linestyle="--")

                lines_real = ax.plot(outplane_energy_real, outplane_density_xx_real, color=color_sampling(data[2])[2], alpha=data[3], lw=data[4], label=f"Real part {current_label}")
                lines_real[0].set_dashes([2, 0])
                lines_imag = ax.plot(outplane_energy_imag, outplane_density_xx_imag, color=color_sampling(data[2])[2], alpha=data[3], lw=data[4], label=f"Imaginary part {current_label}")
                lines_imag[0].set_dashes([2, 1])

        # axis label
        if supplot_index == 0:
            ax.set_ylabel(r"Dielectric function")
        ax.set_xlabel(r"Photon energy (eV)")

        # Legend
        # handles, labels = ax.get_legend_handles_labels()
        # plas_index = handles.index(plas_line)
        # legend_order = [i for i in range(len(handles)) if i != plas_index] + [plas_index]
        # ax.legend([handles[idx] for idx in legend_order], [labels[idx] for idx in legend_order], loc="upper right")
        ax.legend(loc="upper right")

        # Subplots label
        # orderlab_shift = 0.05
        # x_loc = 0+orderlab_shift*0.75
        # y_loc = 1-orderlab_shift
        # ax.annotate(f"({order_labels[supplot_index]})",
        #             xy=(x_loc,y_loc),
        #             xycoords="axes fraction",
        #             fontsize=1.0 * 16,
        #             ha="center", va="center",
        #             bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})

    plt.tight_layout()

#

def plot_dielectric_function_XZ_block(title, dielectric_list=None, inplane_energy_boundary=(None, None), outplane_energy_boundary=(None, None)):
    # Help information
    help_info = "Usage: plot_dielectric_function_XZ_block" + \
                "The independent value includes \n" +\
                "\t title, \n" +\
                "\t dielectric function data list, \n" +\
                "\t Inplane photon energy range (Optional), \n" +\
                "\t Outplane photon energy range (Optional). \n"
    if title in ["help", "Help"]:
        print(help_info)
    # Figure settings
    fig_setting = canvas_setting(16, 12)
    params = fig_setting[2]; plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 2, figsize=fig_setting[0], dpi=fig_setting[1])
    axes_element = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

    # Colors calling
    # annotate_color = color_sampling("Grey")
    # order_labels = ["a","b","c","d"]

    # Materials information
    dataset = create_matters_dielectric_function(dielectric_list)
    # data_length = len(dataset)
    subtitles = ["In-plane real part", "Out-of-plane real part", "In-plane imaginary part", "Out-of-plane imaginary part"]

    # Suptitle
    # fig.suptitle(f"Dielectric function {title}", fontsize=fig_setting[3][0], y=1.00)
    fig.suptitle(f"Dielectric function {title}", fontsize=fig_setting[3][0], y=1.00)

    # Data boundary
    inplane_start, inplane_end = process_boundary(inplane_energy_boundary)
    outplane_start, outplane_end = process_boundary(outplane_energy_boundary)

    # Data plotting
    starts_inplane = []; ends_inplane = []; starts_outplane = []; ends_outplane = []
    for supplot_index in range(4):
        ax = axes_element[supplot_index]
        ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
        ax.set_title(subtitles[supplot_index])

        # Dielectric function = 0
        for index, data in enumerate(dataset):
            # Labels
            current_label = data[0]
            # Inplane real part
            if supplot_index == 0:
                inplane_energy_real, inplane_density_xx_real = extract_part(data[1]["density_energy_real"],data[1]["density_xx_real"],inplane_start,inplane_end)

                # plasmon resonance line
                starts_inplane.append(inplane_energy_real[0]); ends_inplane.append(inplane_energy_real[-1])
                start_inplane = min(starts_inplane); end_inplane = max(ends_inplane)
                if index == len(dielectric_list)-1:
                    plas_line, = ax.plot([start_inplane,end_inplane],[0,0],color=color_sampling("grey")[1],linestyle="--")

                lines_real = ax.plot(inplane_energy_real, inplane_density_xx_real, color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"Real part {current_label}")
                lines_real[0].set_dashes([2, 0])

            # Outplane real part
            elif supplot_index == 1:
                outplane_energy_real, outplane_density_xx_real = extract_part(data[1]["density_energy_real"],data[1]["density_zz_real"],outplane_start,outplane_end)

                # plasmon resonance line
                starts_outplane.append(outplane_energy_real[0]); ends_outplane.append(outplane_energy_real[-1])
                start_outplane = min(starts_outplane); end_outplane = max(ends_outplane)
                if index == len(dielectric_list)-1:
                    # print(start_outplane,end_outplane)
                    plas_line, = ax.plot([start_outplane,end_outplane],[0,0],color=color_sampling("grey")[1],linestyle="--")

                lines_real = ax.plot(outplane_energy_real, outplane_density_xx_real, color=color_sampling(data[2])[1], alpha=data[3], lw=data[4], label=f"Real part {current_label}")
                lines_real[0].set_dashes([2, 0])

            # Inplane imag part
            elif supplot_index == 2:
                inplane_energy_imag, inplane_density_xx_imag = extract_part(data[1]["density_energy_imag"],data[1]["density_xx_imag"],inplane_start,inplane_end)

                lines_imag = ax.plot(inplane_energy_imag, inplane_density_xx_imag, color=color_sampling(data[2])[2], alpha=data[3], lw=data[4], label=f"Imaginary part {current_label}")
                lines_imag[0].set_dashes([2, 0])

            # Outplane imag part
            elif supplot_index == 3:
                outplane_energy_imag, outplane_density_xx_imag = extract_part(data[1]["density_energy_imag"],data[1]["density_zz_imag"],outplane_start,outplane_end)

                lines_imag = ax.plot(outplane_energy_imag, outplane_density_xx_imag, color=color_sampling(data[2])[2], alpha=data[3], lw=data[4], label=f"Imaginary part {current_label}")
                lines_imag[0].set_dashes([2, 0])

        # axis label
        if supplot_index in [0,2]:
            ax.set_ylabel(r"Dielectric function")
        if supplot_index in [2,3]:
            ax.set_xlabel(r"Photon energy (eV)")

        # Legend
        ax.legend(loc="upper right")

        # Subplots label
        # orderlab_shift = 0.05
        # x_loc = 0+orderlab_shift*0.75
        # y_loc = 1-orderlab_shift

        # ax.annotate(f"({order_labels[supplot_index]})",
        #             xy=(x_loc,y_loc),
        #             xycoords="axes fraction",
        #             fontsize=1.0 * 16,
        #             ha="center", va="center",
        #             bbox = {"facecolor": "white", "alpha": 0.75, "edgecolor": annotate_color[2], "linewidth": 1.5, "boxstyle": "round, pad=0.2"})
    plt.tight_layout()

def plot_dielectric_function_XZ(*args):
    # return plot_dielectric_function_XZ_col(*args)
    return plot_dielectric_function_XZ_row(*args)
    return plot_dielectric_function_XZ_zoom(*args)
    # return plot_dielectric_function_XZ_block(*args)
