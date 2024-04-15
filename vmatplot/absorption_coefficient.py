#### Absorption coefficient plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, W0612

import numpy as np
import matplotlib.pyplot as plt

from vmatplot.output import canvas_setting, color_sampling
from vmatplot.algorithms import process_boundary, extract_part, energy_to_wavelength, energy_to_frequency
from vmatplot.dielectric_function_plotting import create_matters_dielectric_function

def cal_absorption_coefficient(frequency, density_energy_real,density_energy_imag):
    absorption = np.sqrt(2)*frequency*(np.sqrt(np.square(density_energy_real)+np.square(density_energy_imag))-density_energy_real)
    return absorption

def create_matters_absorption(*args):
    # data = create_matters_dielectric_function(dielectric_list)
    # data[0] = current curve label
    # data[1] = dielectric data
    # data[2] = color family
    # data[3] = alpha
    # data[4] = linewidth
    return create_matters_dielectric_function(*args)

def plot_absorption_XZ_col(title, absorption_list=None, inplane_boundary=(None, None), outplane_boundary=(None, None)):
    help_info = "Usage: absorption_XZ" + \
                "The independent value includes \n" +\
                "\t title, \n" +\
                "\t dielectric function data list, \n" +\
                "\t Inplane photon wavelenght range (Optional), \n" +\
                "\t Outplane photon wavelenght range (Optional). \n"
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
    dataset = create_matters_absorption(absorption_list)
    subtitles = ["In-plane", "Out-of-plane"]

    # Suptitle
    fig.suptitle(f"Dielectric function for {title}", fontsize=fig_setting[3][0], y=0.96)

    # Boundary
    inplane_start, inplane_end = process_boundary(inplane_boundary)
    outplane_start, outplane_end = process_boundary(outplane_boundary)

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
                inplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                inplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])
                inplane_absorption_full = cal_absorption_coefficient(inplane_frequency_full,data[1]["density_xx_real"],data[1]["density_xx_imag"])
                inplane_wavelength, inplane_absorption = extract_part(inplane_wavelength_full,inplane_absorption_full,inplane_start,inplane_end)
                lines_real = ax.plot(inplane_wavelength,inplane_absorption)

            # Outplane
            elif supplot_index == 1:
                outplane_wavelength_full = energy_to_wavelength(data[1]["density_energy_real"])
                outplane_frequency_full = energy_to_frequency(data[1]["density_energy_real"])
                outplane_absorption_full = cal_absorption_coefficient(outplane_frequency_full,data[1]["density_zz_real"],data[1]["density_zz_imag"])
                outplane_wavelength, outplane_absorption = extract_part(outplane_wavelength_full,outplane_absorption_full,outplane_start,outplane_end)
                lines_real = ax.plot(outplane_wavelength,outplane_absorption)
