#### Absorption coefficient plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, W0612

import matplotlib.pyplot as plt

from vmatplot.output import canvas_setting, color_sampling
from vmatplot.algorithms import process_boundary, extract_part
from vmatplot.dielectric_function_plotting import create_matters_dielectric_function

# Physical constants
hbar_ev = 4.135667662e-15
c_vacuum = 2.99792458e8
c_vacuum_nm = 2.99792458e17
pi = 3.141592654

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
    dataset_source = create_matters_absorption(absorption_list)
    subtitles = ["In-plane", "Out-of-plane"]

    # Suptitle
    fig.suptitle(f"Dielectric function for {title}", fontsize=fig_setting[3][0], y=0.96)

    # Boundary
    inplane_start, inplane_end = process_boundary(inplane_boundary)
    outplane_start, outplane_end = process_boundary(outplane_boundary)
