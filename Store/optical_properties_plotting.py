#### Ploting for Charge Density Response for Specific NBANDS
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

### Import the necessary packages and data for plotting
import matplotlib.pyplot as plt
import numpy as np

params = {"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
          "axes.titlesize": 18, "axes.labelsize": 12, "figure.facecolor": "w"}
plt.rcParams.update(params)

### Plotting settings
def set_plot_style(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

def plot_range_setting(opt_data):
    max_value = np.max([opt_data[f"density_{ax}_{comp}"] for ax in ["xx", "yy", "zz"] for comp in ["imag", "real"]])
    min_value = np.min([opt_data[f"density_{ax}_{comp}"] for ax in ["xx", "yy", "zz"] for comp in ["imag", "real"]])
    x_range = np.max([opt_data["density_energy_real"], opt_data["density_energy_imag"],])
    x_left = -x_range * 0.01
    x_right = x_range * 0.25
    return min_value * 1.025, max_value * 1.025, x_left, x_right

### Charge density response
def density_data(ax, energy_data_real, density_data_real, energy_data_imag, density_data_imag, color, linewidth, subtitle_label, title):
    set_plot_style(ax)
    lines_real = ax.plot(energy_data_real, density_data_real, c=color, linewidth=linewidth, label= "Real part "+ subtitle_label)
    lines_real[0].set_dashes([2, 0])
    lines_imag = ax.plot(energy_data_imag, density_data_imag, c=color, linewidth=linewidth, label= "Imaginary part "+ subtitle_label)
    lines_imag[0].set_dashes([2, 1])
    ax.set_title(title)
    ax.legend(loc="best")

def density_plotting(opt_data, matter, y_bot, y_top):
    # y_range_min, y_range_max, x_range_left, x_range_right = plot_range_setting(opt_data)
    fig, axs = plt.subplots(1, 2, dpi=196, figsize=(14.4, 6.4))
    # fig.suptitle(f"Charge density response for {matter} with {str(NBANDS)} vacant energy bands")
    fig.suptitle(f"Charge density response for {matter}", fontsize =1.0*18)
    axs[0].set_xlabel(r"Energy (eV)", fontsize =1.0* 12); axs[0].set_ylabel(r"$\varepsilon(\omega)$", fontsize =1.0* 16)
    axs[1].set_xlabel(r"Energy (eV)", fontsize =1.0* 12); axs[1].set_ylabel(r"$\varepsilon(\omega)$", fontsize =1.0* 16)

    density_data(axs[0], opt_data["density_energy_real"], opt_data["density_xx_real"], opt_data["density_energy_imag"], opt_data["density_xx_imag"], "#1473D2", 1.5, "", "In Plane")
    density_data(axs[1], opt_data["density_energy_real"], opt_data["density_zz_real"], opt_data["density_energy_imag"], opt_data["density_zz_imag"], "#AF5AE1", 1.5, "", "Out Plane")

    for ax in axs:
        # ax.set_ylim(y_range_min, y_range_max)
        # ax.set_xlim(x_range_left, x_range_right)
        # ax.set_ylim(-10, 25)
        ax.set_ylim(y_bot, y_top)
        ax.set_xlim(0, 6)
    
    plt.tight_layout()
    # plt.savefig(f"Figures/optical_{Matter}.svg")
    # plt.show()

### Charge density response inplane
def density_plotting_in(opt_data, matter, y_bot, y_top):

    plt.figure(dpi=196, figsize=(9.6, 6.4))
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    plt.title(f"Charge density response for {matter} inplane", fontsize =1.0*18)

    line_real, = plt.plot(opt_data["density_energy_real"], opt_data["density_xx_real"], color="#1473D2", label="Real part")
    line_imag, = plt.plot(opt_data["density_energy_imag"], opt_data["density_xx_imag"], color="#1473D2", label="Imaginary part")
    line_imag.set_dashes([2, 1])

    plt.ylabel(r"Absorption Coefficient", fontsize =1.0* 12)
    plt.xlabel(r"Energy (eV)", fontsize =1.0* 12)
    # plt.ylim(0, 40)
    plt.ylim(y_bot, y_top)
    plt.xlim(0, 6)
    plt.legend(handles=[line_real, line_imag], loc="best")

### Charge density response outplane
def density_plotting_out(opt_data, matter, y_bot, y_top):

    plt.figure(dpi=196, figsize=(9.6, 6.4))
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    plt.title(f"Charge density response for {matter} outplane", fontsize =1.0*18)

    line_real, = plt.plot(opt_data["density_energy_real"], opt_data["density_zz_real"], color="#AF5AE1", label="Real part")
    line_imag, = plt.plot(opt_data["density_energy_imag"], opt_data["density_zz_imag"], color="#AF5AE1", label="Imaginary part")
    line_imag.set_dashes([2, 1])

    plt.ylabel(r"Absorption Coefficient", fontsize =1.0* 12)
    plt.xlabel(r"Energy (eV)", fontsize =1.0* 12)
    # plt.ylim(0, 40)
    plt.ylim(y_bot, y_top)
    plt.xlim(0, 6)
    plt.legend(handles=[line_real, line_imag], loc="best")

### Absorption coefficient
def absorption_data(ax, energy_data_real, density_data_real, energy_data_imag, density_data_imag, color, linewidth, subtitle_label, title):
    set_plot_style(ax)
    absorption_coefficient = np.sqrt(2)*energy_data_real*(np.sqrt(density_data_real**2+density_data_imag**2)-density_data_real)
    lines_abs = ax.plot(energy_data_imag, absorption_coefficient, c=color, linewidth=linewidth, label= "Absorption Coefficient"+ subtitle_label)
    lines_abs[0].set_dashes([2, 0])
    ax.set_title(title)
    ax.legend(loc='best')

def absorption_plotting(opt_data, matter, y_bot, y_top):
    # y_range_min, y_range_max, x_range_left, x_range_right = plot_range_setting(opt_data)
    fig, axs = plt.subplots(1, 2, dpi=196, figsize=(14.4, 6.4))
    fig.suptitle(f"Absorption coefficient for {matter}", fontsize =1.0*18)
    axs[0].set_xlabel(r"Energy (eV)", fontsize =1.0* 12); axs[0].set_ylabel(r"Absorption Coefficient", fontsize =1.0* 12)
    axs[1].set_xlabel(r"Energy (eV)", fontsize =1.0* 12); axs[1].set_ylabel(r"Absorption Coefficient", fontsize =1.0* 12)

    absorption_data(axs[0], opt_data["density_energy_real"], opt_data["density_xx_real"], opt_data["density_energy_imag"], opt_data["density_xx_imag"], "#1473D2", 1.5, "", "In Plane")
    absorption_data(axs[1], opt_data["density_energy_real"], opt_data["density_zz_real"], opt_data["density_energy_imag"], opt_data["density_zz_imag"], "#AF5AE1", 1.5, "", "Out Plane")

    for ax in axs:
        # ax.set_ylim(y_range_min, y_range_max)
        # ax.set_xlim(x_range_left, x_range_right)
        # ax.set_ylim(0, 40)
        ax.set_ylim(y_bot, y_top)
        ax.set_xlim(0, 6)

    plt.tight_layout()
    # plt.savefig(f"Figures/optical_{Matter}.svg")
    # plt.show()

### Absorption coefficient inplane
def absorption_in(opt_data, matter, y_bot, y_top):

    plt.figure(dpi=196, figsize=(9.6, 6.4))
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    plt.title(f"Absorption coefficient for {matter} inplane", fontsize =1.0*18)

    absorption_coefficient = np.sqrt(2)*opt_data["density_energy_real"]*(np.sqrt(opt_data["density_xx_real"]**2+opt_data["density_xx_imag"]**2)-opt_data["density_xx_real"])
    plt.plot(opt_data["density_energy_real"], absorption_coefficient, color="#1473D2")

    plt.ylabel(r"Absorption Coefficient", fontsize =1.0* 12)
    plt.xlabel(r"Energy (eV)", fontsize =1.0* 12)
    # plt.ylim(0, 40)
    plt.ylim(y_bot, y_top)
    plt.xlim(0, 6)

### Absorption coefficient outplane
def absorption_out(opt_data, matter, y_bot, y_top):

    plt.figure(dpi=196, figsize=(9.6, 6.4))
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    plt.title(f"Absorption coefficient for {matter} outplane", fontsize =1.0*18)

    absorption_coefficient = np.sqrt(2)*opt_data["density_energy_real"]*(np.sqrt(opt_data["density_zz_real"]**2+opt_data["density_zz_imag"]**2)-opt_data["density_zz_real"])
    plt.plot(opt_data["density_energy_real"], absorption_coefficient, color="#AF5AE1")

    plt.ylabel(r"Absorption Coefficient", fontsize =1.0* 12)
    plt.xlabel(r"Energy (eV)", fontsize =1.0* 12)
    # plt.ylim(0, 40)
    plt.ylim(y_bot, y_top)
    plt.xlim(0, 6)

def absorption_inplane_comparing(matter_1, opt_data_1, matter_2, opt_data_2,  matter_3, opt_data_3, y_bot, y_top):
    # matter_1 means the double layer which is compared with single layer matter_2 and matter_3

    plt.figure(dpi=196, figsize=(9.6, 6.4))
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    plt.title(f"Absorption coefficient for {matter_1}", fontsize =1.0*18)

    absorption_coefficient = {}
    for i, opt_data in enumerate([opt_data_1, opt_data_2, opt_data_3], start=1):
        absorption_coefficient[i] = np.sqrt(2)*opt_data["density_energy_real"]*(np.sqrt(opt_data["density_xx_real"]**2+opt_data["density_xx_imag"]**2)-opt_data["density_xx_real"])

    plt.plot(opt_data_1["density_energy_real"], absorption_coefficient[1], color="#AF5AE1", label=f"{matter_1}", zorder = 2)
    plt.plot(opt_data_2["density_energy_real"], absorption_coefficient[2], color="#1473D2", label=f"{matter_2}", zorder = 1)
    plt.plot(opt_data_3["density_energy_real"], absorption_coefficient[3], color="#288C3C", label=f"{matter_3}", zorder = 1)

    plt.ylabel(r"Absorption Coefficient", fontsize =1.0* 12)
    plt.xlabel(r"Energy (eV)", fontsize =1.0* 12)
    # plt.ylim(0, 40)
    plt.ylim(y_bot, y_top)
    plt.xlim(0, 6)

    plt.legend(loc="best")
