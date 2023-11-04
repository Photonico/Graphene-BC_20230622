##### Data Process and Ploting for DOS
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915

### Import the necessary packages and data for plotting
import matplotlib.pyplot as plt

params = {"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
          "axes.titlesize": 18, "axes.labelsize": 12, "figure.facecolor": "w"}
plt.rcParams.update(params)

def set_plot_style():
    plt.figure(dpi=256, figsize=(9.6, 6.4))
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

def plot_dos(pos_data, matter, y_top):
    # Plotting settings
    set_plot_style()

    # Data plotting range
    # y_axis_top = max(pos_data[6]); y_limit = y_axis_top * 0.6
    y_limit = y_top
    x_range = 6

    # Data plotting
    plt.plot(pos_data[5], pos_data[6], c="#1473D2", label="Total DOS")
    # plt.plot(pos_data[5], pos_data[7], c="#8C64F0", label="Integrated DOS")

    # Plot Fermi energy as a vertical line
    efermi = pos_data[0]
    shift = efermi
    plt.axvline(x = efermi-shift, linestyle="--", color="#F5820F", alpha=0.95, label="Fermi energy")
    fermi_energy_text = f"Fermi energy: {efermi:.3f} eV"
    plt.text(efermi-shift-x_range*0.02, y_limit*0.98, fermi_energy_text, fontsize =1.0*12, color="#EB731E", rotation=0, ha="right")

    # Title
    plt.title(f"Electronic density of state for {matter}")
    plt.ylabel(r"Density of States", fontsize =1.0* 12)
    plt.xlabel(r"Energy (eV)", fontsize =1.0* 12)

    # y_axis_top = max(max(total_dos_list), max(integrated_dos_list))
    plt.xlim(-x_range, x_range)
    plt.ylim(0, y_limit)
    plt.legend(loc="best")
    # plt.show()
