#### Bandstructure for plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0302, C0321, R0913, R0914, R0915, W0612, W0105

import matplotlib.pyplot as plt

from vmatplot.commons import extract_fermi
from vmatplot.output import canvas_setting, color_sampling
from vmatplot.bandstructure import extract_kpath, kpoints_path
from vmatplot.bandstructure import extract_eigenvalues_bands_nonpolarized
from vmatplot.bandstructure import extract_eigenvalues_bands_spinUp, extract_eigenvalues_bands_spinDown
from vmatplot.bandstructure import extract_eigenvalues_conductionBands_nonpolarized
from vmatplot.bandstructure import extract_eigenvalues_valenceBands_nonpolarized
from vmatplot.bandstructure import extract_eigenvalues_conductionBands_spinUp
from vmatplot.bandstructure import extract_eigenvalues_valenceBands_spinUp
from vmatplot.bandstructure import extract_eigenvalues_conductionBands_spinDown
from vmatplot.bandstructure import extract_eigenvalues_valenceBands_spinDown

def create_matters_bs(matters_list):
    matters = []
    for current_matter in matters_list:
        bstype, label, directory, color, *optional = current_matter
        alpha = optional[0] if optional else 1.0
        # Bandstructure plotting style: monocolor
        if bstype.lower() in ["monocolor", "monocolor nonpolarized"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            bands = extract_eigenvalues_bands_nonpolarized(directory)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, alpha])
        elif bstype.lower() in ["monocolor spin up", "spin up monocolor"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            bands = extract_eigenvalues_bands_spinUp(directory)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, alpha])
        elif bstype.lower() in ["monocolor spin down", "spin down monocolor"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            bands = extract_eigenvalues_bands_spinDown(directory)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, alpha])
        # Bandstructure plotting style: bands
        elif bstype.lower() in ["bands", "bands nonpolarized"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            conduction_bands = extract_eigenvalues_conductionBands_nonpolarized(directory)
            valence_bands = extract_eigenvalues_valenceBands_nonpolarized(directory)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, alpha])
        elif bstype.lower() in ["bands spin up", "spin up bands"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            conduction_bands = extract_eigenvalues_conductionBands_spinUp(directory)
            valence_bands = extract_eigenvalues_valenceBands_spinUp(directory)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, alpha])
        elif bstype.lower() in ["bands spin down", "spin down bands"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            conduction_bands = extract_eigenvalues_conductionBands_spinDown(directory)
            valence_bands = extract_eigenvalues_valenceBands_spinDown(directory)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, alpha])
    return matters

def create_matters_bsDos(matters_list):
    matters = []
    for current_matter in matters_list:
        bstype, label, directory, color, *optional = current_matter
        alpha = optional[0] if optional else 1.0
        # Bandstructure plotting style: monocolor
        if bstype.lower() in ["monocolor", "monocolor nonpolarized"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            bands = extract_eigenvalues_bands_nonpolarized(directory)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, alpha])
        elif bstype.lower() in ["monocolor spin up", "spin up monocolor"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            bands = extract_eigenvalues_bands_spinUp(directory)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, alpha])
        elif bstype.lower() in ["monocolor spin down", "spin down monocolor"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            bands = extract_eigenvalues_bands_spinDown(directory)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, alpha])
        # Bandstructure plotting style: bands
        elif bstype.lower() in ["bands", "bands nonpolarized"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            conduction_bands = extract_eigenvalues_conductionBands_nonpolarized(directory)
            valence_bands = extract_eigenvalues_valenceBands_nonpolarized(directory)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, alpha])
        elif bstype.lower() in ["bands spin up", "spin up bands"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            conduction_bands = extract_eigenvalues_conductionBands_spinUp(directory)
            valence_bands = extract_eigenvalues_valenceBands_spinUp(directory)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, alpha])
        elif bstype.lower() in ["bands spin down", "spin down bands"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            conduction_bands = extract_eigenvalues_conductionBands_spinDown(directory)
            valence_bands = extract_eigenvalues_valenceBands_spinDown(directory)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, alpha])
    return matters

"""
# Plot bandstructure for a list of matters with customize style 
def plot_bandstructure

# Plot bandstructure for a list of matters and DoS with customize style 
def plot_bandstructure_DoS

# Plot bandstructure for a list of matters and two DoS frames with customize style 
def plot_bandstructure_duoDoS

# Plot bandstructure for a list of matters and PDoS with customize style 
def plot_bandstructure_PDoS

# Plot bandstructure for a list of matters and two PDoS frames of with customize style 
def_plot_bandstructure_duoPDoS
"""

def plot_bandstructure(title, eigen_range=None, matters_list=None, legend_loc="False"):
    # Help information
    help_info = """
    Usage: plot_bandstructure
        arg[0]: title;
        arg[1]: the range of eigenvalues, from -arg[1] to arg[1];
        arg[2]: matters list;
        arg[3]: legend location;
    """
    if title in ["help", "Help"]:
        print(help_info)

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")

    # Data calling and plotting
    matters = create_matters_bs(matters_list)
    for matter in matters:
        if matter[0].lower() in ["monocolor"]:
            fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_band = [eigenvalue - fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    if matter[1] != "":
                        plt.plot(matter[3], current_band, c=color_sampling(matter[5])[1], alpha=matter[6], label=f"Bandstructure for {matter[1]}", zorder=4)
                    else:
                        plt.plot(matter[3], current_band, c=color_sampling(matter[5])[1], alpha=matter[6], label="Bandstructure", zorder=4)
                else:
                    plt.plot(matter[3], current_band, c=color_sampling(matter[5])[1], alpha=matter[6], zorder=4)
        elif matter[0] in ["bands"]:
            fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_conduction_band = [eigenvalue - fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    if matter[1] != "":
                        plt.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], alpha=matter[7], label=f"Conduction bands for {matter[1]}", zorder=4)
                    else:
                        plt.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], alpha=matter[7], label="Conduction bands", zorder=4)
                else:
                    plt.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], alpha=matter[7], zorder=4)
            for bands_index in range(0, len(matter[5])):
                current_valence_band = [eigenvalue - fermi for eigenvalue in matter[5][bands_index]]
                if bands_index == 0:
                    if matter[1] != "":
                        plt.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], alpha=matter[7], label=f"Valence bands for {matter[1]}", zorder=4)
                    else:
                        plt.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], alpha=matter[7], label="Valence bands", zorder=4)
                else:
                    plt.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], alpha=matter[7], zorder=4)
        kpath_start = matter[3][0]
        kpath_end = matter[3][-1]
        fermi_last = matter[2]

    # Fermi energy as a horizon line
    plt.axhline(y = 0, color=fermi_color[0], alpha=1.00, linestyle="--", label="Fermi energy", zorder=2)
    efermi = fermi_last
    kpath_range = kpath_end-kpath_start
    # fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
    # plt.text(kpath_start+kpath_range*0.98, eigen_range*0.02, fermi_energy_text, fontsize=10, c=fermi_color[0], rotation=0, va = "bottom", ha="right", zorder=5)

    # Title
    plt.title(f"Bandstructure for {title}")
    plt.ylabel("Energy (eV)")
    # plt.ylabel("$E-E_\text{F}$ (eV)")

    # y-axis range
    plt.ylim(eigen_range*(-1), eigen_range)

    # x-axis ticks

    plt.xlim(kpath_start, kpath_end)

    high_symmetry_paths = kpoints_path(matters_list[-1][2])
    high_symmetry_positions = list(high_symmetry_paths.values())
    # high_symmetry_positions = list(kpoints_path(matters_list[-1][2]).values())

    high_symmetry_positions.append(kpath_end)
    high_symmetry_labels = list(high_symmetry_paths.keys())
    # high_symmetry_labels = list(kpoints_path(matters_list[-1][2]).keys())

    high_symmetry_labels.append(high_symmetry_labels[0])
    plt.xticks(high_symmetry_positions, high_symmetry_labels)
    for k_loc in high_symmetry_positions[1:-1]:
        plt.axvline(x=k_loc, color=annotate_color[1], linestyle="--", zorder=1)

    # Legend
    if legend_loc is None:
        legend = plt.legend()
        legend.set_visible(False)
    else:
        legend = plt.legend(loc=legend_loc)
