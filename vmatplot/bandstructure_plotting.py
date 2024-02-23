#### Bandstructure for plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0302, C0321, R0913, R0914, R0915, W0612, W0105

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vmatplot.commons import extract_fermi
from vmatplot.output import canvas_setting, color_sampling
from vmatplot.DoS import extract_dos
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
        bstype, label, directory, *optional = current_matter
        if not optional:
            color = "orbital"
            lstyle = "solid"
            alpha = 1.0
        elif len(optional) == 1:
            color = optional[0]
            lstyle = "solid"
            alpha = 1.0
        elif len(optional) == 2:
            color = optional[0]
            lstyle =optional[1]
            alpha = 1.0
        else:
            color, lstyle, alpha = optional[0], optional[1], optional[2]
        # Bandstructure plotting style: monocolor
        if bstype.lower() in ["monocolor", "monocolor nonpolarized"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            bands = extract_eigenvalues_bands_nonpolarized(directory)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, lstyle, alpha])
        elif bstype.lower() in ["monocolor spin up", "spin up monocolor"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            bands = extract_eigenvalues_bands_spinUp(directory)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, lstyle, alpha])
        elif bstype.lower() in ["monocolor spin down", "spin down monocolor"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            bands = extract_eigenvalues_bands_spinDown(directory)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, lstyle, alpha])
        # Bandstructure plotting style: bands
        elif bstype.lower() in ["bands", "bands nonpolarized"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            conduction_bands = extract_eigenvalues_conductionBands_nonpolarized(directory)
            valence_bands = extract_eigenvalues_valenceBands_nonpolarized(directory)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, lstyle, alpha])
        elif bstype.lower() in ["bands spin up", "spin up bands"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            conduction_bands = extract_eigenvalues_conductionBands_spinUp(directory)
            valence_bands = extract_eigenvalues_valenceBands_spinUp(directory)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, lstyle, alpha])
        elif bstype.lower() in ["bands spin down", "spin down bands"]:
            fermi_energy = extract_fermi(directory)
            kpath = extract_kpath(directory)
            conduction_bands = extract_eigenvalues_conductionBands_spinDown(directory)
            valence_bands = extract_eigenvalues_valenceBands_spinDown(directory)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, lstyle, alpha])
    return matters

def create_matters_bsDos(matters_list):
    matters = []
    for current_matter in matters_list:
        bstype, label, bs_dir, dos_dir, *optional = current_matter
        if not optional:
            color = "orbital"
            lstyle = "solid"
            alpha = 1.0
        elif len(optional) == 1:
            color = optional[0]
            lstyle = "solid"
            alpha = 1.0
        elif len(optional) == 2:
            color = optional[0]
            lstyle =optional[1]
            alpha = 1.0
        else:
            color, lstyle, alpha = optional[0], optional[1], optional[2]
        # Bandstructure plotting style: monocolor
        if bstype.lower() in ["monocolor", "monocolor nonpolarized"]:
            fermi_energy = extract_fermi(bs_dir)
            kpath = extract_kpath(bs_dir)
            bands = extract_eigenvalues_bands_nonpolarized(bs_dir)
            dos = extract_dos(dos_dir)
            matters.append([bstype, label, fermi_energy, kpath, bands, dos, color, lstyle, alpha])
        elif bstype.lower() in ["monocolor spin up", "spin up monocolor"]:
            fermi_energy = extract_fermi(bs_dir)
            kpath = extract_kpath(bs_dir)
            bands = extract_eigenvalues_bands_spinUp(bs_dir)
            dos = extract_dos(dos_dir)
            matters.append([bstype, label, fermi_energy, kpath, bands, dos, color, lstyle, alpha])
        elif bstype.lower() in ["monocolor spin down", "spin down monocolor"]:
            fermi_energy = extract_fermi(bs_dir)
            kpath = extract_kpath(bs_dir)
            bands = extract_eigenvalues_bands_spinDown(bs_dir)
            dos = extract_dos(dos_dir)
            matters.append([bstype, label, fermi_energy, kpath, bands, dos, color, lstyle, alpha])
        # Bandstructure plotting style: bands
        elif bstype.lower() in ["bands", "bands nonpolarized"]:
            fermi_energy = extract_fermi(bs_dir)
            kpath = extract_kpath(bs_dir)
            conduction_bands = extract_eigenvalues_conductionBands_nonpolarized(bs_dir)
            valence_bands = extract_eigenvalues_valenceBands_nonpolarized(bs_dir)
            dos = extract_dos(dos_dir)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, dos, color, lstyle, alpha])
        elif bstype.lower() in ["bands spin up", "spin up bands"]:
            fermi_energy = extract_fermi(bs_dir)
            kpath = extract_kpath(bs_dir)
            conduction_bands = extract_eigenvalues_conductionBands_spinUp(bs_dir)
            valence_bands = extract_eigenvalues_valenceBands_spinUp(bs_dir)
            dos = extract_dos(dos_dir)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, dos, color, lstyle, alpha])
        elif bstype.lower() in ["bands spin down", "spin down bands"]:
            fermi_energy = extract_fermi(bs_dir)
            kpath = extract_kpath(bs_dir)
            conduction_bands = extract_eigenvalues_conductionBands_spinDown(bs_dir)
            valence_bands = extract_eigenvalues_valenceBands_spinDown(bs_dir)
            dos = extract_dos(dos_dir)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, dos, color, lstyle, alpha])
    return matters

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
                        plt.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], alpha=matter[8], label=f"Conduction bands for {matter[1]}", zorder=4)
                    else:
                        plt.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], alpha=matter[8], label="Conduction bands", zorder=4)
                else:
                    plt.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], alpha=matter[8], zorder=4)
            for bands_index in range(0, len(matter[5])):
                current_valence_band = [eigenvalue - fermi for eigenvalue in matter[5][bands_index]]
                if bands_index == 0:
                    if matter[1] != "":
                        plt.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], alpha=matter[8], label=f"Valence bands for {matter[1]}", zorder=4)
                    else:
                        plt.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], alpha=matter[8], label="Valence bands", zorder=4)
                else:
                    plt.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], alpha=matter[8], zorder=4)
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

def plot_bsDoS(title, eigen_range=None, dos_range=None, matters_list=None, legend_loc="False"):
    # Figure setting
    fig_setting = canvas_setting(15, 6)
    params = fig_setting[2]; plt.rcParams.update(params)

    fig = plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Colors calling
    bs_fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")

    # Data calling and plotting
    matters = create_matters_bsDos(matters_list)

    # Title
    fig.suptitle(f"Bandstructure and DoS for {title}", fontsize=fig_setting[3][0])

    # ax1 Bandstructure
    ax1.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    ax1.set_title("Bandstructure", fontsize=fig_setting[3][1])

    for matter in matters:
        if matter[0].lower() in ["monocolor"]:
            bs_fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_band = [eigenvalue - bs_fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    if matter[1] != "":
                        ax1.plot(matter[3], current_band, c=color_sampling(matter[6])[1], alpha=matter[7], label=f"Bandstructure for {matter[1]}", zorder=4)
                    else:
                        ax1.plot(matter[3], current_band, c=color_sampling(matter[6])[1], alpha=matter[7], label="Bandstructure", zorder=4)
                else:
                    ax1.plot(matter[3], current_band, c=color_sampling(matter[6])[1], alpha=matter[7], zorder=4)
        elif matter[0] in ["bands"]:
            bs_fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_conduction_band = [eigenvalue - bs_fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    if matter[1] != "":
                        ax1.plot(matter[3], current_conduction_band, c=color_sampling(matter[7])[2], linestyle=matter[8], alpha=matter[9], label=f"Conduction bands for {matter[1]}", zorder=4)
                    else:
                        ax1.plot(matter[3], current_conduction_band, c=color_sampling(matter[7])[2], linestyle=matter[8], alpha=matter[9], label="Conduction bands", zorder=4)
                else:
                    ax1.plot(matter[3], current_conduction_band, c=color_sampling(matter[7])[2], linestyle=matter[8], alpha=matter[9], zorder=4)
            for bands_index in range(0, len(matter[5])):
                current_valence_band = [eigenvalue - bs_fermi for eigenvalue in matter[5][bands_index]]
                if bands_index == 0:
                    if matter[1] != "":
                        ax1.plot(matter[3], current_valence_band, c=color_sampling(matter[7])[0], linestyle=matter[8], alpha=matter[9], label=f"Valence bands for {matter[1]}", zorder=4)
                    else:
                        ax1.plot(matter[3], current_valence_band, c=color_sampling(matter[7])[0], linestyle=matter[8], alpha=matter[9], label="Valence bands", zorder=4)
                else:
                    ax1.plot(matter[3], current_valence_band, c=color_sampling(matter[7])[0], linestyle=matter[8], alpha=matter[9], zorder=4)
        kpath_start = matter[3][0]
        kpath_end = matter[3][-1]
        bs_fermi_last = matter[2]

    # Fermi energy as a horizon line
    ax1.axhline(y = 0, color=bs_fermi_color[0], alpha=1.00, linestyle="--", label="Fermi energy", zorder=2)
    bs_efermi = bs_fermi_last
    kpath_range = kpath_end-kpath_start
    # bs_fermi_energy_text = f"Fermi energy\n{bs_efermi:.3f} (eV)"
    # ax1.text(kpath_start+kpath_range*0.98, eigen_range*0.02, bs_fermi_energy_text, fontsize=10, c=bs_fermi_color[0], rotation=0, va = "bottom", ha="right", zorder=5)

    # y-axis
    ax1.set_ylabel("Energy (eV)")
    ax1.set_ylim(eigen_range*(-1), eigen_range)
    # x-axis
    ax1.set_xlim(kpath_start, kpath_end)

    high_symmetry_paths = kpoints_path(matters_list[-1][2])
    high_symmetry_positions = list(high_symmetry_paths.values())
    # high_symmetry_positions = list(kpoints_path(matters_list[-1][2]).values())

    high_symmetry_positions.append(kpath_end)
    high_symmetry_labels = list(high_symmetry_paths.keys())
    # high_symmetry_labels = list(kpoints_path(matters_list[-1][2]).keys())

    high_symmetry_labels.append(high_symmetry_labels[0])

    ax1.set_xticks(high_symmetry_positions)
    ax1.set_xticklabels(high_symmetry_labels)

    for k_loc in high_symmetry_positions[1:-1]:
        ax1.axvline(x=k_loc, color=annotate_color[1], linestyle="--", zorder=1)

    # ax2 DoS
    ax2.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    ax2.set_title("DoS", fontsize=fig_setting[3][1])
    for matter in matters:
        if matter[0].lower() in ["monocolor"]:
            plt.plot(matter[5][6], matter[5][5], c=color_sampling(matter[6])[1], label=f"Total DoS for {matter[1]}", zorder = 2)
            dos_efermi = matter[5][0]

        elif matter[0] in ["bands"]:
            plt.plot(matter[6][6], matter[6][5], c=color_sampling(matter[7])[1], label=f"Total DoS for {matter[1]}", zorder = 2)
            dos_efermi = matter[6][0]

    ax2.set_ylim(eigen_range*(-1), eigen_range)
    ax2.set_xlim(0, dos_range)

    ax2.axhline(y = 0, color=bs_fermi_color[0], alpha=1.00, linestyle="--", label="Fermi energy", zorder=2)

    if legend_loc not in [None, "False", False]:
        ax1.legend(loc=legend_loc)
        ax2.legend(loc=legend_loc)
    