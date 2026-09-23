"""Band structures and projected DoS from the original local VASP outputs."""
import json
import h5py
import numpy as np
from matplotlib.lines import Line2D
from style import *

SOURCES = {}

BAND_COLORS = ["#14A0FF", "#145AAA", "#FFA03C", "#EB731E"]
PDOS_COLORS = [BLUE, "#8C64F0", "#D25ADC", "#F03C64", ORANGE]


def bands(folder):
    path = ROOT / "3_Bandstructure" / folder
    suffix = "_kpoints_opt" if (path / "KPOINTS_OPT").exists() else ""
    with h5py.File(path / "vaspout.h5") as f:
        group = f["results/electron_eigenvalues" + suffix]
        energy = group["eigenvalues"][0].T
        k = group["kpoint_coords"][:]
        fermi = float(f["results/electron_dos" + suffix + "/efermi"][()])
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(k, axis=0), axis=1))]
    distance /= distance[-1]  # Retain the original fractional-coordinate path.
    count = len(k) // 3
    ticks = distance[[0, count - 1, 2 * count - 1, len(k) - 1]]
    # The original plotting code classified whole bands using these OUTCAR edges.
    lines = (path / "OUTCAR").read_text().splitlines()
    homo = int(float(next(s for s in lines if "NELECT" in s).split()[2]) / 2)
    nk = int(next(s for s in lines if "NKPTS" in s).split()[3])
    high = max(float(s.split()[1]) for s in [s for s in lines if f"     {homo}     " in s][:nk])
    low = min(float(s.split()[1]) for s in [s for s in lines if f"     {homo+1}     " in s][:nk])
    SOURCES[folder] = {"file": str((path / "vaspout.h5").relative_to(ROOT)), "group": "results/electron_eigenvalues" + suffix, "fermi_eV": fermi, "bands_shape": list(energy.shape), "k_ticks_normalized": ticks.tolist(), "classification_tolerance_eV": .8}
    return distance, energy - fermi, ticks, energy.min(axis=1) >= low - .8, energy.max(axis=1) <= high + .8


def dos(folder, projected_grid=False):
    path = ROOT / "4_PDoS" / folder
    suffix = "_kpoints_opt" if (path / "KPOINTS_OPT").exists() and not projected_grid else ""
    with h5py.File(path / "vaspout.h5") as f:
        group = f["results/electron_dos" + suffix]
        fermi = float(group["efermi"][()])
        energy, total = group["energies"][:] - fermi, group["dos"][0]
        # All curves use the same grid, energy array and Fermi energy.
        projected = group["dospar"][0]
    SOURCES[folder + suffix] = {"file": str((path / "vaspout.h5").relative_to(ROOT)), "group": "results/electron_dos" + suffix, "fermi_eV": fermi, "energy_points": len(energy), "projected_shape": list(projected.shape)}
    return energy, total, projected, fermi


def draw_bands(ax, prefix):
    for i, functional in enumerate(["PBE", "HSE"]):
        x, energy, ticks, conduction, valence = bands(prefix + functional)
        for mask, color in [(conduction, BAND_COLORS[2*i]), (valence, BAND_COLORS[2*i+1])]:
            ax.plot(x, energy[mask].T, color=color)
    ax.set(xlim=(0, 1), ylim=(-6, 6), yticks=np.arange(-6, 7, 2),
           xticks=ticks, xticklabels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])
    for x in ticks[1:-1]: ax.axvline(x, color=GREY, ls="--", zorder=0)
    ax.axhline(0, color=FERMI, ls="--", zorder=0)


if __name__ == "__main__":
    handles = [Line2D([], [], color=c) for c in BAND_COLORS] + [Line2D([], [], color=FERMI, ls="--")]
    labels = ["Conduction\n(PBE)", "Valence\n(PBE)", "Conduction\n(HSE06)", "Valence\n(HSE06)", "Fermi energy"]
    fig, axes = grid(2, 2, 6.4, right_legend=True)
    for i, (ax, prefix, label) in enumerate(zip(axes.flat,
            ["D_Graphene_", "B_Borophene_", "A_BC3_", "C_B4C3_"],
            ["Graphene", "Borophene", r"BC$_3$", r"B$_4$C$_3$"])):
        draw_bands(ax, prefix)
        title(ax, f"({chr(97+i)}) " + label)
        if i % 2 == 0: ax.set_ylabel("Energy (eV)")
        else: ax.tick_params(labelleft=False)
        if i >= 2: ax.set_xlabel(r"Wave vector ($k$)")
    legend(fig, handles, labels, right=True)
    save(fig, "proj1.3_bands.pdf")

    for name, prefix, dprefix, label in [
        ("proj1.5.pdf", "E_Graphene-BC3_hollow_", "E_Graphene-BC3_", r"Graphene-BC$_3$ (Hollow)"),
        ("proj1.6.pdf", "F_Graphene-Borophene_top_", "F_Graphene-Borophene_", "Graphene-Borophene (Top)"),
        ("proj1.7.pdf", "G_Graphene-B4C3_top_", "G_Graphene-B4C3_", r"Graphene-B$_4$C$_3$ (Top)"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True,
                                 gridspec_kw={"width_ratios": [3, 1]})
        fig.subplots_adjust(left=.09, right=.72, bottom=.15, top=.90, wspace=.15)
        draw_bands(axes[0], prefix)
        for i, functional in enumerate(["PBE", "HSE"]):
            folder = dprefix + functional + ("_K33" if prefix.startswith("G_") and i == 0 else "")
            energy, total, _, _ = dos(folder)
            for mask, color in [(energy > 0, BAND_COLORS[2*i]), (energy < 0, BAND_COLORS[2*i+1])]:
                axes[1].plot(total[mask], energy[mask], color=color)
        axes[0].set(xlabel=r"Wave vector ($k$)", ylabel="Energy (eV)")
        axes[1].set(xlim=(0, 10), xticks=[0, 5, 10], xlabel="DoS")
        axes[1].axhline(0, color=FERMI, ls="--", zorder=0)
        title(axes[0], "(a) Band structure"); title(axes[1], "(b) DoS")
        fig.suptitle(label, fontsize=16, y=.99)
        legend(fig, handles, labels, right=True)
        save(fig, name)

    for name, folder, material, segments, ymax in [
        ("S1.9.pdf", "F_Graphene-Borophene_HSE", "Borophene", [(8,16),(0,8)], 4),
        ("S1.10.pdf", "E_Graphene-BC3_HSE", r"BC$_3$", [(8,16),(0,2),(2,8)], 3),
        ("S1.11.pdf", "G_Graphene-B4C3_HSE", r"B$_4$C$_3$", [(7,15),(0,4),(4,7)], 2.5),
    ]:
        energy, total, projected, fermi = dos(folder, projected_grid=True)
        fig, axes = grid(2, 2, 6.4, right_legend=True)
        titles = ["Total DoS + projections", "Graphene: C", material + ": B", material + ": C"]
        for i, ax in enumerate(axes.flat):
            if i > len(segments): ax.axis("off"); continue
            partial = projected.sum(axis=0) if i == 0 else projected[slice(*segments[i-1])].sum(axis=0)
            sums = total if i == 0 else partial.sum(axis=0)
            for values, color in zip([sums, partial[0], partial[3], partial[1], partial[2]], PDOS_COLORS):
                ax.plot(energy, values, color=color)
            ax.axvline(0, color=FERMI, ls="--", zorder=0)
            ax.set(xlim=(-6,6), ylim=(0,10 if i == 0 else ymax), xticks=np.arange(-6,7,3))
            title(ax, f"({chr(97+i)}) " + titles[i])
            if i % 2 == 0: ax.set_ylabel("Density of states")
            if i >= 2 or (len(segments) == 2 and i == 1): ax.set_xlabel("Energy (eV)")
        pdos_handles = [Line2D([], [], color=c) for c in PDOS_COLORS] + [handles[-1]]
        legend(fig, pdos_handles, ["Total", r"$s$", r"$p_x$", r"$p_y$", r"$p_z$", "Fermi energy"], right=True)
        save(fig, name)

    (HERE / "electronic_sources.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
