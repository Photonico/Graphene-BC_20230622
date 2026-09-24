"""Optical figures directly from local VASP dielectric tensors; no symmetrization."""
from functools import lru_cache
import xml.etree.ElementTree as ET
import h5py
import json
import numpy as np
from matplotlib.lines import Line2D
from style import *

SOURCES = {}

HBAR = 6.582119569e-16  # eV s
C = 2.99792458e17       # nm / s
BILAYERS = [(r"Graphene-BC$_3$", "E_Graphene-BC3", "HSE_K17", BLUE),
            ("Graphene-Borophene", "F_Graphene-Borophene", "HSE_K65_Accurate", GREEN),
            (r"Graphene-B$_4$C$_3$", "G_Graphene-B4C3", "HSE_K17", VIOLET)]
MONOLAYERS = [(r"BC$_3$", "A_BC3_PBE_K65", BLUE),
              ("Borophene", "B_Borophene_PBE_K65", GREEN),
              (r"B$_4$C$_3$", "C_B4C3_PBE_K65", VIOLET),
              ("Graphene", "D_Graphene_PBE_K129", "#AAAFBE")]
HSE = [(label, prefix + "_" + suffix, color) for label, prefix, suffix, color in BILAYERS]
COMPONENTS = [(0,0),(1,1),(2,2),(0,1),(1,2),(2,0),(1,0),(2,1),(0,2)]
NAMES = [r"$xx$ (in-plane)", r"$yy$ (in-plane)", r"$zz$ (out-of-plane)",
         r"$xy$", r"$yz$", r"$zx$", r"$yx$", r"$zy$", r"$xz$"]


@lru_cache(None)
def dielectric(source):
    path = ROOT / "5_Dielectric_function" / source
    if not (path / "vaspout.h5").exists():
        root = ET.parse(path / "vasprun.xml").getroot()
        node = root.find(".//dielectricfunction")
        real = np.array([np.fromstring(r.text, sep=" ") for r in node.findall("real/array/set/r")])
        imag = np.array([np.fromstring(r.text, sep=" ") for r in node.findall("imag/array/set/r")])
        epsilon = np.full((3,3,len(real),2), np.nan)
        for col,(i,j) in enumerate([(0,0),(1,1),(2,2),(0,1),(1,2),(2,0)],1):
            epsilon[i,j,:,0],epsilon[i,j,:,1]=real[:,col],imag[:,col]
        SOURCES[source] = {"file": str((path / "vasprun.xml").relative_to(ROOT)), "group": "first density-density dielectricfunction", "energy_points": len(real)}
        return real[:,0],epsilon
    with h5py.File(path / "vaspout.h5") as f:
        group = "results/linear_response"
        if (path / "KPOINTS_OPT").exists() and group + "_kpoints_opt" in f:
            group += "_kpoints_opt"
        energy = f[group + "/energies_dielectric_function"][:]
        epsilon = f[group + "/density_density_dielectric_function"][:]
    SOURCES[source] = {"file": str((path / "vaspout.h5").relative_to(ROOT)), "group": group, "energy_points": len(energy), "tensor_shape": list(epsilon.shape)}
    return energy, epsilon


def values(source, direction, quantity):
    energy, tensor = dielectric(source)
    real, imag = tensor[direction, direction].T
    magnitude = np.hypot(real, imag)
    n = np.sqrt((magnitude + real) / 2)
    k = np.sqrt((magnitude - real) / 2)
    result = {"alpha": 2*energy*k/(HBAR*C), "n": n, "k": k,
              "R": ((n-1)**2+k*k)/((n+1)**2+k*k),
              "loss": imag/(real*real+imag*imag)}[quantity]
    keep = energy <= 24
    return energy[keep], result[keep]


def spectrum(ax, systems, direction, quantity):
    for label, source, color in systems:
        x, y = values(source, direction, quantity)
        ax.plot(x, y, color=color, label=label)
    ax.set(xlim=(0,25), xticks=[0,5,10,15,20,25], ylim=(0,None))


def optical(name, quantity, ylabel, systems, directions, material):
    if len(directions) == 3:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
    for col, direction in enumerate(directions):
        ax = axes.flat[col]
        spectrum(ax, systems, direction, quantity)
        ax.set_title(["in-plane", "yy-component", "out-of-plane"][direction])
        ax.set_xlabel("Photon energy (eV)")
        if col % 2 == 0:
            ax.set_ylabel(ylabel)
        if len(directions) == 2:
            ax.legend(loc="best")
    if len(directions) == 3:
        axes[1,1].axis("off")
        handles, labels = axes[0,0].get_legend_handles_labels()
        axes[1,1].legend(handles, labels, loc="center")
    fig.suptitle({"alpha":"Absorption coefficient", "loss":"Energy-loss spectrum", "n":"Refractive index",
                 "R":"Reflectivity", "k":"Extinction coefficient"}[quantity] + " for " + material, fontsize=20)
    fig.tight_layout()
    save(fig, name)


if __name__ == "__main__":
    # %% Main-text bilayer optical properties
    for name, quantity, ylabel in [
        ("proj1.12a.pdf", "alpha", r"Absorption coefficient (nm$^{-1}$)"),
        ("proj1.12b.pdf", "loss", "Energy-loss spectrum"),
        ("proj1.13a_cor.pdf", "R", "Reflectivity"),
        ("proj1.13b.pdf", "n", "Refractive index"),
        ("proj1.14_cor.pdf", "k", "Extinction coefficient"),
    ]:
        optical(name, quantity, ylabel, HSE, [0,2], "bilayers by HSE06")

    # %% Monolayer optical properties: three data panels plus legend
    for name, quantity, ylabel in [
        ("S1.14a.pdf", "alpha", r"Absorption coefficient (nm$^{-1}$)"),
        ("S1.14b.pdf", "loss", "Energy-loss spectrum"),
        ("S1.14c.pdf", "n", "Refractive index"),
        ("S1.14d_correct.pdf", "R", "Reflectivity"),
        ("S1.14e_correct.pdf", "k", "Extinction coefficient"),
    ]:
        optical(name, quantity, ylabel, MONOLAYERS, [0,1,2], "monolayers by PBE")

    # %% Bilayer optical properties: three data panels plus legend
    for name, quantity, ylabel in [
        ("S1.20.pdf", "alpha", r"Absorption coefficient (nm$^{-1}$)"),
        ("S1.21.pdf", "loss", "Energy-loss spectrum"),
        ("S1.22.pdf", "n", "Refractive index"),
        ("S1.23_correct.pdf", "R", "Reflectivity"),
        ("S1.24_correct.pdf", "k", "Extinction coefficient"),
    ]:
        optical(name, quantity, ylabel, HSE, [0,1,2], "bilayers by HSE06")

    # %% Monolayer dielectric functions
    for letter, (label, source, color) in zip("abcd", MONOLAYERS):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        energy, epsilon = dielectric(source)
        keep = energy <= 24
        for ax, direction in zip(axes, [0,2]):
            ax.plot(energy[keep], epsilon[direction,direction,keep,0], color=color, label="Real part")
            ax.plot(energy[keep], epsilon[direction,direction,keep,1], color=color, ls="--", label="Imaginary part")
            ax.axhline(0, color=GREY, ls="--", zorder=0)
            ax.set(xlim=(0,25), xticks=[0,5,10,15,20,25], xlabel="Photon energy (eV)")
            ax.set_title("in-plane" if direction == 0 else "out-of-plane")
            ax.legend(loc="best")
        axes[0].set_ylabel("Dielectric function")
        fig.suptitle("Dielectric function for " + label, fontsize=20)
        fig.tight_layout()
        save(fig, "S1.13" + letter + ".pdf")

    # %% Main-text dielectric functions: original real/imaginary rows
    for index, name in [(1,"proj1.9.pdf"), (0,"proj1.10_diff.pdf"), (2,"proj1.11_diff.pdf")]:
        label, prefix, hse, color = BILAYERS[index]
        components = [(0,0),(2,2)] + ([(0,1)] if index != 1 else [])
        fig, axes = plt.subplots(2, len(components), figsize=(8*len(components), 12))
        for row in range(2):
            for col, (i,j) in enumerate(components):
                ax = axes[row,col]
                upper = (2 if index == 0 else 16) if i != j else 24
                for suffix, shade, functional in [("PBE_K65_Normal",color,"PBE"), (hse,ORANGE,"HSE06")]:
                    energy, epsilon = dielectric(prefix + "_" + suffix)
                    keep = energy <= upper
                    ax.plot(energy[keep],epsilon[i,j,keep,row],color=shade,label=functional)
                ax.axhline(0, color=GREY, ls="--", zorder=0)
                ax.set_xlim(0, upper if i != j else 25)
                component = ["in-plane", "out-of-plane", "xy-component"][col]
                ax.set_title(("Real part for " if row == 0 else "Imaginary part for ") + component)
                ax.legend(loc="best")
                if row == 1:
                    ax.set_xlabel("Photon energy (eV)")
            axes[row,0].set_ylabel("Dielectric function")
        fig.suptitle("Dielectric function for " + label, fontsize=20)
        fig.tight_layout()
        save(fig,name)

    # %% Complete dielectric tensors, without symmetrization
    for index, name in enumerate(["S1.18.pdf", "S1.17_alt.pdf", "S1.19.pdf"]):
        label, prefix, hse, color = BILAYERS[index]
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        for p, (ax, (i,j), heading) in enumerate(zip(axes.flat, COMPONENTS, NAMES)):
            for suffix, shade, functional in [("PBE_K65_Normal",color,"PBE"), (hse,ORANGE,"HSE06")]:
                energy, epsilon = dielectric(prefix + "_" + suffix)
                keep = energy <= 24
                for part, style in [(0,"-"),(1,"--")]:
                    ax.plot(energy[keep], epsilon[i,j,keep,part], color=shade, ls=style,
                            label=("Real part " if part == 0 else "Imaginary part ") + functional)
            ax.axhline(0, color=GREY, ls="--", zorder=0)
            ax.set(xlim=(0,25), xticks=[0,5,10,15,20,25])
            if index == 1 and i != j:
                ax.set_ylim(-.1,.4)
            ax.set_title(heading)
            ax.legend(loc="best")
            if p % 3 == 0:
                ax.set_ylabel("Dielectric function")
            if p >= 6:
                ax.set_xlabel("Photon energy (eV)")
        fig.suptitle("Dielectric function for " + label, fontsize=20)
        fig.tight_layout()
        save(fig,name)

    # %% Optical convergence
    for name, sources, labels, colors, windows in [
        ("S1.12_alt.pdf", ["D_Graphene_PBE_K33","D_Graphene_PBE_K65","D_Graphene_PBE_K129"],
         [r"$33\times33\times1$",r"$65\times65\times1$",r"$129\times129\times1$"],
         [BLUE,VIOLET,"#C82364"], [(0,5),(10,15)]),
        ("S1.15.pdf", [f"G_Graphene-B4C3_PBE_K17_N{n}" for n in [32,64,128,256,512]],
         [f"{n} bands" for n in [32,64,128,256,512]],
         ["#F03C64",ORANGE,GREEN,BLUE,"#643CC3"], [(0,24),(0,24)]),
        ("S1.16_alt.pdf", ["F_Graphene-Borophene_"+s for s in ["PBE_K65_Normal","HSE_K17","HSE_K65_Normal_EDIFF-4","HSE_K65_Normal_EDIFF-5","HSE_K65_Accurate"]],
         ["PBE",r"HSE06, $17\times17\times1$",r"HSE06, EDIFF=$10^{-4}$",r"HSE06, EDIFF=$10^{-5}$",r"HSE06, EDIFF=$10^{-6}$ (Accurate)"],
         ["#AAAFBE",GREEN,BLUE,"#643CC3",ORANGE], [(0,4),(12,15)]),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for row in range(2):
            for col, direction in enumerate([0,2]):
                ax = axes[row,col]
                low, high = windows[col]
                for source, label, color in zip(sources,labels,colors):
                    energy, epsilon = dielectric(source)
                    keep = (energy >= low) & (energy <= high)
                    ax.plot(energy[keep], epsilon[direction,direction,keep,row], color=color, label=label)
                ax.axhline(0, color=GREY, ls="--", zorder=0)
                ax.set_xlim(low,high)
                ax.set_title(("Real part for " if row == 0 else "Imaginary part for ") + ("in-plane" if col == 0 else "out-of-plane"))
                ax.legend(loc="best")
                if row == 1:
                    ax.set_xlabel("Photon energy (eV)")
            axes[row,0].set_ylabel("Dielectric function")
        fig.suptitle("Dielectric function convergence", fontsize=20)
        fig.tight_layout()
        save(fig,name)

    (HERE / "optical_sources.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
