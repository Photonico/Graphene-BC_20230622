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
    low,high=ax.get_ylim();ax.set_ylim(low,high+.18*(high-low))


def optical(name, quantities, systems, directions, height):
    three_panels=len(quantities)==1 and len(directions)==3
    if three_panels:
        fig,axes=grid(2,2,4.9,width=8.5)
    else:
        fig,axes=grid(len(quantities),len(directions),height,width=8 if len(directions)==2 else 10)
    right=.785 if len(directions)==2 else .985
    bottom=.12 if three_panels else (.065 if len(quantities)>2 else (.12 if len(quantities)==2 else .19))
    fig.subplots_adjust(left=.09 if len(directions)==2 else (.085 if three_panels else .075),right=right,bottom=bottom,top=.97,
                        hspace=.30 if three_panels else .14,wspace=.20)
    for row,(quantity,ylabel) in enumerate(quantities):
        for col,direction in enumerate(directions):
            ax=axes.flat[col] if three_panels else axes[row,col]
            spectrum(ax,systems,direction,quantity)
            if name=="S1.14_optics.pdf" and row==1 and col==2:
                low,high=ax.get_ylim();ax.set_ylim(low,high+.03*(high-low))
            title(ax,f"({chr(97+row*len(directions)+col)}) "+NAMES[direction])
            if (three_panels and col%2==0) or (not three_panels and col==0): ax.set_ylabel(ylabel)
            if row==len(quantities)-1: ax.set_xlabel("Photon energy (eV)")
            else: ax.tick_params(labelbottom=False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    if three_panels:
        axes[1,1].axis("off")
        axes[1,1].legend(handles,labels,loc="center",frameon=True,fancybox=True)
    elif len(directions)==2:
        labels=[label.replace("Graphene-","Graphene-\n") for label in labels]
        fig.legend(handles,labels,loc="upper left",bbox_to_anchor=(.80,.95),
                   handlelength=1.2,labelspacing=.5,borderaxespad=0,frameon=True,fancybox=True)
    else:
        axes[2,0].legend(handles,labels,loc="upper right",handlelength=.9,handletextpad=.4,
                         labelspacing=.25,borderpad=.3,frameon=True,fancybox=True)
    save(fig,name)


if __name__ == "__main__":
    optical("proj1.12_optics.pdf", [("alpha",r"$\alpha$ (nm$^{-1}$)"),("loss","Energy-loss spectrum")], HSE,[0,2],4.8)
    optical("proj1.13_optics.pdf", [("R","Reflectivity"),("n","Refractive index")], HSE,[0,2],4.8)
    optical("proj1.14_cor.pdf", [("k","Extinction coefficient")], HSE,[0,2],3.0)
    optical("S1.14_optics.pdf", [("alpha",r"$\alpha$ (nm$^{-1}$)"),("loss",r"$L$"),
            ("n",r"$n$"),("R",r"$R$"),("k",r"$\kappa$")],MONOLAYERS,[0,1,2],8.6)
    for name, quantity, ylabel in [("S1.20.pdf","alpha",r"$\alpha$ (nm$^{-1}$)"),
        ("S1.21.pdf","loss","Energy-loss spectrum"),("S1.22.pdf","n","Refractive index"),
        ("S1.23_correct.pdf","R","Reflectivity"),("S1.24_correct.pdf","k","Extinction coefficient")]:
        optical(name,[(quantity,ylabel)],HSE,[0,1,2],3.6)

    # Four monolayers, two tensor directions, one legend for real and imaginary parts.
    fig, axes = grid(4,2,7.5,width=8)
    fig.subplots_adjust(left=.09,right=.985,bottom=.075,top=.975,wspace=.19,hspace=.12)
    for row,(label,source,color) in enumerate(MONOLAYERS):
        energy, epsilon = dielectric(source)
        keep = energy <= 24
        for col,direction in enumerate([0,2]):
            ax=axes[row,col]
            ax.plot(energy[keep],epsilon[direction,direction,keep,0],color=color)
            ax.plot(energy[keep],epsilon[direction,direction,keep,1],color=color,ls="--")
            ax.set(xlim=(0,25),xticks=[0,5,10,15,20,25])
            low,high=ax.get_ylim();ax.set_ylim(low,high+.24*(high-low))
            title(ax,f"({chr(97+row*2+col)}) "+label+": "+NAMES[direction])
            if col == 0: ax.set_ylabel("Dielectric function")
            if row == 3: ax.set_xlabel("Photon energy (eV)")
            else: ax.tick_params(labelbottom=False)
    axes[0,0].legend([Line2D([],[],color=GREY),Line2D([],[],color=GREY,ls="--")],
                     ["Real part","Imaginary part"],loc="upper right",bbox_to_anchor=(1,.78),frameon=True,fancybox=True)
    save(fig,"S1.13_dielectric.pdf")

    # Main-text real/imaginary rows preserve the original off-diagonal zoom ranges.
    for index,name in [(1,"proj1.9.pdf"),(0,"proj1.10_diff.pdf"),(2,"proj1.11_diff.pdf")]:
        label,prefix,hse,color=BILAYERS[index]
        components=[(0,0),(2,2)]+([(0,1)] if index != 1 else [])
        fig,axes=grid(2,len(components),4.8,width=8 if len(components)==2 else 10)
        fig.subplots_adjust(left=.09 if len(components)==2 else .075,right=.985,bottom=.12,top=.97,wspace=.22,hspace=.15)
        for row in range(2):
            for col,(i,j) in enumerate(components):
                ax=axes[row,col]
                upper=(2 if index == 0 else 16) if i != j else 24
                for suffix,shade,functional in [("PBE_K65_Normal",color,"PBE"),(hse,ORANGE,"HSE06")]:
                    energy,epsilon=dielectric(prefix+"_"+suffix); keep=energy<=upper
                    ax.plot(energy[keep],epsilon[i,j,keep,row],color=shade,label=functional)
                ax.set_xlim(0,upper if i != j else 25)
                low,high=ax.get_ylim();ax.set_ylim(low,high+.15*(high-low))
                title(ax,f"({chr(97+row*len(components)+col)}) "+[NAMES[0],NAMES[2],NAMES[3]][col])
                if row == 1: ax.set_xlabel("Photon energy (eV)")
                else: ax.tick_params(labelbottom=False)
            axes[row,0].set_ylabel(r"Real part, $\varepsilon_1$" if row==0 else r"Imaginary part, $\varepsilon_2$")
        handles,labels=axes[0,0].get_legend_handles_labels()
        axes[0,0].legend(handles,labels,loc="upper right",frameon=True,fancybox=True)
        save(fig,name)

    # Retain all nine raw tensor components, including unequal transposed entries.
    for index,name in enumerate(["S1.18.pdf","S1.17_alt.pdf","S1.19.pdf"]):
        label,prefix,hse,color=BILAYERS[index]
        fig,axes=grid(3,3,6.6,width=9)
        fig.subplots_adjust(left=.08,right=.985,bottom=.09,top=.97,wspace=.24,hspace=.17)
        for p,(ax,(i,j),heading) in enumerate(zip(axes.flat,COMPONENTS,NAMES)):
            for suffix,shade,functional in [("PBE_K65_Normal",color,"PBE"),(hse,ORANGE,"HSE06")]:
                energy,epsilon=dielectric(prefix+"_"+suffix);keep=energy<=24
                for part,style in [(0,"-"),(1,"--")]:
                    ax.plot(energy[keep],epsilon[i,j,keep,part],color=shade,ls=style,
                            label=("Real" if part==0 else "Imaginary")+f" ({functional})")
            ax.set(xlim=(0,25),xticks=[0,10,20])
            if index==1 and i!=j: ax.set_ylim(-.1,.4)
            else:
                low,high=ax.get_ylim();ax.set_ylim(low,high+.15*(high-low))
            if index!=1 and p==2:
                low,high=ax.get_ylim();ax.set_ylim(low,high+.03*(high-low))
            title(ax,f"({chr(97+p)}) "+heading)
            if p%3==0: ax.set_ylabel("Dielectric function")
            if p>=6: ax.set_xlabel("Photon energy (eV)")
            else: ax.tick_params(labelbottom=False)
        handles,labels=axes[0,0].get_legend_handles_labels()
        legend_ax=axes[1,0] if index==1 else axes[0,0]
        legend_ax.legend(handles,labels,loc="upper right",bbox_to_anchor=(1,.84),handlelength=1.2,
                         labelspacing=.25,borderpad=.3,frameon=True,fancybox=True)
        save(fig,name)

    # Convergence plots: the original separate energy windows are retained.
    for name,sources,labels,colors,windows in [
        ("S1.12_alt.pdf",["D_Graphene_PBE_K33","D_Graphene_PBE_K65","D_Graphene_PBE_K129"],
         [r"$33\times33\times1$",r"$65\times65\times1$",r"$129\times129\times1$"],
         [BLUE,VIOLET,"#C82364"],[(0,5),(10,15)]),
        ("S1.15.pdf",[f"G_Graphene-B4C3_PBE_K17_N{n}" for n in [32,64,128,256,512]],
         [f"{n} bands" for n in [32,64,128,256,512]],
         ["#F03C64",ORANGE,GREEN,BLUE,"#643CC3"],[(0,24),(0,24)]),
        ("S1.16_alt.pdf",["F_Graphene-Borophene_"+s for s in ["PBE_K65_Normal","HSE_K17","HSE_K65_Normal_EDIFF-4","HSE_K65_Normal_EDIFF-5","HSE_K65_Accurate"]],
         ["PBE",r"HSE06, $17\times17\times1$",r"HSE06, EDIFF=$10^{-4}$",r"HSE06, EDIFF=$10^{-5}$",r"HSE06, EDIFF=$10^{-6}$ (Accurate)"],
         ["#AAAFBE",GREEN,BLUE,"#643CC3",ORANGE],[(0,4),(12,15)]),
    ]:
        fig,axes=grid(2,2,5.0,width=8)
        fig.subplots_adjust(left=.09,right=.78 if name=="S1.16_alt.pdf" else .975,
                            bottom=.12,top=.97,wspace=.22,hspace=.17)
        for row in range(2):
            for col,direction in enumerate([0,2]):
                ax=axes[row,col];low,high=windows[col]
                for source,label,color in zip(sources,labels,colors):
                    energy,epsilon=dielectric(source);keep=(energy>=low)&(energy<=high)
                    ax.plot(energy[keep],epsilon[direction,direction,keep,row],color=color,label=label)
                ax.set_xlim(low,high)
                ymin,ymax=ax.get_ylim();ax.set_ylim(ymin,ymax+.24*(ymax-ymin))
                title(ax,f"({chr(97+2*row+col)}) "+NAMES[direction])
                if row==1: ax.set_xlabel("Photon energy (eV)")
                else: ax.tick_params(labelbottom=False)
            axes[row,0].set_ylabel(r"Real part, $\varepsilon_1$" if row==0 else r"Imaginary part, $\varepsilon_2$")
        handles,labels=axes[0,0].get_legend_handles_labels()
        if name != "S1.16_alt.pdf":
            axes[0,0].legend(handles,labels,loc="upper right",frameon=True,fancybox=True)
        else:
            labels=[label.replace(", ","\n").replace(" (Accurate)","\n(Accurate)") for label in labels]
            fig.legend(handles,labels,loc="upper left",bbox_to_anchor=(.80,.95),
                       handlelength=1.1,labelspacing=.5,borderaxespad=0,frameon=True,fancybox=True)
        save(fig,name)

    (HERE / "optical_sources.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
