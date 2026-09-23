"""Combine the original Schottky sketch with the raw HSE06 band structure."""
import h5py
import numpy as np
from matplotlib.lines import Line2D
from style import *

fig,axes=grid(1,2,3.6,width=8.5)
fig.subplots_adjust(left=.06,right=.985,bottom=.16,top=.96,wspace=.23)
ax=axes[0,0]
ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
title(ax,"(a) Schottky barriers")
for y,label,color in [(.76,r"$E_C$",BLUE),(.60,r"$E_F$",FERMI),(.29,r"$E_V$","#EB731E")]:
    ax.plot([.12,.92],[y,y],color=color)
    ax.text(.07,y,label,ha="right",va="center",fontsize=13)
for x,lo,hi,label,color in [(.62,.60,.76,r"$\Phi_n$",BLUE),(.35,.29,.60,r"$\Phi_p$","#EB731E")]:
    ax.annotate("",(x,hi),(x,lo),arrowprops={"arrowstyle":"<->","color":color,"lw":1.5})
    ax.text(x+.04,(lo+hi)/2,label,va="center",fontsize=13)
for y,label in [(.18,r"$\Phi_n<\Phi_p$: n-type SB"),(.09,r"$\Phi_n>\Phi_p$: p-type SB"),
                (0,r"$\Phi_n$ or $\Phi_p\approx0$: Ohmic contact")]:
    ax.text(.04,y,label,fontsize=11,va="center")

source=ROOT/"3_Bandstructure/G_Graphene-B4C3_top_HSE/vaspout.h5"
with h5py.File(source) as f:
    g=f["results/electron_eigenvalues_kpoints_opt"]
    e=g["eigenvalues"][0]-f["results/electron_dos_kpoints_opt/efermi"][()]
    k=g["kpoint_coords"][:]
x=np.r_[0,np.cumsum(np.linalg.norm(np.diff(k,axis=0),axis=1))];length=x[-1];x/=length
ticks=x[[0,len(k)//3-1,2*len(k)//3-1,len(k)-1]]
ax=axes[0,1];ax.plot(x,e,color=VIOLET)
ax.set(xlim=(0,1),ylim=(-3,3),yticks=np.arange(-3,4),ylabel="Energy (eV)",
       xlabel=r"Wave vector ($k$)",xticks=ticks,xticklabels=[r"$\Gamma$","K","M",r"$\Gamma$"])
title(ax,r"(b) Graphene-B$_4$C$_3$ (HSE06)")
for tick in ticks[1:-1]:ax.axvline(tick,color=GREY,ls="--",zorder=0)
ax.axhline(0,color=FERMI,ls="--",zorder=0)
for level,color,start,label in [(1.42,BLUE,ticks[-2],r"$\Phi_n$"),(-.87,"#EB731E",1-.08/length,r"$\Phi_p$")]:
    ax.hlines(level,start,1,colors=color,linestyles="--")
    ax.annotate("",(1-.05/length,level),(1-.05/length,0),
                arrowprops={"arrowstyle":"<->","color":color,"lw":1.5})
    ax.text(1-.2/length,level/2,label,color=color,fontsize=13,va="center")
ax.text(.035,.51,r"$E_\mathrm{F}=0$",transform=ax.transAxes,color=FERMI,fontsize=11,va="bottom")
save(fig,"proj1.8_schottky.pdf")
