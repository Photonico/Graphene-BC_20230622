"""Lattice and convergence figures from the archived numerical tables."""
import re
import json
import sys
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scipy.interpolate import griddata
from style import *
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT))
from vmatplot.algorithms import fit_eos


SOURCES = {}

def table(path):
    data = np.loadtxt(ROOT / path, skiprows=1)
    SOURCES[path] = {"columns": (ROOT / path).read_text().splitlines()[0], "shape": list(data.shape), "energy_column": -1}
    return data


fig,axes=grid(1,1,4.4)
fig.subplots_adjust(left=.18,bottom=.17,top=.96)
ax=axes[0,0]
lines=(ROOT / "1_Kpoints/Graphene_BC3_Hollow/energy_kpoint.dat").read_text().splitlines()[1:]
xy=np.array([(int(re.search(r"\((\d+),",s)[1]),float(s.split()[-1])) for s in lines])
xy=xy[(xy[:,0]>=9)&(xy[:,0]<=41)]
ax.plot(xy[:,0],xy[:,-1],"o-",color=BLUE,mfc="white")
ax.set(xlabel=r"$k$-point mesh ($X\times X\times1$)",ylabel="Energy (eV)",xticks=xy[::2,0])
ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
title(ax,r"Graphene-BC$_3$ (Hollow)")
save(fig,"S1.1.pdf")


fig,axes=grid(2,2,6.4,right_legend=True)
fig.subplots_adjust(left=.14,wspace=.44)
for i,(ax,folder,label,color) in enumerate(zip(axes.flat,
    ["A_BC3","B_Borophene","C_B4C3","D_Graphene"],
    [r"BC$_3$","Borophene",r"B$_4$C$_3$","Graphene"],[BLUE,GREEN,VIOLET,GREY])):
    xy=table("0_Lattice/"+folder+"/free_energy_lattice.dat")
    x,y=fit_eos(xy[:,0],xy[:,-1])
    ax.plot(x,y,color=color);ax.plot(xy[:,0],xy[:,-1],"o",color=color,mfc="white",ms=5)
    j=np.argmin(y);ax.plot(x[j],y[j],"o",color=color,ms=6)
    title(ax,f"({chr(97+i)}) "+label)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    if i%2==0: ax.set_ylabel("Energy (eV)")
    if i>=2: ax.set_xlabel(r"Lattice constant ($\mathrm{\AA}$)")
legend(fig,[Line2D([],[],color=GREY),Line2D([],[],color=GREY,marker="o",mfc="white",ls=""),
            Line2D([],[],color=GREY,marker="o",ls="")],
       ["Fitted curve","Source data","Fitted\nminimum"],right=True)
save(fig,"S1.2.pdf")

fig,axes=grid(2,2,6.5)
fig.subplots_adjust(left=.14,bottom=.13,wspace=.44,hspace=.43)
for i,(ax,prefix,label) in enumerate(zip(axes.flat,
    ["E_Graphene-BC3_","F_Graphene-Borophene_","G_Graphene-B4C3_"],
    [r"Graphene-BC$_3$","Graphene-Borophene",r"Graphene-B$_4$C$_3$"])):
    sites=["Top","Bridge","Hollow"] if i==0 else ["Top","Bridge","Hollow1","Hollow2"]
    for site,color in zip(sites,[BLUE,GREEN,VIOLET,"#C82364"]):
        xy=table("0_Lattice/"+prefix+site+"/free_energy_lattice.dat")
        x,y=fit_eos(xy[:,0],xy[:,-1]);ax.plot(x,y,color=color)
        ax.plot(xy[:,0],xy[:,-1],"o",color=color,mfc="white",ms=5)
        j=np.argmin(y);ax.plot(x[j],y[j],"o",color=color,ms=6)
    title(ax,f"({chr(97+i)}) "+label)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel(r"Lattice constant ($\mathrm{\AA}$)")
    if i%2==0: ax.set_ylabel("Energy (eV)")
axes[1,1].axis("off")
axes[1,1].legend([Line2D([],[],color=c,marker="o",mfc="white") for c in [BLUE,GREEN,VIOLET,"#C82364"]]
    +[Line2D([],[],color=GREY,marker="o",ls="")],
    ["Top","Bridge",r"Hollow (BC$_3$)"+"\nHollow 1 (others)","Hollow 2","Fitted minimum"],loc="center",frameon=True,fancybox=True)
save(fig,"S1.3.pdf")

# The colour field uses the original linear interpolation. Only the actual
# sampled minimum is marked; the old diagonal scan was not a 2D minimization.
for name,prefix,sites,cmap in [
    ("S1.4.pdf","E_Graphene-BC3_",["Bridge","Hollow","Top"],"Blues_r"),
    ("S1.5.pdf","F_Graphene-Borophene_",["Bridge","Hollow1","Hollow2","Top"],"Greens_r"),
    ("S1.6.pdf","G_Graphene-B4C3_",["Bridge","Hollow1","Hollow2","Top"],"Purples_r"),
]:
    fig,axes=grid(2,2,7.1)
    fig.subplots_adjust(left=.10,right=.86,bottom=.18,wspace=.63,hspace=.35)
    for i,(ax,site) in enumerate(zip(axes.flat,sites)):
        xyz=table("0_Lattice_Distance/"+prefix+site+"/lattice_distance.dat")
        a,d,e=xyz.T
        aa,dd=np.meshgrid(np.linspace(a.min(),a.max(),400),np.linspace(d.min(),d.max(),400))
        ee=griddata((a,d),e,(aa,dd),method="linear")
        cp=ax.pcolormesh(aa,dd,ee,shading="auto",cmap=cmap,alpha=.75,
                         vmax=e.min()+np.ptp(e)*.125,rasterized=True)
        cbar=fig.colorbar(cp,ax=ax,pad=.025,fraction=.045)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(4))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        idx=np.argmin(e)
        ax.plot(a[idx],d[idx],"o",mfc="white",mec="black",ms=6,zorder=5)
        title(ax,f"({chr(97+i)}) "+site.replace("Hollow1","Hollow 1").replace("Hollow2","Hollow 2"))
        ax.set_xticks(np.round(a.min() + np.ptp(a) * np.array([.2, .8]), 2))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if i%2==0: ax.set_ylabel(r"Interlayer spacing ($\mathrm{\AA}$)")
        if i>=2: ax.set_xlabel(r"Lattice constant ($\mathrm{\AA}$)")
    if len(sites)==3: axes[1,1].axis("off")
    fig.text(.98,.58,"Energy (eV)",ha="right",va="center",rotation=90,fontsize=16)
    legend(fig,[Line2D([],[],marker="o",color="black",mfc="white",ls="")],
           ["Minimum of sampled data"],columns=1)
    save(fig,name)

(HERE / "energy_sources.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
