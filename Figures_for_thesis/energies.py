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


fig,axes=grid(1,1,6,width=10)
fig.subplots_adjust(left=.13,right=.97,bottom=.13,top=.90)
ax=axes[0,0]
lines=(ROOT / "1_Kpoints/Graphene_BC3_Hollow/energy_kpoint.dat").read_text().splitlines()[1:]
xy=np.array([(int(re.search(r"\((\d+),",s)[1]),float(s.split()[-1])) for s in lines])
xy=xy[(xy[:,0]>=9)&(xy[:,0]<=41)]
ax.plot(xy[:,0],xy[:,-1],"o-",color=BLUE,mfc="white")
ax.set(xlabel=r"$k$-point mesh ($X\times X\times1$)",ylabel="Energy (eV)",xticks=xy[::2,0])
ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
ax.set_title(r"Total energy versus k-points for Graphene-BC$_3$ (Hollow)")
save(fig,"S1.1.pdf")


fig,axes=grid(2,2,12,width=16)
fig.subplots_adjust(left=.115,right=.985,bottom=.12,wspace=.28,hspace=.20)
for i,(ax,folder,label,color) in enumerate(zip(axes.flat,
    ["A_BC3","B_Borophene","C_B4C3","D_Graphene"],
    [r"BC$_3$","Borophene",r"B$_4$C$_3$","Graphene"],[BLUE,GREEN,VIOLET,GREY])):
    xy=table("0_Lattice/"+folder+"/free_energy_lattice.dat")
    x,y=fit_eos(xy[:,0],xy[:,-1])
    ax.plot(x,y,color=color);ax.plot(xy[:,0],xy[:,-1],"o",color=color,mfc="white",ms=7)
    j=np.argmin(y);ax.plot(x[j],y[j],"o",color=color,ms=7)
    low,high=ax.get_ylim();ax.set_ylim(low,high+.15*(high-low))
    ax.set_title(label, fontsize=18)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_ylabel("Energy (eV)")
    ax.set_xlabel(r"Lattice constant ($\mathrm{\AA}$)")
axes[0,0].legend([Line2D([],[],color=GREY),Line2D([],[],color=GREY,marker="o",mfc="white",ls=""),
                   Line2D([],[],color=GREY,marker="o",ls="")],
                  ["Fitted curve","Source data","Fitted minimum"],
                  loc="upper right",frameon=True,fancybox=True)
fig.suptitle("Total energy versus lattice", fontsize=20)
fig.tight_layout()
save(fig,"S1.2.pdf")

fig,axes=grid(2,2,12,width=16)
fig.subplots_adjust(left=.11,right=.985,bottom=.12,wspace=.28,hspace=.30)
for i,(ax,prefix,label) in enumerate(zip(axes.flat,
    ["E_Graphene-BC3_","F_Graphene-Borophene_","G_Graphene-B4C3_"],
    [r"Graphene-BC$_3$","Graphene-Borophene",r"Graphene-B$_4$C$_3$"])):
    sites=["Top","Bridge","Hollow"] if i==0 else ["Top","Bridge","Hollow1","Hollow2"]
    for site,color in zip(sites,[BLUE,GREEN,VIOLET,"#C82364"]):
        xy=table("0_Lattice/"+prefix+site+"/free_energy_lattice.dat")
        x,y=fit_eos(xy[:,0],xy[:,-1]);ax.plot(x,y,color=color)
        ax.plot(xy[:,0],xy[:,-1],"o",color=color,mfc="white",ms=7)
        j=np.argmin(y);ax.plot(x[j],y[j],"o",color=color,ms=7)
    low,high=ax.get_ylim();ax.set_ylim(low,high+.18*(high-low))
    ax.set_title(label, fontsize=18)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel(r"Lattice constant ($\mathrm{\AA}$)")
    ax.set_ylabel("Energy (eV)")
axes[1,1].axis("off")
axes[1,1].legend([Line2D([],[],color=c,marker="o",mfc="white") for c in [BLUE,GREEN,VIOLET,"#C82364"]]
    +[Line2D([],[],color=GREY,marker="o",ls="")],
    ["Top","Bridge",r"Hollow (BC$_3$) / Hollow 1 (others)","Hollow 2","Fitted minimum"],
    loc="center",ncol=1,frameon=True,fancybox=True)
fig.suptitle("Total energy versus lattice", fontsize=20)
fig.tight_layout()
save(fig,"S1.3.pdf")

# The colour field uses the original linear interpolation. Only the actual
# sampled minimum is marked; each figure uses one common energy colour scale.
surface_scales = {}
for name,prefix,sites,cmap in [
    ("S1.4.pdf","E_Graphene-BC3_",["Bridge","Hollow","Top"],"Blues_r"),
    ("S1.5.pdf","F_Graphene-Borophene_",["Bridge","Hollow1","Hollow2","Top"],"Greens_r"),
    ("S1.6.pdf","G_Graphene-B4C3_",["Bridge","Hollow1","Hollow2","Top"],"Purples_r"),
]:
    datasets = [table("0_Lattice_Distance/"+prefix+site+"/lattice_distance.dat") for site in sites]
    vmin = min(data[:,2].min() for data in datasets)
    vmax = max(data[:,2].min()+np.ptp(data[:,2])*.125 for data in datasets)
    surface_scales[name] = {"minimum_eV":float(vmin),"maximum_eV":float(vmax),
                            "rule":"Shared range spanning sampled minima and prior per-panel upper limits; identical composition within each figure."}
    fig,axes=grid(2,2,10,width=13)
    fig.subplots_adjust(left=.095,right=.985 if len(sites)==3 else .87,
                        bottom=.12 if len(sites)==3 else .11,
                        top=.96,wspace=.12,hspace=.18 if len(sites)==3 else .16)
    for i,(ax,site,xyz) in enumerate(zip(axes.flat,sites,datasets)):
        a,d,e=xyz.T
        aa,dd=np.meshgrid(np.linspace(a.min(),a.max(),400),np.linspace(d.min(),d.max(),400))
        ee=griddata((a,d),e,(aa,dd),method="linear")
        cp=ax.pcolormesh(aa,dd,ee,shading="auto",cmap=cmap,alpha=.75,
                         vmin=vmin,vmax=vmax,rasterized=True)
        idx=np.argmin(e)
        ax.plot(a[idx],d[idx],"o",mfc="white",mec="black",ms=6,zorder=5)
        title(ax,f"({chr(97+i)}) "+site.replace("Hollow1","Hollow 1").replace("Hollow2","Hollow 2"))
        ax.set_xticks(np.round(a.min() + np.ptp(a) * np.array([.2, .8]), 2))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if i%2==0: ax.set_ylabel(r"Interlayer spacing ($\mathrm{\AA}$)")
        if len(sites)==3 or i>=2: ax.set_xlabel(r"Lattice constant ($\mathrm{\AA}$)")
    if len(sites)==3:
        legend_ax=axes[1,1]
        legend_ax.axis("off")
        cbar=fig.colorbar(cp,cax=legend_ax.inset_axes([.20,.19,.055,.63]))
        cbar.ax.yaxis.set_major_locator(MaxNLocator(3))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cbar.ax.set_title("Energy\n(eV)",fontsize=16,pad=10)
        legend_ax.legend([Line2D([],[],marker="o",color="black",mfc="white",ls="")],
                         ["Sampled minimum"],loc="center left",bbox_to_anchor=(.55,.5),
                         frameon=True,fancybox=True)
    else:
        cbar=fig.colorbar(cp,cax=fig.add_axes([.90,.16,.018,.67]))
        cbar.ax.yaxis.set_major_locator(MaxNLocator(4))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cbar.ax.set_title("Energy\n(eV)",x=1.1,fontsize=16,pad=10)
        fig.legend([Line2D([],[],marker="o",color="black",mfc="white",ls="")],
                   ["Sampled\nminimum"],loc="lower left",bbox_to_anchor=(.865,.01),
                   borderaxespad=0,handlelength=.8,handletextpad=.35,borderpad=.25,
                   frameon=True,fancybox=True)
    save(fig,name)
(HERE / "energy_colour_scales.json").write_text(json.dumps(surface_scales,indent=2)+"\n")

(HERE / "energy_sources.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
