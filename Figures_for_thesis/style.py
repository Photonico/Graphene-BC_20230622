"""Shared settings, matching the thesis Chapter 4 figures."""
from pathlib import Path
import shutil
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
THESIS = ROOT.parent / "PhD_thesis_20251216" / "figures_proj1"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 14,
    "axes.labelsize": 16, "axes.titlesize": 20, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "legend.fontsize": 14, "figure.dpi": 196,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round", "lines.solid_joinstyle": "round",
    "lines.dash_joinstyle": "round", "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "pdf.fonttype": 42, "path.simplify": False,
})
BLUE, GREEN, VIOLET, GREY, ORANGE = "#1478E1", "#28AF3C", "#8C64E1", "#787878", "#FA8C00"
FERMI = "#643CC3"
TAB = {"boxstyle": "round", "facecolor": "white",
       "edgecolor": plt.rcParams["legend.edgecolor"],
       "alpha": plt.rcParams["legend.framealpha"]}


def title(ax, text):
    ax.set_title(text, loc="left", x=0.045, y=0.955, pad=0, va="top",
                 fontsize=12, bbox=TAB, zorder=10)


def grid(rows, cols, height, right_legend=False):
    fig, axes = plt.subplots(rows, cols, figsize=(10, height), squeeze=False)
    fig.subplots_adjust(left=0.10, right=0.75 if right_legend else 0.98,
                        bottom=0.12 if right_legend else 0.21, top=0.97,
                        wspace=0.27, hspace=0.25)
    return fig, axes


def legend(fig, handles, labels=None, right=False, columns=2):
    if right:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.77, 0.59),
                   frameon=True, fancybox=True)
    else:
        fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.09, 0.005),
                   ncol=columns, frameon=True, fancybox=True)


def save(fig, name):
    fig.savefig(HERE / name, metadata={"CreationDate": None})
    plt.close(fig)
    shutil.copyfile(HERE / name, THESIS / name)
