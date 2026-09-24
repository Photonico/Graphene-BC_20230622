"""Original author figure settings, written explicitly for independent editing."""
from pathlib import Path
import shutil
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
THESIS = ROOT.parent / "PhD_thesis_20251216" / "figures_proj1"
plt.rcdefaults()
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "axes.titlesize": 20,
    "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 12, "figure.dpi": 196, "figure.facecolor": "white",
    "lines.linewidth": 1.5, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42,
    "path.simplify": False,
})
BLUE, GREEN, VIOLET, GREY, ORANGE = "#1478E1", "#28AF3C", "#8C64E1", "#787878", "#FA8C00"
FERMI = "#643CC3"
TAB = {"boxstyle": "round,pad=0.2", "facecolor": "white",
       "edgecolor": "#B4B4B4", "alpha": .75, "linewidth": 1.5}


def title(ax, text):
    ax.text(.04, .95, text, transform=ax.transAxes, va="top",
            fontsize=16, bbox=TAB, zorder=10)


def grid(rows, cols, height, width=10):
    return plt.subplots(rows, cols, figsize=(width, height), squeeze=False)


def save(fig, name):
    fig.savefig(HERE / name, metadata={"CreationDate": None})
    plt.close(fig)
    shutil.copyfile(HERE / name, THESIS / name)
