"""Recompose the original structure renders with the thesis figure typography."""
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from style import HERE, ROOT, TAB, save, title

SOURCE = ROOT / "2_Structure_and_CDD"
manifest = {}


def panel(ax, filename, heading):
    path = SOURCE / filename
    pixels = np.asarray(Image.open(path))
    ax.imshow(pixels, interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])
    title(ax, heading)
    manifest[filename] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": list(pixels.shape[:2][::-1]),
    }
    return pixels.shape[1], pixels.shape[0]


# The seven bond markers retain the coordinates and colours in the original
# 0.1_structure_figure.ipynb. Coordinates there refer to a 2 x 2 composite.
fig, axes = plt.subplots(2, 2, figsize=(8, 4.8))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.015, top=0.985,
                    wspace=0.025, hspace=0.035)
names = ["A_BC3.png", "B_Borophene.png", "C_B4C3.png", "D_Graphene.png"]
headings = [r"(a) BC$_3$", "(b) Borophene", r"(c) B$_4$C$_3$", "(d) Graphene"]
for ax, name, heading in zip(axes.flat, names, headings):
    width, height = panel(ax, name, heading)

shift_x, shift_y = 0.008 * 0.866, 0.008 * 0.5
markers = [
    (1, (0, 1), (3772, 654), (4052, 655), (0, -0.008), (0, -0.035), "#145AAA"),
    (2, (0, 1), (4654, 653), (4931, 653), (0, -0.008), (0, -0.035), "#238C4B"),
    (3, (0, 1), (4414, 1010), (4556, 763), (shift_x, shift_y), (0.035, 0.060), "#643CC3"),
    (4, (1, 0), (1647, 2403), (1785, 2163), (shift_x, shift_y), (0.035, 0.060), "#787D8C"),
    (5, (1, 0), (820, 2379), (968, 2161), (shift_x, shift_y), (0.035, 0.060), "#EB731E"),
    (6, (1, 0), (1396, 2730), (1518, 2518), (-shift_x, -shift_y), (-0.025, -0.040), "#AA1E64"),
    (7, (1, 0), (1241, 3104), (1364, 2892), (shift_x, shift_y), (0.035, 0.060), "#AA3CB9"),
]
for number, (row, col), start, end, offset, label_offset, colour in markers:
    scale = np.array([width, height])
    origin = np.array([col, row])
    start = np.array(start) / [2976, 1749] - origin
    end = np.array(end) / [2976, 1749] - origin
    ax = axes[row, col]
    ax.annotate("", xy=(start + offset) * scale, xytext=(end + offset) * scale,
                arrowprops=dict(arrowstyle="<->", color=colour, lw=1.5,
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    position = ((start + end) / 2 + label_offset) * scale
    ax.text(*position, rf"$l_{number}$", fontsize=13, ha="center", color=colour)
save(fig, "proj1.1.pdf")


fig, axes = plt.subplots(2, 2, figsize=(8, 4.8))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.015, top=0.985,
                    wspace=0.025, hspace=0.035)
materials = [r"Graphene–BC$_3$", "Graphene–Borophene", r"Graphene–B$_4$C$_3$"]
prefixes = ["E_Graphene-BC3", "F_Graphene-Borophene", "G_Graphene-B4C3"]
for ax, prefix, material, letter in zip(axes.flat, prefixes, materials, "abc"):
    panel(ax, prefix + ".png", f"({letter}) {material}")
axes[1, 1].axis("off")
axes[1, 1].legend(handles=[
    Line2D([], [], ls="none", marker="o", ms=8, color="#58B947", label="Boron"),
    Line2D([], [], ls="none", marker="o", ms=8, color="#A67655", label="Carbon"),
    Patch(facecolor="none", edgecolor="#FF8000", label="Unit cell"),
], loc="center", frameon=True)
save(fig, "proj1.3.pdf")


fig, axes = plt.subplots(3, 3, figsize=(10, 6.1))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.015, top=0.985,
                    wspace=0.025, hspace=0.04)
for col, (prefix, material) in enumerate(zip(prefixes, materials)):
    for row, (suffix, view) in enumerate(zip(
            ["top1", "side1a", "bottom1"], ["Top view", "Side view", "Bottom view"])):
        heading = f"({chr(97 + col)}) {material}" if row == 0 else view
        panel(axes[row, col], prefix + "_" + suffix + ".png", heading)
save(fig, "proj1.4.pdf")

(HERE / "structure_manifest.json").write_text(json.dumps({
    "source_directory": "2_Structure_and_CDD",
    "source_notebooks": ["0.1_structure_figure.ipynb", "2.0_charge_density_differences_figure.ipynb"],
    "changes": "Original raster pixels; new native panel tabs; original bond-marker positions and colours.",
    "files": manifest,
}, indent=2) + "\n")
