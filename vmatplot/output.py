#### Output settings
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import os
import matplotlib.pyplot as plt

def vasprun_directory(directory="."):
    """Find folders with complete vasprun.xml and print incomplete ones."""
    complete_folders = []

    for dirpath, _, filenames in os.walk(directory):
        if "vasprun.xml" in filenames:
            file_name_xml = os.path.join(dirpath, "vasprun.xml")

            # Check if vasprun.xml is complete
            try:
                with open(file_name_xml, "r", encoding="utf-8") as f:
                    # Check the last few lines for the closing tag
                    last_lines = f.readlines()[-10:]  # read the last 10 lines
                    for line in last_lines:
                        if "</modeling>" in line or "</vasp>" in line:
                            complete_folders.append(dirpath)
                            break
                    else:
                        print(f"vasprun.xml in {dirpath} is incomplete.")
            except IOError as e:
                print(f"Error reading {file_name_xml}: {e}")

    return complete_folders

def canvas_setting(*args):
    help_info = "Usage: canvas_setting(length, width, dpi, font)\n" + \
                "The default setting is length: 10, width: 6, dpi: 196, font: 'Serif'" + \
                "The return values are :" + \
                "\t[0]: size_setting"+\
                "\t[1]: dpi"+\
                "\t[2]: figure params (font)"+\
                "\t[3]: Suptitle and subtitle fontsize"+\
                "\t[4]: legend location"
    default_style = {"length": 10,
                     "width": 6,
                     "dpi": 256,
                     "font_style": "serif"}
    default_params = {"text.usetex": False,
                      "font.family": "serif",
                      "mathtext.fontset": "cm",
                      "axes.titlesize": 20,
                      "axes.labelsize": 16,
                      "xtick.labelsize": 14,
                      "ytick.labelsize": 14,
                      "legend.fontsize": 12,
                      "figure.facecolor": "w"}
    default_suptitle = 20
    default_subtitle = 18
    default_legend = "upper right"
    if len(args) == 0:
        return (default_style["length"],default_style["width"]), default_style["dpi"], default_params, (default_suptitle, default_subtitle), default_legend
    if len(args) == 1:
        if args[0] == "help":
            print(help_info)
            return
        else:
            return (args[0], 6), 196, default_params, (default_suptitle, default_subtitle), default_legend
    if len(args) == 2:
        return (args[0], args[1]), 196, default_params, (default_suptitle, default_subtitle), default_legend
    if len(args) == 3:
        return (args[0], args[1]), args[2], default_params, (default_suptitle, default_subtitle), default_legend
    if len(args) == 4:
        customized_params = {"text.usetex": False,
                             "font.family": args[3],
                             "mathtext.fontset": "cm",
                             "axes.titlesize": 20,
                             "axes.labelsize": 16,
                             "xtick.labelsize": 14,
                             "ytick.labelsize": 14,
                             "legend.fontsize": 12,
                             "figure.facecolor": "w"}
        return (args[0], args[1]), args[2], customized_params, (default_suptitle, default_subtitle), default_legend
    if len(args) == 6:
        customized_params = {"text.usetex": False,
                             "font.family": args[3],
                             "mathtext.fontset": "cm",
                             "axes.titlesize": 20,
                             "axes.labelsize": 16,
                             "xtick.labelsize": 14,
                             "ytick.labelsize": 14,
                             "legend.fontsize": 12,
                             "figure.facecolor": "w"}
        return (args[0],args[1]), args[2], customized_params, (args[4],args[5]), default_legend
    if len(args) == 7:
        customized_params = {"text.usetex": False,
                             "font.family": args[3],
                             "mathtext.fontset": "cm",
                             "axes.titlesize": 20,
                             "axes.labelsize": 16,
                             "xtick.labelsize": 14,
                             "ytick.labelsize": 14,
                             "legend.fontsize": 12,
                             "figure.facecolor": "w"}
        return (args[0],args[1]), args[2], customized_params, (args[4],args[5]), args[6]

# def fermi_energy_color():
#     return "#64B4DC"

def color_sampling(color_family):
    help_info = "Usage: color_family(color_family)\n" + \
                "Input the name of color family will return a series colors." + \
                "Color families: Grey, Red, Orange, Yellow, Green, Blue, Violet, Purple, Wine, Brown, Orbit\n" + \
                "Return values:\n" + \
                "color[0]: deep color\n" + \
                "color[1]: major color\n" + \
                "color[2]: shallow color\n" + \
                "color[3]: comparison color 1\n" + \
                "color[4]: comparison color 2\n" + \
                "color[5]: comparison color 3\n" + \
                "color[6]: comparison color 4\n" + \
                "color[7]: comparison color 5\n" + \
                "color[8]: comparison color 6\n"
    # Check if the user asked for help
    if color_family == "help":
        print(help_info)
        return

    color_set = []
    if color_family in ("Default", "default", "Normal", "normal", "Orbital", "orbital", "Orbitals", "orbitals"):
        color_set.append("#145AAA") # colors[0]: Base
        color_set.append("#1478E1") # colors[1]: total
        color_set.append("#14A0FF") # colors[2]: Integral

        color_set.append("#8C64F0") # colors[3]: s-orbital
        color_set.append("#D25ADC") # colors[4]: px-orbital
        color_set.append("#F03C64") # colors[5]: py-orbital
        color_set.append("#FA8C00") # colors[6]: pz-orbital
        color_set.append("#96C800") # colors[7]: d-orbital
        color_set.append("#14AFAF") # colors[8]: f-orbital
        return color_set

    if color_family in ("Grey", "grey", "Gray", "grey"):
        color_set.append("#3C3C3C")
        color_set.append("#787878")
        color_set.append("#B4B4B4")

        color_set.append("#E1322D")
        color_set.append("#FA8C00")
        color_set.append("#FAC828")
        color_set.append("#28AF3C")
        color_set.append("#1478E1")
        color_set.append("#8C64F0")
        return color_set

    if color_family in ("Silver", "silver"):
        color_set.append("#787D8C")
        color_set.append("#AAAFBE")
        color_set.append("#C8CDD7")

        color_set.append("#E1322D")
        color_set.append("#FA8C00")
        color_set.append("#FAC828")
        color_set.append("#28AF3C")
        color_set.append("#1478E1")
        color_set.append("#8C64F0")
        return color_set

    if color_family in ("Red", "red"):
        color_set.append("#C81423")
        color_set.append("#E1322D")
        color_set.append("#FF644B")

        color_set.append("#FA8C00")
        color_set.append("#FAC828")
        color_set.append("#28AF3C")
        color_set.append("#1478E1")
        color_set.append("#8C64F0")
        color_set.append("#D25ADC")
        return color_set

    if color_family in ("Orange", "orange"):
        color_set.append("#EB731E")
        color_set.append("#FA8C00")
        color_set.append("#FFA03C")

        color_set.append("#FAC828")
        color_set.append("#28AF3C")
        color_set.append("#1478E1")
        color_set.append("#8C64F0")
        color_set.append("#D25ADC")
        color_set.append("#F03C64")
        return color_set

    if color_family in ("Yellow", "yellow", "Gold", "gold"):
        color_set.append("#EBC31E")
        color_set.append("#FAC828")
        color_set.append("#FFD732")

        color_set.append("#28AF3C")
        color_set.append("#1478E1")
        color_set.append("#8C64F0")
        color_set.append("#D25ADC")
        color_set.append("#F03C64")
        color_set.append("#FA8C00")
        return color_set

    if color_family in ("Green", "green"):
        color_set.append("#238C4B")
        color_set.append("#28AF3C")
        color_set.append("#73C81E")

        color_set.append("#1478E1")
        color_set.append("#8C64F0")
        color_set.append("#D25ADC")
        color_set.append("#F03C64")
        color_set.append("#FA8C00")
        color_set.append("#FAC828")
        return color_set

    if color_family in ("Blue", "blue", "Azure", "azure"):
        color_set.append("#145AAA") # colors[0]
        color_set.append("#1478E1") # colors[1]
        color_set.append("#14A0FF") # colors[2]

        color_set.append("#8C64F0") # colors[3]
        color_set.append("#D25ADC") # colors[4]
        color_set.append("#F03C64") # colors[5]
        color_set.append("#FA8C00") # colors[6]
        color_set.append("#FAC828") # colors[7]
        color_set.append("#28AF3C") # colors[8]
        return color_set

    if color_family in ("Violet", "violet"):
        color_set.append("#643CC3")
        color_set.append("#8C64E1")
        color_set.append("#AF96FF")

        color_set.append("#D25ADC")
        color_set.append("#F03C64")
        color_set.append("#FA8C00")
        color_set.append("#FAC828")
        color_set.append("#28AF3C")
        color_set.append("#1478E1")
        return color_set

    if color_family in ("Purple", "purple"):
        color_set.append("#AA3CB9")
        color_set.append("#D25ADC")
        color_set.append("#F078FF")

        color_set.append("#F03C64")
        color_set.append("#FA8C00")
        color_set.append("#FAC828")
        color_set.append("#28AF3C")
        color_set.append("#1478E1")
        color_set.append("#8C64F0")
        return color_set

    if color_family in ("Wine", "wine"):
        color_set.append("#AA1E64")
        color_set.append("#C82364")
        color_set.append("#F03C64")

        color_set.append("#FA8C00")
        color_set.append("#FAC828")
        color_set.append("#28AF3C")
        color_set.append("#1478E1")
        color_set.append("#8C64F0")
        color_set.append("#D25ADC")
        return color_set

    if color_family in ("Brown", "brown"):
        color_set.append("#966450")
        color_set.append("#B47D50")
        color_set.append("#D29650")

        color_set.append("#FA8C00")
        color_set.append("#FAC828")
        color_set.append("#28AF3C")
        color_set.append("#1478E1")
        color_set.append("#8C64F0")
        color_set.append("#D25ADC")
        return color_set

    if color_family == "all_families":
        return ["Silver", "Grey", "Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Purple", "Wine", "Brown", "Default"]

def plot_color_families():
    color_families = color_sampling("all_families")
    all_colors = [color_sampling(family) for family in color_families]

    # Figure Settings
    fig_setting = canvas_setting(10,8)
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.rcParams.update(params)

    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    plt.title("Color families")

    for row, color_row in enumerate(all_colors):
        for col, color in enumerate(color_row):
            plt.gca().add_patch(plt.Rectangle((col, row), 1, 1, color=color))
            plt.text(col + 0.5, row + 0.5, color, ha="center", va="center", fontsize=8, color="white")

    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    max_length = max([len(colors) for colors in all_colors])
    plt.xlim(0, max_length)
    plt.ylim(0, len(all_colors))
    plt.xticks([])
    plt.yticks([])

    yaxis_offset = 0.5
    for i, label in enumerate(color_families):
        plt.text(-max_length*0.01, i + yaxis_offset, label, ha="right", va="center")

    plt.show()
    plt.tight_layout()
