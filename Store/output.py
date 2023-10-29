#### Output settings
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import os

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
                "\t[3]: legend location"
    default_style = {"length": 10,
                     "width": 6,
                     "dpi": 196,
                     "font_style": "serif"}
    default_params = {"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
        "axes.titlesize": 18, "axes.labelsize": 14, "figure.facecolor": "w"}
    default_legend = "upper right"
    if len(args) == 0:
        return (default_style["length"],default_style["width"]), default_style["dpi"], default_params, default_legend
    if len(args) == 1:
        if args[0] == "help":
            print(help_info)
            return
        else:
            return (args[0], 6), 196, default_params, default_legend
    if len(args) == 2:
        return (args[0], args[1]), 196, default_params, default_legend
    if len(args) == 3:
        return (args[0], args[1]), args[2], default_params, default_legend
    if len(args) == 4:
        customized_params = {"text.usetex": False, "font.family": args[3], "mathtext.fontset": "cm",
            "axes.titlesize": 18, "axes.labelsize": 14, "figure.facecolor": "w"}
        return (args[0], args[1]), args[2], customized_params, default_legend
    if len(args) == 5:
        customized_params = {"text.usetex": False, "font.family": args[3], "mathtext.fontset": "cm",
            "axes.titlesize": 18, "axes.labelsize": 14, "figure.facecolor": "w"}
        return (args[0], args[1]), args[2], customized_params, args[4]

def color_sampling(color_family):
    help_info = "Usage: color_family(color_family)\n" + \
                "Input the name of color family will return a series colors." + \
                "Color families: Grey, Red, Orange, Yellow, Green, Blue, Violet, Purple\n" + \
                "Return values:\n" + \
                "color[0]: deep color\n" + \
                "color[1]: major color\n" + \
                "color[2]: shallow color\n" + \
                "color[3]: comparison color 1\n" + \
                "color[4]: comparison color 2"
    # Check if the user asked for help
    if color_family == "help":
        print(help_info)
        return
    color_set = []
    if color_family in ("Grey", "grey"):
        color_set.append("#3C3C3C")
        color_set.append("#787878")
        color_set.append("#B4B4B4")
        color_set.append("#FA961E")
        color_set.append("#F078FF")
        return color_set
    if color_family in ("Red", "red"):
        color_set.append("#C81423")
        color_set.append("#E1322D")
        color_set.append("#FF644B")
        color_set.append("#FA8C00")
        color_set.append("#8C64F0")
        return color_set
    if color_family in ("Orange", "orange"):
        color_set.append("#EB731E")
        color_set.append("#FA8C00")
        color_set.append("#FFA03C")
        color_set.append("#FAC81E")
        color_set.append("#1473E1")
        return color_set
    if color_family in ("Yellow", "yellow"):
        color_set.append("#EBC31E")
        color_set.append("#FAC81E")
        color_set.append("#FFD71E")
        color_set.append("#FA8C00")
        color_set.append("#28AF3C")
        return color_set
    if color_family in ("Green", "green"):
        color_set.append("#286E41")
        color_set.append("#28AF3C")
        color_set.append("#5FDC5A")
        color_set.append("#FFD71E")
        color_set.append("#FA8C00")
        return color_set
    if color_family in ("Blue", "blue", "Azure", "azure"):
        color_set.append("#145AAA")
        color_set.append("#1473E1")
        color_set.append("#1EB4FF")
        color_set.append("#8C64F0")
        color_set.append("#FA8C00")
        return color_set
    if color_family in ("Violet", "violet"):
        color_set.append("#5A3CBE")
        color_set.append("#8C64F0")
        color_set.append("#AF96FF")
        color_set.append("#D25ADC")
        color_set.append("#FA8C00")
        return color_set
    if color_family in ("Purple", "purple"):
        color_set.append("#AA3CB9")
        color_set.append("#D25ADC")
        color_set.append("#F078FF")
        color_set.append("#1473E1")
        color_set.append("#FA8C00")
        return color_set

    if color_family in ("Brown", "brown"):
        color_set.append("#966450")#
        color_set.append("#B47D50")#
        color_set.append("#D29650")
        color_set.append("#1473E1")#
        color_set.append("#FA8C00")#
        return color_set
