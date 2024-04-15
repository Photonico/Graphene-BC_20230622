#### Absorption coefficient plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

from vmatplot.dielectric_function_plotting import create_matters_dielectric_function

def create_matters_absorption(*args):
    # data = create_matters_dielectric_function(dielectric_list)
    # data[0] = current curve label
    # data[1] = dielectric data
    # data[2] = color family
    # data[3] = alpha
    # data[4] = linewidth
    return create_matters_dielectric_function(*args)
