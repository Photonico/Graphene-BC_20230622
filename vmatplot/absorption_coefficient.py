#### Absorption coefficient plotting
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

from vmatplot.dielectric_function_plotting import create_matters_dielectric_function

# Physical constants
hbar_ev = 4.135667662e-15
c_vacuum = 2.99792458e8
c_vacuum_nm = 2.99792458e17
pi = 3.141592654

def create_matters_absorption(*args):
    # data = create_matters_dielectric_function(dielectric_list)
    # data[0] = current curve label
    # data[1] = dielectric data
    # data[2] = color family
    # data[3] = alpha
    # data[4] = linewidth
    return create_matters_dielectric_function(*args)

def plot_absorption_XZ_col(title, absorption_list=None, inplane_boundary=(None, None), outplane_boundary=(None, None)):
    help_info = "Usage: absorption_XZ" + \
                "The independent value includes \n" +\
                "\t title, \n" +\
                "\t dielectric function data list, \n" +\
                "\t Inplane photon wavelenght range (Optional), \n" +\
                "\t Outplane photon wavelenght range (Optional). \n"
    if title in ["help", "Help"]:
        print(help_info)
