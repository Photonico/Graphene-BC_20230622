### Distance
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# Free energy versus lattice constant and distance

#%% Selected cases

from Store.lattice_bilayer import specify_bilayer_lattice
from Store.lattice_plotting import plot_bilayer_lattice

selected_G_BC3 = specify_bilayer_lattice("2_Self_Consistent/E_Graphene-BC3_Hollow_K41")
selected_G_Borophene = specify_bilayer_lattice("2_Self_Consistent/F_Graphene-Borophene_Top_K41")
selected_G_B4C3 = specify_bilayer_lattice("2_Self_Consistent/G_Graphene-B4C3_Top_K41")

#%% Graphene - BC₃ Hollow

plot_bilayer_lattice("Graphene-BC₃ (Hollow)","0_Lattice_distance/E_Graphene-BC3_Hollow/lattice_distance.dat", "Blues_r", "blue", "2_Self_Consistent/E_Graphene-BC3_Hollow_K41")

#%% Graphene - Borophene Top

plot_bilayer_lattice("Graphene-Borophene (Top)","0_Lattice_distance/F_Graphene-Borophene_Top/lattice_distance.dat", "Oranges_r", "orange")

#%% Graphene - B₄C₃ Top

plot_bilayer_lattice("Graphene-B₄C₃ (Top)","0_Lattice_distance/G_Graphene-B4C3_Top/lattice_distance.dat", "Purples_r", "violet")

# %%
