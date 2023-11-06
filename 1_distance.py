### Distance
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# Free energy versus lattice constant and distance

#%% Selected cases

from vmatplot.lattice_bilayer import specify_bilayer_lattice, plot_bilayer_lattice

selected_G_BC3 = specify_bilayer_lattice("0_Selected_Lattice/E_Graphene-BC3_Hollow")
selected_G_Borophene = specify_bilayer_lattice("0_Selected_Lattice/F_Graphene-Borophene_Top")
selected_G_B4C3 = specify_bilayer_lattice("0_Selected_Lattice/G_Graphene-B4C3_Top")

#%% Graphene - BC₃ Hollow

plot_bilayer_lattice("Graphene-BC₃ (Hollow)","0_Lattice_Distance/E_Graphene-BC3_Hollow/lattice_distance.dat", "Blues_r", "blue", "0_Selected_Lattice/E_Graphene-BC3_Hollow")

#%% Graphene - Borophene Top

plot_bilayer_lattice("Graphene-Borophene (Top)","0_Lattice_Distance/F_Graphene-Borophene_Top/lattice_distance.dat", "Oranges_r", "orange", "0_Selected_Lattice/F_Graphene-Borophene_Top")

#%% Graphene - B₄C₃ Top

plot_bilayer_lattice("Graphene-B₄C₃ (Top)","0_Lattice_Distance/G_Graphene-B4C3_Top/lattice_distance.dat", "Purples_r", "violet", "0_Selected_Lattice/G_Graphene-B4C3_Top")

# %%
