### Distance
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# Free energy versus kpoints constant

#%% BC3

from vmatplot.kpoints import summarize_kpoints_free_energy, plot_kpoints_free_energy

summarize_kpoints_free_energy("1_Kpoints/BC3")

plot_kpoints_free_energy("BC₃","1_Kpoints/BC3/kpoints_energy.dat","X", 11, 23)

#%% Graphene - BC3: Hollow

summarize_kpoints_free_energy("1_Kpoints/Graphene_BC3_Hollow")

plot_kpoints_free_energy("Graphene-BC₃: Hollow","1_Kpoints/Graphene_BC3_Hollow/kpoints_energy.dat","X", 9, 61, "violet")

# %%
