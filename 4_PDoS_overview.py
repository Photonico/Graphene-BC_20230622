### Electronic density of state
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

## Electronic density of state versus energy & total PDoS

#%% BC₃ DoS

from Store.DoS import plot_dos
from Store.PDoS import plot_total_pdos

matters_dir = [["17 Kpoints", "4_PDoS_PBE/A_BC3_K17", "grey"],
               ["33 Kpoints", "4_PDoS_PBE/A_BC3_K33", "blue"],
               ["65 Kpoints", "4_PDoS_PBE/A_BC3_K33", "wine"],]

plot_dos("BC₃", 6, 10, "PBE","total", matters_dir)

#%% BC₃ total PDoS

plot_total_pdos("BC₃", 6, 10, "PBE", "total", "4_PDoS_PBE/A_BC3_K65", "blue")

#%% Borophene DoS

matters_dir = [["17 Kpoints", "4_PDoS_PBE/B_Borophene_K17", "grey"],
               ["33 Kpoints", "4_PDoS_PBE/B_Borophene_K33", "blue"],
               ["65 Kpoints", "4_PDoS_PBE/B_Borophene_K65", "wine"]]

plot_dos("Borophene",6, 10, "PBE","total", matters_dir)

#%% Borophene total PDoS

plot_total_pdos("Borophene", 6, 10, "PBE", "total", "4_PDoS_PBE/B_Borophene_K65", "blue")

#%% B₄C₃ DoS

matters_dir = [["17 Kpoints", "4_PDoS_PBE/C_B4C3_K17", "grey"],
               ["33 Kpoints", "4_PDoS_PBE/C_B4C3_K33", "blue"],
               ["65 Kpoints", "4_PDoS_PBE/C_B4C3_K65", "wine"]]

plot_dos("B₄C₃", 6, 10, "PBE","total", matters_dir)

#%% B₄C₃ total PDoS

plot_total_pdos("B₄C₃", 6, 10, "PBE", "total", "4_PDoS_PBE/C_B4C3_K65", "blue")

#%% Graphene DoS

matters_dir = [["33 Kpoints", "4_PDoS_PBE/D_Graphene_K33", "grey"],
               ["65 Kpoints", "4_PDoS_PBE/D_Graphene_K65", "blue"],
               ["129 Kpoints", "4_PDoS_PBE/D_Graphene_K129", "wine"],]
            #    ["from BS", "4_PDoS_PBE/D_Graphene_K33", "brown"]]

plot_dos("Graphene", 6, 1.25, "PBE","total", matters_dir)

#%% Graphene total DoS

plot_total_pdos("Graphene", 6, 1.25, "PBE", "total", "4_PDoS_PBE/D_Graphene_K129", "blue")

#%% Graphene - BC₃: Hollow DoS

matters_dir = [["17 Kpoints", "4_PDoS_PBE/E_Graphene-BC3_Hollow_K17", "grey"],
           ["33 Kpoints", "4_PDoS_PBE/E_Graphene-BC3_Hollow_K33", "blue"],
           ["65 Kpoints", "4_PDoS_PBE/E_Graphene-BC3_Hollow_K65", "wine"]]

plot_dos("Graphene-BC₃: Hollow", 6, 12, "PBE","total", matters_dir)

#%% Graphene - BC₃: Hollow total PDoS

plot_total_pdos("Graphene-BC₃: Hollow", 6, 12, "PBE", "total", "4_PDoS_PBE/E_Graphene-BC3_Hollow_K65", "blue")

#%% Graphene - Borophene: Top DoS

matters_dir = [["17 Kpoints", "4_PDoS_PBE/F_Graphene-Borophene_Top_K17", "grey"],
               ["33 Kpoints", "4_PDoS_PBE/F_Graphene-Borophene_Top_K33", "blue"],
               ["65 Kpoints", "4_PDoS_PBE/F_Graphene-Borophene_Top_K65", "wine"],]

plot_dos("Graphene-Borophene: Top", 6, 12, "PBE","total", matters_dir)

#%% Graphene - Borophene: Top total PDoS

plot_total_pdos("Graphene-Borophene: Top", 6, 12, "PBE", "total", "4_PDoS_PBE/F_Graphene-Borophene_Top_K65", "blue")

#%% Graphene - B₄C₃: Top DoS

matters_dir = [["17 Kpoints", "4_PDoS_PBE/G_Graphene-B4C3_Top_K17", "grey"],
           ["33 Kpoints", "4_PDoS_PBE/G_Graphene-B4C3_Top_K33", "blue"],
           ["65 Kpoints", "4_PDoS_PBE/G_Graphene-B4C3_Top_K65", "wine"]]

plot_dos("Graphene-B₄C₃: Top", 6, 12, "PBE","total", matters_dir)

#%% Graphene - B₄C₃: Top total PDoS

plot_total_pdos("Graphene-B₄C₃: Top", 6, 12, "PBE", "total", "4_PDoS_PBE/G_Graphene-B4C3_Top_K65", "blue")

#%%
