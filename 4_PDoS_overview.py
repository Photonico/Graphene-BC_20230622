### DoS
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

## Electronic density of state versus energy & total PDoS

#%% BC₃ DoS

from Store.DoS import extract_dos, plot_dos
from Store.PDoS import extract_pdos, plot_total_pdos

dos_BC3_K17 = extract_dos("4_PDoS_PBE/A_BC3_K17/vasprun.xml")
dos_BC3_K33 = extract_dos("4_PDoS_PBE/A_BC3_K33/vasprun.xml")
dos_BC3_K65 = extract_dos("4_PDoS_PBE/A_BC3_K65/vasprun.xml")

matters = [["17 Kpoints", dos_BC3_K17, "grey"],
           ["33 Kpoints", dos_BC3_K33, "blue"],
           ["65 Kpoints", dos_BC3_K65, "wine"],]


plot_dos("BC₃", 6, 10, "PBE","total", matters)

#%% BC₃ total PDoS

pdos_BC3 = extract_pdos("4_PDoS_PBE/A_BC3_K65/vasprun.xml")

plot_total_pdos("BC₃", 6, 10, "PBE", "total", "blue", pdos_BC3)

#%% Borophene DoS

dos_Borophene_K17 = extract_dos("4_PDoS_PBE/B_Borophene_K17/vasprun.xml")
dos_Borophene_K33 = extract_dos("4_PDoS_PBE/B_Borophene_K33/vasprun.xml")
dos_Borophene_K65 = extract_dos("4_PDoS_PBE/B_Borophene_K65/vasprun.xml")

matters = [["17 Kpoints", dos_Borophene_K17, "grey"],
           ["33 Kpoints", dos_Borophene_K33, "blue"],
           ["65 Kpoints", dos_Borophene_K65, "wine"]]

plot_dos("Borophene",6, 10, "PBE","total", matters)

#%% Borophene total PDoS

pdos_Borophene = extract_pdos("4_PDoS_PBE/B_Borophene_K65/vasprun.xml")

plot_total_pdos("Borophene", 6, 10, "PBE", "total", "blue", pdos_Borophene)

#%% B₄C₃ DoS

dos_B4C3_K17 = extract_dos("4_PDoS_PBE/C_B4C3_K17/vasprun.xml")
dos_B4C3_K33 = extract_dos("4_PDoS_PBE/C_B4C3_K33/vasprun.xml")
dos_B4C3_K65 = extract_dos("4_PDoS_PBE/C_B4C3_K65/vasprun.xml")

matters = [["17 Kpoints", dos_B4C3_K17, "grey"],
           ["33 Kpoints", dos_B4C3_K33, "blue"],
           ["65 Kpoints", dos_B4C3_K65, "wine"]]

plot_dos("B₄C₃", 6, 10, "PBE","total", matters)

#%% B₄C₃ total PDoS

pdos_B4C3 = extract_pdos("4_PDoS_PBE/C_B4C3_K65/vasprun.xml")

plot_total_pdos("B₄C₃", 6, 10, "PBE", "total", "blue", pdos_B4C3)

#%% Graphene DoS

dos_Graphene_K33 = extract_dos("4_PDoS_PBE/D_Graphene_K33/vasprun.xml")
dos_Graphene_K65 = extract_dos("4_PDoS_PBE/D_Graphene_K65/vasprun.xml")
dos_Graphene_K129 = extract_dos("4_PDoS_PBE/D_Graphene_K129/vasprun.xml")

matters = [["33 Kpoints", dos_Graphene_K33, "grey"],
           ["65 Kpoints", dos_Graphene_K65, "blue"],
           ["129 Kpoints", dos_Graphene_K129, "wine"]]

plot_dos("Graphene", 6, 1.25, "PBE","total", matters)

#%% Graphene total DoS

pdos_Graphene = extract_pdos("4_PDoS_PBE/D_Graphene_K129/vasprun.xml")

plot_total_pdos("Graphene", 6, 1.25, "PBE", "total", "blue", pdos_Graphene)

#%% Graphene - BC₃: Hollow DoS

dos_E_K17 = extract_dos("4_PDoS_PBE/E_Graphene-BC3_Hollow_K17/vasprun.xml")
dos_E_K33 = extract_dos("4_PDoS_PBE/E_Graphene-BC3_Hollow_K33/vasprun.xml")
dos_E_K65 = extract_dos("4_PDoS_PBE/E_Graphene-BC3_Hollow_K65/vasprun.xml")

matters = [["17 Kpoints", dos_E_K17, "grey"],
           ["33 Kpoints", dos_E_K33, "blue"],
           ["65 Kpoints", dos_E_K65, "wine"]]

plot_dos("Graphene-BC₃: Hollow", 6, 12, "PBE","total", matters)

#%% Graphene - BC₃: Hollow total PDoS

pdos_E = extract_pdos("4_PDoS_PBE/E_Graphene-BC3_Hollow_K65/vasprun.xml")

plot_total_pdos("Graphene-BC₃: Hollow", 6, 12, "PBE", "total", "blue", pdos_E)

#%% Graphene - Borophene: Top DoS

dos_F_K17 = extract_dos("4_PDoS_PBE/F_Graphene-Borophene_Top_K17/vasprun.xml")
dos_F_K33 = extract_dos("4_PDoS_PBE/F_Graphene-Borophene_Top_K33/vasprun.xml")
dos_F_K65 = extract_dos("4_PDoS_PBE/F_Graphene-Borophene_Top_K65/vasprun.xml")

matters = [["17 Kpoints", dos_F_K17, "grey"],
           ["33 Kpoints", dos_F_K33, "blue"],
           ["65 Kpoints", dos_F_K65, "wine"],]

plot_dos("Graphene-Borophene: Top", 6, 12, "PBE","total", matters)

#%% Graphene - Borophene: Top total PDoS

pdos_F = extract_pdos("4_PDoS_PBE/F_Graphene-Borophene_Top_K65/vasprun.xml")

plot_total_pdos("Graphene-Borophene: Top", 6, 12, "PBE", "total", "blue", pdos_F)

#%% Graphene - B₄C₃: Top DoS

dos_G_K17 = extract_dos("4_PDoS_PBE/G_Graphene-B4C3_Top_K17/vasprun.xml")
dos_G_K33 = extract_dos("4_PDoS_PBE/G_Graphene-B4C3_Top_K33/vasprun.xml")
dos_G_K65 = extract_dos("4_PDoS_PBE/G_Graphene-B4C3_Top_K65/vasprun.xml")

matters = [["17 Kpoints", dos_G_K17, "grey"],
           ["33 Kpoints", dos_G_K33, "blue"],
           ["65 Kpoints", dos_G_K65, "wine"]]

plot_dos("Graphene-B₄C₃: Top", 6, 12, "PBE","total", matters)

#%% Graphene - B₄C₃: Top total PDoS

pdos_G = extract_pdos("4_PDoS_PBE/G_Graphene-B4C3_Top_K65/vasprun.xml")

plot_total_pdos("Graphene-B₄C₃: Top", 6, 12, "PBE", "total", "blue", pdos_G)

#%%
