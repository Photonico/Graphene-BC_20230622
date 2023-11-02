### DoS
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# Electronic density of state versus energy

#%% BC₃

from Store.DoS import extract_dos, plot_dos

dos_BC3_K11 = extract_dos("4_PDoS_PBE/A_BC3_K11/vasprun.xml")
dos_BC3_K41 = extract_dos("4_PDoS_PBE/A_BC3_K41/vasprun.xml")

matters = [["11 Kpoints", dos_BC3_K11, "grey"],["41 Kpoints", dos_BC3_K41, "blue"]]

plot_dos("BC₃", 6, 12, "PBE","total", matters)

#%% Borophene

dos_Borophene_K11 = extract_dos("4_PDoS_PBE/B_Borophene_K11/vasprun.xml")
dos_Borophene_K41 = extract_dos("4_PDoS_PBE/B_Borophene_K41/vasprun.xml")

matters = [["11 Kpoints", dos_Borophene_K11, "grey"], ["41 Kpoints", dos_Borophene_K41, "green"]]

plot_dos("Borophene",6, 12, "PBE","total", matters)

#%% B₄C₃

dos_B4C3_K11 = extract_dos("4_PDoS_PBE/C_B4C3_K11/vasprun.xml")
dos_B4C3_K41 = extract_dos("4_PDoS_PBE/C_B4C3_K41/vasprun.xml")

matters = [["11 Kpoints", dos_B4C3_K11, "grey"], ["41 Kpoints", dos_B4C3_K41, "violet"]]

plot_dos("B₄C₃", 6, 12, "PBE","total", matters)

#%% Graphene

dos_Graphene_K11 = extract_dos("4_PDoS_PBE/D_Graphene_K11/vasprun.xml")
dos_Graphene_K21 = extract_dos("4_PDoS_PBE/D_Graphene_K21/vasprun.xml")

matters = [["11 Kpoints", dos_Graphene_K11, "grey"], ["21 Kpoints", dos_Graphene_K21, "wine"]]

plot_dos("Graphene", 6, 3, "PBE","total", matters)

#%% Graphene - BC₃: Hollow

dos_E_K11 = extract_dos("4_PDoS_PBE/E_Graphene-BC3_Hollow_K11/vasprun.xml")
dos_E_K41 = extract_dos("4_PDoS_PBE/E_Graphene-BC3_Hollow_K41/vasprun.xml")

matters = [["11 Kpoints", dos_E_K11, "grey"], ["41 Kpoints", dos_E_K41, "purple"]]

plot_dos("Graphene-BC₃: Hollow", 6, 16, "PBE","total", matters)

#%% Graphene - Borophene: Top

dos_F_K11 = extract_dos("4_PDoS_PBE/F_Graphene-Borophene_Top_K11/vasprun.xml")
dos_F_K41 = extract_dos("4_PDoS_PBE/F_Graphene-Borophene_Top_K41/vasprun.xml")

matters = [["11 Kpoints", dos_F_K11, "grey"], ["41 Kpoints", dos_F_K41, "purple"]]

plot_dos("Graphene-Borophene: Top", 6, 16, "PBE","total", matters)

#%% Graphene - B₄C₃: Top

dos_G_K11 = extract_dos("4_PDoS_PBE/G_Graphene-B4C3_Top_K11/vasprun.xml")
dos_G_K21 = extract_dos("4_PDoS_PBE/G_Graphene-B4C3_Top_K21/vasprun.xml")
dos_G_K31 = extract_dos("4_PDoS_PBE/G_Graphene-B4C3_Top_K31/vasprun.xml")
dos_G_K41 = extract_dos("4_PDoS_PBE/G_Graphene-B4C3_Top_K41/vasprun.xml")

matters = [["11 Kpoints", dos_G_K11, "grey"],
           ["21 Kpoints", dos_G_K21, "blue"],
           ["31 Kpoints", dos_G_K31, "violet"],
           ["41 Kpoints", dos_G_K41, "purple"]]

plot_dos("Graphene-B₄C₃: Top", 6, 16, "PBE","total", matters)

# %%
