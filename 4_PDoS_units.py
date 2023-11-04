### PDoS
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# Projected electronic density of state versus energy

## Work space

#%% Elements analysis
from Store.PDoS import get_elements, extract_pdos, extract_element_pdos, extract_segment_pdos, plot_pdos_element

element_BC3 = get_elements("4_PDoS_PBE/A_BC3_K65/vasprun.xml")
print(f"The element analysis for BC₃ is \n {element_BC3} \n")
# BC₃: {'B': (1, 2), 'C': (3, 8)}

element_Borophene = get_elements("4_PDoS_PBE/B_Borophene_K65/vasprun.xml")
print(f"The element analysis for Borophene is \n {element_Borophene} \n")
# Borophene: {'B': (1, 8)}

element_B4C3 = get_elements("4_PDoS_PBE/C_B4C3_K65/vasprun.xml")
print(f"The element analysis for B₄C₃ is \n {element_B4C3} \n")
# B₄C₃: {'B': (1, 4), 'C': (5, 7)}

element_Graphene = get_elements("4_PDoS_PBE/D_Graphene_K129/vasprun.xml")
print(f"The element analysis for Graphene is \n {element_Graphene} \n")
# Graphene: {'C': (1, 2)}

element_G_BC3 = get_elements("4_PDoS_PBE/E_Graphene-BC3_Hollow_K65/vasprun.xml")
print(f"The element analysis for Graphene-BC₃ (Hollow) is \n {element_G_BC3} \n")
# Graphene-BC₃: {'B': (1, 2), 'C': (3, 16)}

element_G_Borophene = get_elements("4_PDoS_PBE/F_Graphene-Borophene_Top_K65/vasprun.xml")
print(f"The element analysis for Graphene-Borophene (Top) is \n {element_G_Borophene} \n")
# Graphene-Borophene: {'B': (1, 8), 'C': (9, 16)}

element_G_B4C3 = get_elements("4_PDoS_PBE/G_Graphene-B4C3_Top_K65/vasprun.xml")
print(f"The element analysis for Graphene-B₄C₃ (Top) is \n {element_G_B4C3} \n")
# Graphene-B₄C₃: {'B': (1, 4), 'C': (5, 15)}

#%% BC3

pdos_BC3_total = extract_pdos("/home/lu/Repos/Graphene-BC 2023/4_PDoS_PBE/A_BC3_K65/vasprun.xml")
pdos_BC3_B = extract_element_pdos("/home/lu/Repos/Graphene-BC 2023/4_PDoS_PBE/A_BC3_K65/vasprun.xml","B")
pdos_BC3_C = extract_element_pdos("/home/lu/Repos/Graphene-BC 2023/4_PDoS_PBE/A_BC3_K65/vasprun.xml","C")

plot_pdos_element("BC₃", 6, 10, "PBE", pdos_BC3_total, "Boron", pdos_BC3_B, "Carbon", pdos_BC3_C, "blue")

#%% Borophene

pdos_Borophene_total = extract_pdos("/home/lu/Repos/Graphene-BC 2023/4_PDoS_PBE/B_Borophene_K65/vasprun.xml")
pdos_Borophene_B = extract_element_pdos("/home/lu/Repos/Graphene-BC 2023/4_PDoS_PBE/B_Borophene_K65/vasprun.xml","B")

plot_pdos_element("Borophene", 6, 10, "PBE", pdos_Borophene_total, "Boron", pdos_Borophene_B, "blue")

#%% B4C3

pdos_B4C3_total = extract_pdos("/home/lu/Repos/Graphene-BC 2023/4_PDoS_PBE/C_B4C3_K65/vasprun.xml")
pdos_B4C3_B = extract_element_pdos("/home/lu/Repos/Graphene-BC 2023/4_PDoS_PBE/C_B4C3_K65/vasprun.xml","B")
pdos_B4C3_C = extract_element_pdos("/home/lu/Repos/Graphene-BC 2023/4_PDoS_PBE/C_B4C3_K65/vasprun.xml","C")

plot_pdos_element("B₄C₃", 6, 10, "PBE", pdos_B4C3_total, "Boron", pdos_B4C3_B, "Carbon", pdos_B4C3_C, "blue")

#%% Graphene

pdos_Graphene_total = extract_pdos("/home/lu/Repos/Graphene-BC 2023/4_PDoS_PBE/D_Graphene_K129/vasprun.xml")
pdos_Graphene_C = extract_element_pdos("/home/lu/Repos/Graphene-BC 2023/4_PDoS_PBE/D_Graphene_K129/vasprun.xml","C")

plot_pdos_element("Graphene", 6, 10, "PBE", pdos_Graphene_total, "Carbon", pdos_Graphene_C, "blue")

#%% Graphene-BC3: Hollow

pdos_E_total = extract_pdos("4_PDoS_PBE/E_Graphene-BC3_Hollow_K65/vasprun.xml")
pdos_E_B = extract_segment_pdos("4_PDoS_PBE/E_Graphene-BC3_Hollow_K65/vasprun.xml", 1, 2)
pdos_E_C_top = extract_segment_pdos("4_PDoS_PBE/E_Graphene-BC3_Hollow_K65/vasprun.xml", 3, 8)
pdos_E_C_bot = extract_segment_pdos("4_PDoS_PBE/E_Graphene-BC3_Hollow_K65/vasprun.xml", 9, 16)

plot_pdos_element("Graphene-BC₃: Hollow", 6, 10, "PBE", pdos_E_total, "Carbon in Graphene", pdos_E_C_bot, "Borophene in BC₃", pdos_E_B, "Carbon in BC₃", pdos_E_C_top, "blue")

#%% Graphene-Borophene: Top

pdos_F_total = extract_pdos("4_PDoS_PBE/F_Graphene-Borophene_Top_K65/vasprun.xml")
pdos_F_B = extract_segment_pdos("4_PDoS_PBE/F_Graphene-Borophene_Top_K65/vasprun.xml", 1, 8)
pdos_F_C = extract_segment_pdos("4_PDoS_PBE/F_Graphene-Borophene_Top_K65/vasprun.xml", 9, 16)

plot_pdos_element("Graphene-Borophene: Top", 6, 10, "PBE", pdos_F_total, "Carbon in Graphene", pdos_F_C, "Borophene in Borophene", pdos_F_B, "blue")

#%% Graphene-B4C3: Top

pdos_G_total = extract_pdos("4_PDoS_PBE/G_Graphene-B4C3_Top_K65/vasprun.xml")
pdos_G_B = extract_segment_pdos("4_PDoS_PBE/G_Graphene-B4C3_Top_K65/vasprun.xml", 1, 4)
pdos_G_C_top = extract_segment_pdos("4_PDoS_PBE/G_Graphene-B4C3_Top_K65/vasprun.xml", 5, 7)
pdos_G_C_bot = extract_segment_pdos("4_PDoS_PBE/G_Graphene-B4C3_Top_K65/vasprun.xml", 8, 15)

plot_pdos_element("Graphene-B₄C₃: Top", 6, 10, "PBE", pdos_G_total, "Carbon in Graphene", pdos_G_C_bot, "Borophene in B₄C₃", pdos_G_B, "Carbon in B₄C₃", pdos_G_C_top, "blue")

#%%
