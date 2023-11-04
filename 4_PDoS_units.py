### PDoS
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# Projected electronic density of state versus energy

## Work space

#%% Elements analysis
from Store.PDoS import get_elements, extract_pdos, extract_element_pdos
# from Store.PDoS_plotting import 

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

#%% Borophene

#%% B4C3

#%% Graphene

#%% Graphene-BC3: Hollow

#%% Graphene-Borophene: Top

#%% Graphene-B4C3: Top
