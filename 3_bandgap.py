### Bandgap
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# Bandgap of monolayers and bilayers

#%% Single layer BC3

from Store.bandgap import read_bandgap

system_PBE = "Single layer BC3 (PBE)"
gap_PBE = read_bandgap("3_Bandstructure_PBE/A_BC3/OUTCAR")

print(f"Matter: {system_PBE}")
print(f"LUMO band: {gap_PBE[2]:.6f};")
print(f"HOMO band: {gap_PBE[1]:.6f};")
print(f"Band gap: {gap_PBE[0]:.6f}.")

#%% Single layer Borophene

system_PBE = "Single layer Borophene (PBE)"
gap_PBE = read_bandgap("3_Bandstructure_PBE/B_Borophene/OUTCAR")

print(f"Matter: {system_PBE}")
print(f"LUMO band: {gap_PBE[2]:.6f};")
print(f"HOMO band: {gap_PBE[1]:.6f};")
print(f"Band gap: {gap_PBE[0]:.6f}.")

#%% Single layer B4C3

system_PBE = "Single layer B4C3 (PBE)"
gap_PBE = read_bandgap("3_Bandstructure_PBE/C_B4C3/OUTCAR")

print(f"Matter: {system_PBE}")
print(f"LUMO band: {gap_PBE[2]:.6f};")
print(f"HOMO band: {gap_PBE[1]:.6f};")
print(f"Band gap: {gap_PBE[0]:.6f}.")

#%% Single layer Graphene

system_PBE = "Single layer Graphene (PBE)"
gap_PBE = read_bandgap("3_Bandstructure_PBE/D_Graphene_small/OUTCAR")

print(f"Matter: {system_PBE}")
print(f"LUMO band: {gap_PBE[2]:.6f};")
print(f"HOMO band: {gap_PBE[1]:.6f};")
print(f"Band gap: {gap_PBE[0]:.6f}.")

#%% Bilayer Graphene - BC3 Hollow

system_PBE = "Bilayer Graphene - BC3 Hollow (PBE)"
gap_PBE = read_bandgap("3_Bandstructure_PBE/E_Graphene-BC3_Hollow/OUTCAR")

print(f"Matter: {system_PBE}")
print(f"LUMO band: {gap_PBE[2]:.6f};")
print(f"HOMO band: {gap_PBE[1]:.6f};")
print(f"Band gap: {gap_PBE[0]:.6f}.")

#%% Bilayer Graphene - Borophene Top

system_PBE = "Bilayer Graphene - Borophene Top (PBE)"
gap_PBE = read_bandgap("3_Bandstructure_PBE/F_Graphene-Borophene_Top/OUTCAR")

print(f"Matter: {system_PBE}")
print(f"LUMO band: {gap_PBE[2]:.6f};")
print(f"HOMO band: {gap_PBE[1]:.6f};")
print(f"Band gap: {gap_PBE[0]:.6f}.")

#%% Bilayer Graphene - B4C3 Top

system_PBE = "Bilayer Graphene - B4C3 Top (PBE)"
gap_PBE = read_bandgap("3_Bandstructure_PBE/G_Graphene-B4C3_Top/OUTCAR")

print(f"Matter: {system_PBE}")
print(f"LUMO band: {gap_PBE[2]:.6f};")
print(f"HOMO band: {gap_PBE[1]:.6f};")
print(f"Band gap: {gap_PBE[0]:.6f}.")

# %%
