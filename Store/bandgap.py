# Read bandgap from bandstructure calculation

import os

def read_bandgap(filename):
    with open(filename, 'r') as f:
        content = f.readlines()

    # Get HOMO, LUMO, and NKPT
    homo = None
    lumo = None
    nkpt = None
    for line in content:
        if "NELECT" in line:
            homo = int(float(line.split()[2]) / 2)
            lumo = homo + 1
        if "NKPTS" in line:
            nkpt = int(line.split()[3])

    # Extract energies for HOMO and LUMO
    homo_energies = [float(line.split()[1]) for i, line in enumerate(content) if f"     {homo}     " in line]
    lumo_energies = [float(line.split()[1]) for i, line in enumerate(content) if f"     {lumo}     " in line]

    # Get maximum HOMO and minimum LUMO energy considering nkpt values
    e1 = sorted(homo_energies[:nkpt])[-1]
    e2 = sorted(lumo_energies[:nkpt])[0]
    bandgap = e2 - e1

    # print(f"File: {filename}")
    # print(f"HOMO: band: {homo} E= {e1}")
    # print(f"LUMO: band: {lumo} E= {e2}")
    # print(f"Bandgap: {bandgap}")

    return bandgap, e1, e2
