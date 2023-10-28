# Read bandgap from bandstructure calculation
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

def read_bandgap(outcar_path):
    # Check if the user asked for help
    if outcar_path == "help":
        print("Please use this function to OUTCAR of the bandstructure calculation.")
        return "Help provided."

    with open(outcar_path, "r", encoding="utf-8") as outcar_file:
        content = outcar_file.readlines()

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

    # print(f"File: {outcar_path}")
    # print(f"HOMO: band: {homo} E= {e1}")
    # print(f"LUMO: band: {lumo} E= {e2}")
    # print(f"Bandgap: {bandgap}")

    return bandgap, e1, e2
