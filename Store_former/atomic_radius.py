#### Demonstration for Graphene - BC3
# pylint: disable = C0103, C0114, R0914, R0915

## Physical constants
# Atomic radius in Angstrom

def atomic_radius_function():
    """Atomic radius in Angstrom"""
    unit = "Angstrom"

    # 1st period
    Radius_H  = 0.53
    Radius_He = 0.31

    # 2nd period
    Radius_Li = 1.52
    Radius_Be = 1.12
    Radius_B  = 0.82
    Radius_C  = 0.77
    Radius_N  = 0.75
    Radius_O  = 0.73
    Radius_F  = 0.72
    Radius_Ne = 0.71

    # 3rd period
    Radius_Na = 1.86
    Radius_Mg = 1.60
    Radius_Al = 1.43
    Radius_Si = 1.17
    Radius_P  = 1.10
    Radius_S  = 1.04
    Radius_Cl = 0.99
    Radius_Ar = 0.98

    # 4th period
    Radius_K  = 2.31
    Radius_Ca = 1.97
    Radius_Sc = 1.62
    Radius_Ti = 1.45
    Radius_V  = 1.33
    Radius_Cr = 1.35
    Radius_Mn = 1.35
    Radius_Fe = 1.26
    Radius_Co = 1.25
    Radius_Ni = 1.24
    Radius_Cu = 1.28
    Radius_Zn = 1.33
    Radius_Ga = 1.41
    Radius_Ge = 1.22
    Radius_As = 1.21
    Radius_Se = 1.16
    Radius_Br = 1.14
    Radius_Kr = 1.10

    # 5th period
    Radius_Rb = 2.44
    Radius_Sr = 2.15
    Radius_Y  = 1.80
    Radius_Zr = 1.57
    Radius_Nb = 1.41
    Radius_Mo = 1.36
    Radius_Tc = 1.35
    Radius_Ru = 1.34
    Radius_Rh = 1.34
    Radius_Pd = 1.37
    Radius_Ag = 1.44
    Radius_Cd = 1.49
    Radius_In = 1.66
    Radius_Sn = 1.62
    Radius_Sb = 1.41
    Radius_Te = 1.37
    Radius_I  = 1.33
    Radius_Xe = 1.31

    return (unit, Radius_H, Radius_He,
            Radius_Li, Radius_Be, Radius_B, Radius_C, Radius_N, Radius_O, Radius_F, Radius_Ne,
            # 3rd period
            Radius_Na, Radius_Mg, Radius_Al, Radius_Si, Radius_P, Radius_S, Radius_Cl, Radius_Ar,
            # 4th period
            Radius_K, Radius_Ca, Radius_Sc, Radius_Ti, Radius_V, Radius_Cr,
            Radius_Mn, Radius_Fe, Radius_Co, Radius_Ni, Radius_Cu, Radius_Zn,
            Radius_Ga, Radius_Ge, Radius_As, Radius_Se, Radius_Br, Radius_Kr,
            # 5th period
            Radius_Rb, Radius_Sr, Radius_Y, Radius_Zr, Radius_Nb, Radius_Mo,
            Radius_Tc, Radius_Ru, Radius_Rh, Radius_Pd, Radius_Ag, Radius_Cd,
            Radius_In, Radius_Sn, Radius_Sb, Radius_Te, Radius_I, Radius_Xe)

atomic_radius = atomic_radius_function()
