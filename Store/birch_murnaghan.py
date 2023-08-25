# Birch-Murnaghan equation of state

import numpy as np
from scipy.optimize import leastsq

def birch_murnaghan_eos(params, vol):
    """Birch-Murnaghan equation of state."""
    E0, B0, Bp, V0 = params 
    eta = (vol/V0)**(1.0/3.0)
    E = E0 + 9.0*B0*V0/16.0 * (eta**2-1.0)**2 * (6.0 + Bp*(eta**2-1.0) - 4.0*eta**2)
    return E

def fit_eos(lattice_values, energy_values):
    """Fit energy vs volume data using Birch-Murnaghan equation of state."""
    
    # Convert input lists to numpy arrays if they are not already
    lattice_values = np.array(lattice_values)
    energy_values = np.array(energy_values)
    
    # Assuming volume is the cube of the lattice parameter.
    # Modify as per the specifics of your data and system.
    vol = lattice_values**3

    # Initial polynomial fit to get initial estimates
    a, b, c = np.polyfit(vol, energy_values, 2)
    V0 = -b/(2*a)
    E0 = a*V0**2 + b*V0 + c
    B0 = 2*a*V0
    Bp = 4.0
    initial_params = [E0, B0, Bp, V0]

    # Least squares fitting
    objective_func = lambda params, y, x: y - birch_murnaghan_eos(params, x)
    params, ier = leastsq(objective_func, initial_params, args=(energy_values, vol))

    # Calculate fitted energy values using optimized parameters
    fitted_energies = birch_murnaghan_eos(params, vol)
    
    return fitted_energies
