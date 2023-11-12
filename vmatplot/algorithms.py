#### General Algoriths
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import os
import numpy as np

from scipy.optimize import leastsq

def get_matrix_shape(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    return (rows, cols)

def transpose_matrix(matrix):
    return [list(row) for row in zip(*matrix)]

def compute_average(data_lines):
    """Define the function to compute the average of the last value in each line."""
    total = 0
    for line in data_lines:
        values = line.split()       # Split the line into individual values
        total += float(values[-1])  # Add the last value to the total
        print(line)
    return total / len(data_lines)  # Return the average

def birch_murnaghan_eos(params, vol):
    """Birch-Murnaghan equation of state."""
    E0, B0, Bp, V0 = params
    eta = (vol/V0)**(1.0/3.0)
    E = E0 + 9.0*B0*V0/16.0 * (eta**2-1.0)**2 * (6.0 + Bp*(eta**2-1.0) - 4.0*eta**2)
    return E

def fit_eos(lattice_values, energy_values, sample_count=10000):
    """Fit energy vs volume data using Birch-Murnaghan equation of state."""

    # Convert input lists to numpy arrays if they are not already
    lattice_values = np.array(lattice_values)
    energy_values = np.array(energy_values)

    # Resample lattice_values for output
    resampled_lattice = np.linspace(min(lattice_values), max(lattice_values), num=sample_count, endpoint=True)

    # Assuming volume is the cube of the lattice parameter.
    vol = lattice_values**3
    resampled_vol = resampled_lattice**3

    # Provide initial parameters for Birch-Murnaghan EOS
    E0 = min(energy_values)
    V0 = vol[np.argmin(energy_values)]
    B0 = 0.1  # Initial bulk modulus
    Bp = 4.0  # Initial value for its derivative

    initial_params = [E0, B0, Bp, V0]

    # Least squares fitting
    def objective_func(params, y, x):
        return y - birch_murnaghan_eos(params, x)

    params, _ = leastsq(objective_func, initial_params, args=(energy_values, vol))

    # Compute corresponding fitted energy values for resampled lattice
    fitted_energy = birch_murnaghan_eos(params, resampled_vol)

    return resampled_lattice, fitted_energy

def polynomially_fit_curve(lattice_list, free_energy_list = None, degree = None, sample_count = None):
    help_info = "Usage: polynomially_fit_curve(lattice_list, free_energy_list, degree, sample_count)\n" + \
                "sample_count here means the sampling numbers.\n"
    # Check if the user asked for help
    if lattice_list == "help":
        print(help_info)
        return
    # Ensure the other parameters are provided
    if free_energy_list is None or degree is None or sample_count is None:
        raise ValueError("Missing required parameters. Use 'help' for more information.")

    # Apply polynomial fitting to the data
    p = np.polyfit(lattice_list, free_energy_list, degree)
    # Generate a polynomial function from the fitted parameters
    f = np.poly1d(p)

    # Generate new x and y data using the polynomial function
    fitted_lattice = np.linspace(min(lattice_list), max(lattice_list), num=sample_count, endpoint=True)
    fitted_free_energy = f(fitted_lattice)
    return fitted_lattice, fitted_free_energy

def polynomially_fit_surface(lattice_list, distance_list = None, energy_list = None, degree = None, sample_count = None):
    # Help information
    help_info = "Usage: polynomially_fit_surface(lattice_list, distance_list, energy_list, degree, sample_count)\n" + \
                "degree: The degree of the polynomial fit (default is 3).\n" + \
                "sample_count: The number of samples for the fitted data. If not provided, the function returns the coefficients.\n" + \
                "To get help, pass 'help' as the first argument."

    # Check if the user asked for help
    if lattice_list == "help":
        print(help_info)
        return

    # Reshape the source data
    lattice = np.array(lattice_list)
    distance = np.array(distance_list)
    energy = np.array(energy_list)

    # Create polynomial features
    lattice_poly = np.vander(lattice, degree + 1)
    distance_poly = np.vander(distance, degree + 1)
    poly_matrix = np.hstack([lattice_poly, distance_poly[:, 1:]])

    # Fit using the least squares method
    coef, _, _, _ = np.linalg.lstsq(poly_matrix, energy, rcond=None)

    # Generate new data using the polynomial function (if needed)
    if sample_count:
        lattice_samples = np.linspace(lattice.min(), lattice.max(), sample_count)
        distance_samples = np.linspace(distance.min(), distance.max(), sample_count)
        lattice_poly_samples = np.vander(lattice_samples, degree + 1)
        distance_poly_samples = np.vander(distance_samples, degree + 1)
        poly_matrix_samples = np.hstack([lattice_poly_samples, distance_poly_samples[:, 1:]])
        energy_samples = poly_matrix_samples.dot(coef)
        return lattice_samples, distance_samples, energy_samples
    else:
        return coef

def rec_to_cart(klist_source, directory, crystal_type):
    # Define the reciprocal lattice vectors for different crystal types
    # These are the transformation matrices for converting reciprocal lattice
    # points to Cartesian coordinates.
    if crystal_type.lower() == "hcc":
        transform_matrix = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 2
    elif crystal_type.lower() == "bcc":
        transform_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2
    elif crystal_type.lower() == "hcp":
        # For HCP, the transformation depends on the ratio of the lattice constants a and c.
        # We need to read these from a file or define them here.
        # For example, let's assume we have them in a file named 'lattice_constants.txt'
        with open(os.path.join(directory, "lattice_constants.txt"), "r", encoding="utf-8") as file:
            a, c = map(float, file.readline().split())
        transform_matrix = np.array([[1, -1/np.sqrt(3), 0], [0, 2/np.sqrt(3), 0], [0, 0, a/c]])
    elif crystal_type.lower() == "sc":
        transform_matrix = np.identity(3)
    elif crystal_type.lower() == "graphene":
        a = 2.46  # Graphene's lattice constant in angstroms
        b1 = (2 * np.pi / a) * np.array([1 / np.sqrt(3), 1, 0])
        b2 = (2 * np.pi / a) * np.array([1 / np.sqrt(3), -1, 0])
        transform_matrix = np.array([b1, b2, [0, 0, 1]])
    else:
        raise ValueError(f"Unknown crystal type: {crystal_type}")
    # Convert the Klist from reciprocal to Cartesian coordinates
    cartesian_kpoints = np.dot(klist_source, transform_matrix.T)
    return cartesian_kpoints

def cart_to_rec(klist_source, directory, crystal_type):
    # Define the transformation matrices for converting Cartesian coordinates
    # to reciprocal lattice points for different crystal types.
    if crystal_type.lower() == "hcc":
        transform_matrix = np.linalg.inv(np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 2)
    elif crystal_type.lower() == "bcc":
        transform_matrix = np.linalg.inv(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2)
    elif crystal_type.lower() == "hcp":
        # For HCP, the transformation depends on the ratio of the lattice constants a and c.
        # We need to read these from a file or define them here.
        # For example, let's assume we have them in a file named 'lattice_constants.txt'
        with open(os.path.join(directory, "lattice_constants.txt"), "r", encoding="utf-8") as file:
            a, c = map(float, file.readline().split())
        transform_matrix = np.linalg.inv(np.array([[1, -1/np.sqrt(3), 0], [0, 2/np.sqrt(3), 0], [0, 0, a/c]]))
    elif crystal_type.lower() == "sc":
        transform_matrix = np.linalg.inv(np.identity(3))
    elif crystal_type.lower() == "graphene":
        a = 2.46  # Graphene's lattice constant in angstroms
        b1 = (2 * np.pi / a) * np.array([1 / np.sqrt(3), 1, 0])
        b2 = (2 * np.pi / a) * np.array([1 / np.sqrt(3), -1, 0])
        transform_matrix = np.linalg.inv(np.array([b1, b2, [0, 0, 1]]))
    else:
        raise ValueError(f"Unknown crystal type: {crystal_type}")
    # Convert the Kpoints from Cartesian to reciprocal coordinates
    reciprocal_kpoints = np.dot(klist_source, transform_matrix.T)
    return reciprocal_kpoints
