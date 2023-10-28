#### General Algoriths
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import numpy as np

def compute_average(data_lines):
    """Define the function to compute the average of the last value in each line."""
    total = 0
    for line in data_lines:
        values = line.split()       # Split the line into individual values
        total += float(values[-1])  # Add the last value to the total
        print(line)
    return total / len(data_lines)  # Return the average

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
