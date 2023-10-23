# Development note

## Code specifications

* Code comments should be written in English;

* Please use the snake_case naming convention, and try to use full names for functions or variable names;

* Keep the font: default matplotlib serif font: DejaVu Serif.

## Functions

### General functions

* output(length, width, dpi, font_style)

### lattice constant - free energy

* Data extracting, reading
  * check_vasprun(sup_directory)
  * lattice_free_energy_input(directory)
  * specify_lattice_free_energy(directory)
  * specify_lattice_distance(directory)
  * summarize_lattice_free_energy(sup_directory)
  * summarize_lattice_distance(sup_directory)
  * read_lattice_free_energy_data(data_path)
  * read_lattice_free_energy_count(data_path, count)
  * read_lattice_distance(data_path)

* Data processing
  * polynomially_fit_curve(lattice, free_energy, method, count)
  * extract_fitted_extreme_lattice_free_energy(data_path)
    * extract_fitted_minimum_lattice_free_energy(data_path)
    * extract_fitted_maximum_lattice_free_energy(data_path)

* Plotting
  * plot_lattice_free_energy(matter_count, ...)
    * plot_lattice_free_energy_solo(...)
    * plot_lattice_free_energy_duo(...)
    * plot_lattice_free_energy_tri(...)
    * plot_lattice_free_energy_qua(...)





### lattice - distance - free energy

* lattice_distance_free_energy_input(directory)
* lattice_distance_free_energy_specific(directory)

### Bandstructure

* def EIGENVAL_BS_extracting(*args)
  * def EIGENVAL_BS_single_extracting(directory)
  * def EIGENVAL_BS_duo_extracting(directory)

* def EIGENVAL_BS_plotting(directory)
  * def EIGENVAL_BS_single_plotting(directory)
  * def EIGENVAL_BS_duo_plotting(directory)

* def EIGENVAL_BSDOS_plotting(directory)
  * def EIGENVAL_BSDOS_single_plotting(directory)
  * def EIGENVAL_BSDOS_duo_plotting(directory)
