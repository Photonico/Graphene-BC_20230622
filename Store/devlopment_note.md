# Development note

## Code specifications

* Code comments should be written in English;

* Please use the snake_case naming convention, and try to use full names for functions or variable names;

* Keep the font: default matplotlib serif font: DejaVu Serif.

## Functions

### General functions

* output(length, width, dpi, font_style)

### lattice constant - free energy

* Data extracting, reading, and data prcessing
  * lattice_free_energy_input(directory)
  * specify_lattice_free_energy(directory)
  * check_lattice_free_energy(sup_directory)
  * summarize_lattice_free_energy(sup_directory)
  * read_lattice_free_energy_data(data_path)
  * read_lattice_free_energy_count(data_path, count)
  * polynomially_fit_curve(lattice, free_energy, method, count)

* Plotting
  * plot_lattice_free_energy_single(matter, source_data, selected_data, count, color)

* Plotting





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
