# Development note

## Code specifications

* Code comments should be written in English;

* Please use the snake_case naming convention, and try to use full names for functions or variable names;

* Keep the font: default matplotlib serif font: DejaVu Serif.

## General Functions

### output.py: Output setting

* vasprun_directory(directory) - complete

* canvas_setting(*args) - complete

* color_sampling(colors) - complete

### algorithms.py: Algorithms

* compute_average(data_lines) - complete

* polynomially_fit_curve(lattice, energy, degree, count) - complete

* polynomially_fit_surface(lattice, distance, energy, degree, count) - complete

## Lattice

### lattice.py: data extracting

* check_vasprun(sup_directory) - complete

* lattice_free_energy_input(directory) - complete

* specify_lattice_free_energy(directory) - complete

* summarize_lattice_free_energy(sup_directory) - complete

* read_lattice_free_energy_data(data_path) - complete

* read_lattice_free_energy_count(data_path, count) - complete

* extract_fitted_extreme_lattice_free_energy(data_path) - complete
  * extract_fitted_minimum_lattice_free_energy(data_path)
  * extract_fitted_maximum_lattice_free_energy(data_path)

### lattice_biolayer.py: Data extracting with the distance of biolayer

* specify_biolayer_lattice(directory) - complete

* summarize_biolayer_lattice(sup_directory) - complete

* read_biolayer_lattice_data(data_path) - complete

* extract_extreme_biolayer_lattice(data_path) - complete
  * extract_minimum_biolayer_lattice(data_path)
  * extract_maximum_biolayer_lattice(data_path)

### lattice_plotting.py: Plotting

* plot_lattice_free_energy(matter, count, ...) - complete
  * plot_lattice_free_energy_solo(...)
  * plot_lattice_free_energy_duo(...)
  * plot_lattice_free_energy_tri(...)
  * plot_lattice_free_energy_qua(...)

* plot_biolayer_lattice(matter, ...) - complete

### kpoints.py: What is the impact of changing the k-points?

* Free energy versus kpoints
  * identify_kpoints(directory) - complete
  * specify_kpoints_free_energy(directory) - complete
  * summarize_kpoints_free_energy(sup_directory) - complete
  
* DoS versus kpoints

### kpoints_plotting.py: What is the impact of changing the k-points?

* plot_kpoints_free_energy

* plot_kpoints_DoS

### bandstructure.py: Bandstructure analysis and plotting

* def EIGENVAL_BS_extracting(*args)
  * def EIGENVAL_BS_single_extracting(directory)
  * def EIGENVAL_BS_duo_extracting(directory)

* def EIGENVAL_BS_plotting(directory)
  * def EIGENVAL_BS_single_plotting(directory)
  * def EIGENVAL_BS_duo_plotting(directory)

* def EIGENVAL_BSDOS_plotting(directory)
  * def EIGENVAL_BSDOS_single_plotting(directory)
  * def EIGENVAL_BSDOS_duo_plotting(directory)
