# Development note

## Code specifications

* Code comments should be written in English;

* Please use the snake_case naming convention, and try to use full names for functions or variable names;

* Keep the font: default Matplotlib serif font: DejaVu Serif.

### algorithms.py: General algorithms

* compute_average(data) - complete

* birch_murnaghan_eos(...) - complete

* fit_eos(...) - complete

* polynomially_fit_curve(...) - complete

* polynomially_fit_surface(...) - complete

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

* plot_lattice_free_energy(matter, count, ...) - complete
  * plot_lattice_free_energy_solo(...)
  * plot_lattice_free_energy_duo(...)
  * plot_lattice_free_energy_tri(...)
  * plot_lattice_free_energy_qua(...)

### lattice_bilayer.py: Data extracting with the distance of bilayer

* specify_bilayer_lattice(directory) - complete

* summarize_bilayer_lattice(sup_directory) - complete

* read_bilayer_lattice_data(data_path) - complete

* extract_extreme_bilayer_lattice(data_path) - complete
  * extract_minimum_bilayer_lattice(data_path)
  * extract_maximum_bilayer_lattice(data_path)

* plot_bilayer_lattice(matter, ...) - complete

### kpoints.py: What is the impact of changing the k-points?

* Free energy versus K points
  * identify_kpoints(directory) - complete
  * specify_kpoints_free_energy(directory) - complete
  * summarize_kpoints_free_energy(sup_directory) - complete
  
* plot_kpoints_free_energy

## Bandstructure

### bandstructure.py: Bandstructure analysis and plotting

* Bandstructure information
  * extract_high_symlines(directory) - complete
  * extract_fermi(directory) - complete
  * extract_klist(directory) - complete
  * rec_to_cart(...) - complete
  * cart_to_rec(...) - complete
  * clean_kpoints(kpoints) - complete

## DoS/PDoS

### DoS.py

* DoS extracting
  * extract_dos(directory) - complete
  * create_matters(directory_list) - complete

* DoS Plotting
  * plot_dos_sol(...) - complete
  * plot_dos_data(...) - complete
  * plot_dos(...) - complete

### PDoS.py

* PDoS analysis
  * analyze_dpos(directory) - complete
  * get_elements(directory) - complete

* PDoS extracting
  * extract_pdos(directory) - complete
  * extract_element_pdos(directory, element) - complete
  * extract_segment_pdos(directory, start, end) - complete

* PDoS plotting
  * plot_total_pdos(...) - complete
  * plot_pdos_segment(...) - complete
    * plot_total_pdos_data(...)
    * plot_sol_segment(...)
    * plot_duo_segment(...)
    * plot_tri_segment(...)
