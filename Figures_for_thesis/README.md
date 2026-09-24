# Thesis figures

Open `figures_for_thesis.ipynb` in Jupyter or VS Code. Select the Python
`py314` environment, run the first two setup cells, and then run the required
figure cells. **Run All** regenerates all 48 current figures. The notebook
works with either this folder or the repository root as the working directory.

Edit the first setup cell for shared fonts, colours and legend defaults. The
second setup cell contains the numerical readers and reusable plotting helpers.
Each figure group contains editable Matplotlib code; change its `figsize`,
limits or layout directly. The optical groups share the editable `optical()`
helper in the second setup cell. Running a figure cell saves its PDF here,
copies it to `../PhD_thesis_20251216/figures_proj1`, and displays the figure.
The edited heat-map cell's old saved output is cleared; outputs in other cells
remain available for comparison.

The `.py` plotting scripts remain the batch baseline. They do not inherit
subsequent notebook edits, so running them can overwrite a figure customized
in the notebook. Use the notebook as the normal editing entry. No VASP
calculation or input is changed.

## Original figure style

The figures use the original author settings: serif text, Computer Modern
math, 16 pt axis labels, 14 pt ticks, 12 pt legends, 20 pt titles and 18 pt
subtitles where specified in the original plots. Line width is 1.5 pt.
The settings are written explicitly, without `vmatplot.output_settings`.
The existing numerical EOS fitter remains in use.

Original canvases are restored: 10 x 6 inches for single plots, 12 x 6 for
band/DoS figures, 16 x 6 for paired plots, 16 x 12 for four panels, 24 x 12
for six panels and 24 x 18 for nine panels. Structure figures use their own
20 x 12 canvases and original larger annotation sizes.

The retained layout exceptions are:

- The three-structure figure `proj1.3`, `S1.3`, `S1.9`, the five `S1.14`
  optical figures, and `S1.20` through `S1.24` use a 2 x 2 grid with three
  data panels and the shared legend in the fourth cell. Scalar optical
  quantities retain only the three scientifically valid diagonal components.
- `proj1.8_schottky` retains the current two panel titles.
- `S1.4` through `S1.6` retain the heat maps and shared colour scales on
  13 x 10 canvases with tighter panel spacing. `S1.4` places a vertical colour
  bar and the minimum legend in the fourth cell; `S1.5` and `S1.6` retain four
  data panels and a colour bar at the right. The source energies, linear
  interpolation, sampled minima and axis ranges are unchanged.
  `energy_colour_scales.json` records the limits.

## Current exports

The original separate figures are restored:

- `proj1.3a.pdf` through `proj1.3d.pdf`: four monolayer band structures.
- `proj1.12a.pdf`, `proj1.12b.pdf`: absorption and energy loss.
- `proj1.13a_cor.pdf`, `proj1.13b.pdf`: reflectivity and refractive index.
- `S1.13a.pdf` through `S1.13d.pdf`: monolayer dielectric functions.
- `S1.14a.pdf`, `S1.14b.pdf`, `S1.14c.pdf`, `S1.14d_correct.pdf`, and
  `S1.14e_correct.pdf`: five monolayer optical properties.

The superseded combined exports are removed. Existing multi-panel figures
retain their filenames, including `proj1.8_schottky.pdf`. `S1.4.pdf` through
`S1.6.pdf` contain complete panel groups; no TeX cropping is required.
The old `1.3a.pdf` through `1.3d.pdf` and `1.5_alt.pdf` through `1.7_alt.pdf`
are legacy exports already present in this folder; the thesis uses the
`proj1.*` exports.

`original_style_verification.json` is the current export record: 48 PDF
canvases, font sizes, text bounds, hashes and identical thesis copies,
plus the unchanged scientific functions and completed notebook run.
The final notebook-only helper change adds inline display after saving;
it does not change PDF plotting or numerical calculations.
`notebook_verification.json`, `notebook_render_verification.json`, and
`moderate_layout_verification.json` are historical records of the superseded
36-figure layout. Their old PDF hashes and layout descriptions are retained
as history and do not describe the current exports.

## Data corrections made during source verification

The HSE06 atom/orbital-projected DoS figures (`S1.9`–`S1.11`) consistently use
the ordinary `17 x 17 x 1` grid, its energy array, total and projected DoS,
and its Fermi energy. The old plotting program mixed the ordinary projected
array with the `33 x 33 x 1` optional-grid total DoS and Fermi energy; it also
paired the optional-grid total DoS with the ordinary-grid energy axis. The
optional-grid projected arrays in the archived HDF5 files have identically
zero entries for the later atoms and cannot supply a complete alternative.
This is incomplete archived output, not evidence of zero physical projection.
The complete optional-grid **total** DoS remains used in the main band/DoS
figures. Both grids use the original Gaussian width of 0.1 eV.

The Hollow 2 panels in `S1.5` and `S1.6` now read their own numerical tables.
The old notebook accidentally used the Hollow 1 table twice. The surfaces
retain the original linear interpolation. Their markers identify the lowest
sampled energies; the old simultaneous one-dimensional scan in lattice
constant and separation was not a two-dimensional minimization. The separate
one-dimensional lattice plots retain their fitted minima.

Dielectric tensors are used exactly as stored, including unequal transposed
entries. Scalar optical quantities use only diagonal components. Absorption
uses `alpha = 2 E kappa / (hbar c)` in inverse nanometres.

`*_sources.json` records each selected raw group and energy reference.
`verification.json` retains the earlier numerical comparisons and raw-file Git
blob identities. Its PDF presentation records describe the earlier layout;
use `original_style_verification.json` for the current exports.
`spacing_audit.json` compares the old diagonal scan, the bounded minimum of
the same polynomial, and actual sampled minima. The polynomial minimum is a
diagnostic only; its boundary solutions are not relaxed structures.
