# Thesis figures

Open `figures_for_thesis.ipynb` in Jupyter or VS Code. Select the Python
`py314` kernel, run the first two setup cells, and then run the required figure
cells. **Run All** regenerates all 36 current figures. The notebook works with
either this folder or the repository root as the working directory.

Edit the first cell for shared fonts, colours and legend defaults. Each figure
group contains its own editable Matplotlib layout, canvas height, limits and
panel arrangement. The notebook imports only numerical readers; it does not
run the plotting scripts. Running a figure cell displays the figure, saves its
PDF here, and copies it to `../PhD_thesis_20251216/figures_proj1`.

The saved notebook has cleared image outputs to keep it small; running its
cells displays the figures again. `notebook_verification.json` records the
completed execution. No VASP calculation or input is changed.

The `.py` plotting scripts remain the verified batch baseline. **They do not
inherit subsequent notebook edits**: running them can overwrite a figure
customized in the notebook. Use the notebook as the normal editing entry.

The shared canvas width is 10 inches: labels 16 pt, ticks and legends 14 pt,
inset titles 12 pt and line width 1.5 pt. Panels use ordinary Matplotlib axes,
legends and rounded title boxes. There is no dependency on
`vmatplot.output_settings`; only the existing numerical EOS fitter is reused.

## Combined figures

- `proj1.3_bands.pdf`: four monolayer band structures, one legend.
- `proj1.8_schottky.pdf`: Schottky sketch and the HSE06 band example.
- `proj1.12_optics.pdf`: absorption and energy loss, one material legend.
- `proj1.13_optics.pdf`: reflectivity and refractive index, one material legend.
- `S1.13_dielectric.pdf`: four monolayers, two directions, one line-style legend.
- `S1.14_optics.pdf`: five optical properties, three diagonal directions,
  one material legend.

Existing multi-panel numerical figures retain their filenames. `S1.4.pdf`
through `S1.6.pdf` contain complete panel groups; no TeX cropping is required.
The old `1.3a.pdf`–`1.3d.pdf` and `1.5_alt.pdf`–`1.7_alt.pdf` are legacy exports
already present in this folder; the current thesis uses the `proj1.*` exports.

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
`verification.json` records numerical comparisons with the previously audited
figures, raw-file Git blob identities, PDF dimensions/fonts, and identical
thesis copies. `spacing_audit.json` compares the old diagonal scan, the bounded
minimum of the same polynomial, and actual sampled minima. The polynomial
minimum is a diagnostic only; its boundary solutions are not relaxed structures.
