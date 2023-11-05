#### Bandstructure with Pymatgen

#%% Plot 2D matter bandstructure
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.plotter import BSPlotter, BSPlotterProjected, BSDOSPlotter, DosPlotter
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.inputs import Kpoints

vasprun_PBE = Vasprun("3_Bandstructure_PBE/D_Graphene/vasprun.xml", parse_potcar_file = False, parse_projected_eigen = True)
kpoints_file = "3_Bandstructure_PBE/A_BC3/KPOINTS"
kpoints = Kpoints.from_file(kpoints_file)

# Extract the Fermi energy
fermi_energy_PBE = vasprun_PBE.efermi

# Extract the bandstructre
bandstructure_PBE = vasprun_PBE.get_band_structure(kpoints_filename = kpoints_file, line_mode=True)

# Extract VBM (Valence band maximum) and CBM (Conduction band minimum)
vbm_info_PBE = bandstructure_PBE.get_vbm()
cbm_info_PBE = bandstructure_PBE.get_cbm()
vbm_energy_PBE = vbm_info_PBE["energy"]
cbm_energy_PBE = cbm_info_PBE["energy"]

# Plot
params = {"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
          "axes.titlesize": 18, "axes.labelsize": 14, "figure.facecolor": "w"}
plt.rcParams.update(params)

plotter = BSPlotter(bandstructure_PBE)
plot_data = plotter.bs_plot_data()
plotter.get_plot(ylim=(-6, 6))

plt.axhline(y = cbm_energy_PBE - fermi_energy_PBE, color="#B95FF5", alpha = 0.4, label = "CBM (PBE)")
plt.axhline(y = vbm_energy_PBE - fermi_energy_PBE, color="#37B44B", alpha = 0.4, label = "VBM (PBE)")
plt.axhline(y = 0, color="#EB731E", alpha = 0.4, label=f"Fermi energy\n{fermi_energy_PBE:.3f} (eV)")

ax = plt.gca()
ax.figure.set_dpi(196)
ax.figure.set_size_inches(10, 6)

ax.set_title("Bandstructure for BC₃",fontsize =1.0*params["axes.titlesize"])
ax.set_xlabel(r"Wave Vector", fontsize =1.0*params["axes.labelsize"])
ax.set_ylabel(r"Energy (eV)", fontsize =1.0*params["axes.labelsize"])
ax.tick_params(axis="both", labelsize=params["axes.labelsize"],direction="in")
# plt.axhline(y = fermi_energy, linestyle="--", color="#EB731E", label="Fermi Energy")

plt.legend().remove()
# plt.tight_layout()
plt.legend(loc="upper right")
plt.show()

# %%
