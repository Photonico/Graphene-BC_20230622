"""Check raw-source identity, repaired grids, and exported PDF bounds."""
import json
import hashlib
import subprocess
import numpy as np
import h5py
import fitz
from electronic import bands, dos, BAND_COLORS
from optics import dielectric
from style import ROOT, HERE, THESIS

legacy = ["readability_plot_vectors.npz", "electronic_plot_vectors.npz", "optics_data.npz"]
if all((THESIS / name).exists() for name in legacy):
    audit = {"band_curve_checks": [], "dielectric_identity": [], "pdos_repairs": []}
    for archive,mapping in [
        ("readability_plot_vectors.npz", {"proj1.3a":"D_Graphene_", "proj1.3b":"B_Borophene_",
            "proj1.3c":"A_BC3_", "proj1.3d":"C_B4C3_"}),
        ("electronic_plot_vectors.npz", {"proj1.5":"E_Graphene-BC3_hollow_",
            "proj1.6":"F_Graphene-Borophene_top_", "proj1.7":"G_Graphene-B4C3_top_"}),
    ]:
        old=np.load(THESIS/archive);meta=json.loads(str(old["metadata_json"]))
        for name,prefix in mapping.items():
            raw=[bands(prefix+f) for f in ["PBE","HSE"]];errors=[]
            for record in meta[name]["curves"]:
                if record["panel"] != 0: continue
                xy=old[record["key"]];xy=xy[np.isfinite(xy).all(axis=1)]
                xy=xy[abs(xy[:,1])<=6]  # Compare the visible energy window.
                if not len(xy): continue
                errors.append(min(float(np.max(abs(np.interp(xy[:,0],x,y)-xy[:,1])))
                                  for x,energy,*_ in raw for y in energy))
            audit["band_curve_checks"].append({"old_figure":name,"curves":len(errors),
                "maximum_energy_difference_eV":max(errors),"note":"Visible -6 to 6 eV range; H5 precision versus rounded XML/PDF vertices"})
            assert max(errors)<.0002

    old=np.load(THESIS/"optics_data.npz")
    for source in json.loads((THESIS/"optics_sources.json").read_text())["sources"]:
        name=source["system"];energy,tensor=dielectric(name)
        de=float(np.max(abs(energy-old[name+"_energy"])))
        dt=float(np.max(abs(tensor-old[name+"_epsilon"])))
        audit["dielectric_identity"].append({"source":name,"energy_difference":de,"tensor_difference":dt})
        assert de==0 and dt==0

    old=np.load(THESIS/"electronic_plot_vectors.npz");meta=json.loads(str(old["metadata_json"]))
    for name,folder,split in [("S1.9","F_Graphene-Borophene_HSE",8),
        ("S1.10","E_Graphene-BC3_HSE",8),("S1.11","G_Graphene-B4C3_HSE",7)]:
        path=ROOT/"4_PDoS"/folder
        with h5py.File(path/"vaspout.h5") as f:
            ordinary=f["results/electron_dos"];opt=f["results/electron_dos_kpoints_opt"]
            ef=float(ordinary["efermi"][()]);efopt=float(opt["efermi"][()])
            wrong_x=ordinary["energies"][:]-efopt
            original_total=opt["dos"][0]
            record=next(r for r in meta[name]["curves"] if r["panel"]==0 and r["color"]==0)
            xy=old[record["key"]];xy=xy[np.isfinite(xy).all(axis=1)]
            old_error=float(np.max(abs(np.interp(xy[:,0],wrong_x,original_total)-xy[:,1])))
            incomplete=np.where(np.max(abs(opt["dospar"][0]),axis=(1,2))==0)[0]+1
            energy,total,partial,_=dos(folder,projected_grid=True)
            substrate=partial[:split].sum(axis=(0,1));graphene=partial[split:].sum(axis=(0,1))
            window=(energy>=-.8)&(energy<=1.2)
            near=(energy>=-.2)&(energy<=.2)
            audit["pdos_repairs"].append({"figure":name,"source":str(path.relative_to(ROOT)),
                "old_mixed_grid_reproduction_max_error":old_error,
                "ordinary_energy_range_eV":ordinary["energies"][[0,-1]].tolist(),
                "opt_energy_range_eV":opt["energies"][[0,-1]].tolist(),
                "ordinary_fermi_eV":ef,"opt_fermi_eV":efopt,
                "opt_zero_projected_atoms_one_based":incomplete.tolist(),
                "chosen_consistent_group":"results/electron_dos",
                "chosen_kmesh":[17,17,1],"ISMEAR":0,"SIGMA_eV":.1,
                "substrate_and_graphene_PDoS_at_EF":[float(np.interp(0,energy,substrate)),float(np.interp(0,energy,graphene))],
                "mean_substrate_and_graphene_minus08_to12":[float(substrate[window].mean()),float(graphene[window].mean())],
                "mean_substrate_and_graphene_minus02_to02":[float(substrate[near].mean()),float(graphene[near].mean())]})
            assert old_error<.005
            assert np.all(np.max(abs(partial),axis=(1,2))>0)

else:
    audit = json.loads((HERE / "verification.json").read_text())
    audit["legacy_comparison_note"] = "Recorded before the superseded thesis data caches were removed; current source and PDF checks are rerun."

audit["pdf_exports"]=[]
for path in sorted(HERE.glob("*.pdf")):
    if not (THESIS/path.name).exists():continue
    doc=fitz.open(path);page=doc[0];outside=[];sizes=set()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines",[]):
            for span in line["spans"]:
                sizes.add(round(span["size"],2));x0,y0,x1,y1=span["bbox"]
                if min(x0,y0)<-.2 or x1>page.rect.width+.2 or y1>page.rect.height+.2:
                    outside.append({"text":span["text"],"bbox":span["bbox"]})
    source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    thesis_hash=hashlib.sha256((THESIS/path.name).read_bytes()).hexdigest()
    audit["pdf_exports"].append({"name":path.name,"width_pt":page.rect.width,"height_pt":page.rect.height,
        "font_sizes_pt":sorted(sizes),"out_of_bounds_text":outside,"sha256":source_hash,"thesis_copy_identical":source_hash==thesis_hash})

index={}
for line in subprocess.check_output(["git","ls-files","-s"],cwd=ROOT,text=True).splitlines():
    information,path=line.split("\t");index[path]=information.split()[1]
sources={}
for manifest in ["electronic_sources.json","optical_sources.json","energy_sources.json"]:
    for key,record in json.loads((HERE/manifest).read_text()).items():
        path=record.get("file",key);record["git_blob_sha1"]=index.get(path)
        sources.setdefault(path, {"git_blob_sha1": index.get(path), "selections": []})["selections"].append(record)
audit["sources"]=sources
audit["energy_convergence_source"]="1_Kpoints/Graphene_BC3_Hollow/energy_kpoint.dat"
audit["schottky_reference_levels_eV"]={"conduction":1.42,"valence":-.87,
    "source_notebook":"3.3_schottky_barrier_determination.ipynb"}
(HERE/"verification.json").write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps({"band_curves":sum(r["curves"] for r in audit["band_curve_checks"]),
    "identical_dielectric_sources":len(audit["dielectric_identity"]),"exported_pdfs":len(audit["pdf_exports"]),
    "out_of_bounds":[p["name"] for p in audit["pdf_exports"] if p["out_of_bounds_text"]]},indent=2))
