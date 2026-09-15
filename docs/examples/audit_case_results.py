"""Audit completed local case results and record numerical invariants."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
CASES = ROOT/"docs/_build/cases"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def csv(path):
    a=np.loadtxt(path,delimiter=",")
    assert np.isfinite(a).all(),path
    return a


def identity(directory, suffix, oop=False):
    tag = "oop_static" if oop else "static"
    number = "2" if oop else ""
    cfs=csv(directory/f"cfs_{tag}_{suffix}.csv")
    normal=csv(directory/f"normal_stress{number}_{tag}_{suffix}.csv")
    shear=csv(directory/f"shear_stress{number}_{tag}_{suffix}.csv")
    tensor=np.load(directory/f"stress_tensor_{suffix}.npy")
    assert np.isfinite(tensor).all()
    mean=tensor[:,[0,3,5]].astype(float).mean(axis=1)
    expected=shear+(.6 if oop else .4)*(normal-(.75 if oop else 0)*mean)
    np.testing.assert_allclose(cfs,expected,rtol=2e-12,atol=1e-7)
    return {"points":len(cfs),"tensor_shape":list(tensor.shape),
            "coulomb_identity_max_abs_residual_Pa":float(np.max(np.abs(cfs-expected)))}


def half_space_far_field(library):
    """Unit-potency static stress must decay like mu/r^3; 10 is a loose bound."""
    info=json.loads((library/"green_lib_info.json").read_text())
    assert info["layered"] is False
    sd0,sd1=info["grn_source_depth_range"]
    d0,d1=info["grn_dist_range"]
    n_sd=int(round((sd1-sd0)/info["grn_source_delta_depth"]))+1
    dist=np.arange(int(round((d1-d0)/info["grn_delta_dist"]))+1)*info["grn_delta_dist"]+d0
    stress=np.fromfile(library/"edcmp2_stress.bin",dtype=np.float32).reshape(
        n_sd,len(info["obs_depth_list"]),5,len(dist),6)
    far=dist>=10
    scaled=np.abs(stress[...,far,:]).max(axis=-1)*(dist[far]*1e3)**3/info["mu"]
    assert scaled.max()<10,scaled.max()
    return {"max_abs_stress_times_r3_over_mu_r_ge_10km":float(scaled.max()),"limit":10}


def dynamic_sample(case):
    c=CASES/case
    rows=(np.load(c/"grn_d/results_each/obs_plane_5.00.npy") if case=="ludian"
          else csv(c/"input/obs_plane5.csv"))
    folder=c/"grn_d/results_each"
    outputs=[]
    for i in [0,len(rows)//2,len(rows)-1]:
        point=rows[i]
        name="_".join(f"{x:.4f}" for x in point[:3])
        tensor=np.load(folder/(name+"_stress_ned.npy"))
        kind="_oop" if case=="ludian" else ""
        num="2" if case=="ludian" else ""
        normal=np.load(folder/(name+kind+f"_normal_stress{num}.npy"))
        shear=np.load(folder/(name+kind+f"_shear_stress{num}.npy"))
        cfs=np.load(folder/(name+kind+"_cfs.npy"))
        mean=tensor[:,[0,3,5]].mean(axis=1)
        expected=shear+(.6 if case=="ludian" else .4)*(normal-(.75 if case=="ludian" else 0)*mean)
        np.testing.assert_allclose(cfs,expected,rtol=2e-12,atol=1e-7)
        assert np.isfinite(tensor).all()
        outputs.append({"receiver_index":i,"samples":len(cfs),
                        "coulomb_identity_max_abs_residual_Pa":float(np.max(np.abs(cfs-expected)))})
    return outputs


def compare_duplicate_native_jobs():
    case=CASES/"wenchuan"
    ledger=case/"library-acceleration.json"
    if not ledger.exists():
        return None
    entries=json.loads(ledger.read_text())["jobs"]
    installed={e["index"] for e in entries if e["status"]=="installed_before_primary_start"}
    for name in ("library-acceleration-promotions.json",
                 "library-acceleration-paused-promotions.json"):
        promotions=case/name
        if promotions.exists():
            installed.update(e["index"] for e in json.loads(promotions.read_text())["jobs"]
                             if e["status"]=="installed_before_primary_start")
    checked=[]
    for entry in entries:
        if entry["status"]!="kept_isolated_primary_may_have_started" or entry["index"] in installed:
            continue
        sd,rd,group=entry["job"]
        relative=Path(f"{sd:.2f}/{rd:.2f}/{group}_0")
        main=case/"grn_d/qseis"/relative
        other=ROOT/"docs/_build/case-reruns/extra-qseis"/relative
        components=("szz","szr","srr","stt","szt","srt")
        for component in components:
            name=f"grn_{component}.bin"
            assert sha(main/name)==sha(other/name),(relative,component)
        checked.append(entry["index"])
    return {"meaning":"Independent current-code duplicate runs; not the old baseline",
            "native_jobs":len(checked),"bitwise_equal_binary_files":len(checked)*6,"job_indices":checked}


def main():
    report={}
    for case in ("ludian","wenchuan"):
        c=CASES/case
        run=json.loads((c/"run.json").read_text(encoding="utf-8"))
        assert all(run["stages"].get(s,{}).get("status")=="complete" for s in ("static","dynamic-library","dynamic"))
        for rel,h in {**run["input_sha256"],**run["source_sha256"]}.items():
            assert sha(ROOT/rel)==h,rel
        results=c/"results"
        if case=="ludian":
            static={"map":identity(results/"static","dep_5.00",True)}
            dynamic=csv(results/"dynamic/cfs_oop_dynamic_dep_5.00.csv")
            assert dynamic.shape==(1681,1024)
        else:
            static={"plane":identity(results/"static","plane5")}
            for label in ("static_layer","static_half"):
                static[label]=identity(results/label,"dep_15.00")
                assert static[label]["points"]==361201
            static["half_space_library_far_field"]=half_space_far_field(c/"grn_static_half")
            for nt in range(0,80,2):
                identity(results/f"static/time/{nt}","plane5")
            dynamic=csv(results/"dynamic/cfs_dynamic_plane5.csv")
            assert dynamic.shape==(102,1024)
        report[case]={"scope":"Algebraic/file checks; does not establish scientific validity or convergence",
                      "input_and_core_source_hashes":"unchanged","all_three_stages":"complete",
                      "static":static,"dynamic_shape":list(dynamic.shape),"dynamic_finite":True,
                      "dynamic_sample_checks":dynamic_sample(case)}
    report["independent_current_run_comparison"]=compare_duplicate_native_jobs()
    target=ROOT/"docs/_build/case-reruns/numerical-audit.json"
    target.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
