"""Re-run the bundled earthquake cases with isolated outputs and stage records.

Run from the repository root in an activated numerical environment:
    python docs/examples/run_case_studies.py --case ludian --stage static
    python docs/examples/run_case_studies.py --case ludian --stage dynamic-library
    python docs/examples/run_case_studies.py --case ludian --stage dynamic
"""
import argparse
import configparser
import contextlib
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pygrnwang
from dyncfs import __version__
from dyncfs.configuration import CfsConfig
from dyncfs.cfs_static import create_static_lib, compute_static_cfs, compute_static_cfs_fix_depth
from dyncfs.cfs_dynamic import compute_dynamic_cfs_parallel, compute_dynamic_cfs_fix_depth_parallel
from dyncfs.utils import cal_grid_num, pairwise_spherical_dist_azimuth_km, read_source_array
from pygrnwang.create_qseis2025_bulk import pre_process_qseis2025, create_grnlib_qseis2025_parallel
from pygrnwang.read_qseis2025 import get_sorted_grid_params


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(case, output, processes, map_spacing):
    template = ROOT / "examples" / case
    original_ini = template / f"{case}.ini"
    expected = [original_ini, template / "input/model.nd"]
    source_ids = range(1, 6) if case == "wenchuan" else range(1, 3)
    obs_ids = [5] if case == "wenchuan" else [1, 2]
    expected += [template / f"input/source_plane{i}.csv" for i in source_ids]
    expected += [template / f"input/obs_plane{i}.csv" for i in obs_ids]
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in expected}
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in [
        ROOT / "dyncfs/configuration.py", ROOT / "dyncfs/cfs_static.py",
        ROOT / "dyncfs/cfs_dynamic.py", ROOT / "dyncfs/utils.py",
    ]}
    manifest_path = output / "run.json"
    if output.exists() and any(output.iterdir()) and not manifest_path.exists():
        raise ValueError("Output directory must be empty or contain this runner's run.json")
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
        if report["case"] != case or report["input_sha256"] != input_hashes or report["source_sha256"] != source_hashes:
            raise ValueError("Inputs/source changed; use a fresh output directory")
        if report["map_spacing_deg"] != map_spacing:
            raise ValueError("Map spacing changed; use a fresh output directory")
    else:
        report = {
            "case": case, "input_sha256": input_hashes, "source_sha256": source_hashes,
            "dyncfs": __version__, "pygrnwang": pygrnwang.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "map_spacing_deg": map_spacing, "stages": {},
            "baseline": "Repository PDFs; original numerical baseline not yet located",
        }
        save_json(manifest_path, report)
    input_dir = output / "input"
    input_dir.mkdir(exist_ok=True)
    for path in expected[1:]:
        target = input_dir / path.name
        if target.exists():
            if digest(target) != digest(path):
                raise ValueError(f"Copied input changed: {target}")
        else:
            shutil.copyfile(path, target)
    ini = configparser.ConfigParser()
    ini.read(original_ini)
    ini["path"]["path_input"] = input_dir.as_posix()
    ini["path"]["path_output"] = output.as_posix()
    ini["parallel"]["processes_num"] = str(processes)
    ini["parallel"]["check_finished"] = "True"
    ini_path = output / "case.ini"
    with ini_path.open("w", encoding="utf-8") as stream:
        ini.write(stream)
    config = CfsConfig()
    config.read_config(str(ini_path))
    report["processes"] = processes
    save_json(manifest_path, report)
    return config, report


def summarize(path, name):
    data = np.load(path) if Path(path).suffix == ".npy" else np.loadtxt(path, delimiter=",", ndmin=2)
    if not np.isfinite(data).all():
        raise AssertionError(f"Nonfinite values: {path}")
    return {"quantity": name, "file": str(path), "shape": list(data.shape),
            "min": float(data.min()), "max": float(data.max()),
            "rms": float(np.sqrt(np.mean(data * data))), "sha256": digest(path)}


def record_stage(output, report, name, function, force=False):
    entry = report["stages"].get(name, {})
    if entry.get("status") == "complete" and not force:
        print(f"Stage already complete: {name}", flush=True)
        return
    started = time.perf_counter()
    report["stages"][name] = {"status": "running", "started_local": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_json(output / "run.json", report)
    try:
        result = function()
        report["stages"][name].update(status="complete", elapsed_seconds=time.perf_counter()-started, result=result)
    except BaseException as exc:
        report["stages"][name].update(status="failed", elapsed_seconds=time.perf_counter()-started,
                                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(output / f"stage-{name}.json", report["stages"][name])
        latest = json.loads((output / "run.json").read_text(encoding="utf-8"))
        latest["stages"][name] = report["stages"][name]
        for other in ("static", "dynamic-library", "dynamic"):
            stage_file = output / f"stage-{other}.json"
            if other != name and stage_file.exists():
                saved = json.loads(stage_file.read_text(encoding="utf-8"))
                current = latest["stages"].get(other, {})
                if saved.get("started_local", "") >= current.get("started_local", ""):
                    latest["stages"][other] = saved
        save_json(output / "run.json", latest)
    print(f"Completed {name} in {time.perf_counter()-started:.1f} s", flush=True)


def nearest_nodes(values, grid):
    grid = np.asarray(grid)
    return sorted(set(float(grid[np.argmin(np.abs(grid-value))]) for value in values))


def build_static(config, library, receiver_depths, source_range=None, source_step=None, distance_range=None, distance_step=None, layered=True):
    c = config.copy()
    c.path_green_static = str(library)
    c.static_obs_depth_list = receiver_depths
    if source_range is not None:
        c.static_source_depth_range = source_range
        c.static_source_delta_depth = source_step
    if distance_range is not None:
        c.static_dist_range = distance_range
        c.static_delta_dist = distance_step
    c.layered = layered
    if not layered:
        c.lam = 25.21168e9
        c.mu = 31.12616e9
    Path(library).mkdir(parents=True, exist_ok=True)
    create_static_lib(c)
    return c


@contextlib.contextmanager
def reuse_focal_mechanisms():
    """Pass identical six-component tensors to the unchanged bulk reader."""
    import dyncfs.cfs_static as static_module
    from pygrnwang.focal_mechanism import check_convert_fm
    original = static_module.seek_edcmp2_bulk
    checked = False
    def read(**kwargs):
        nonlocal checked
        mechanisms = kwargs["focal_mechanism_arr"]
        if mechanisms.shape[1] != 3:
            return original(**kwargs)
        starts = np.r_[0, np.flatnonzero(np.any(mechanisms[1:] != mechanisms[:-1], axis=1)) + 1]
        counts = np.diff(np.r_[starts, len(mechanisms)])
        converted = np.repeat(np.asarray([check_convert_fm(row) for row in mechanisms[starts]]), counts, axis=0)
        if not checked:
            # Verify every distinct mechanism on actual query geometries.
            indices = np.unique(mechanisms, axis=0, return_index=True)[1]
            probe = {k: (v[indices] if isinstance(v, np.ndarray) and len(v) == len(mechanisms) else v)
                     for k, v in kwargs.items()}
            reference = original(**probe)
            probe["focal_mechanism_arr"] = converted[indices]
            optimized = original(**probe)
            np.testing.assert_array_equal(reference, optimized)
            print(f"Mechanism conversion check: {len(indices)} distinct mechanisms, bitwise equal", flush=True)
            checked = True
        kwargs["focal_mechanism_arr"] = converted
        return original(**kwargs)
    static_module.seek_edcmp2_bulk = read
    try:
        yield
    finally:
        static_module.seek_edcmp2_bulk = original


def compute_map_batched(config, output, spacing):
    """Use the existing grid kernel in latitude strips to bound pair-array memory."""
    c = config.copy()
    c.source_inds = [1, 2, 3, 4, 5]
    c.source_shapes = [[22, 9], [6, 9], [8, 9], [62, 9], [17, 6]]
    n_lat = cal_grid_num(c.obs_lat_range, spacing)
    n_lon = cal_grid_num(c.obs_lon_range, spacing)
    latitudes = np.linspace(*c.obs_lat_range, n_lat)
    strip_rows = max(1, 512 // n_lon)
    scratch = output / "_strips"
    scratch.mkdir(parents=True, exist_ok=True)
    c.path_output_results_static = str(scratch)
    keys = ["normal_vector_static", "rupture_vector_static", "normal_stress_static", "shear_stress_static", "cfs_static"]
    collected = {key: [] for key in keys}
    tensors = []
    for start in range(0, n_lat, strip_rows):
        stop = min(start + strip_rows, n_lat)
        print(f"Map latitude rows {start+1}-{stop}/{n_lat}", flush=True)
        compute_static_cfs_fix_depth(c, obs_depth=15, optimal_type=0,
            receiver_mechanism=[223, 47, 131],
            obs_lat_range=[float(latitudes[start]), float(latitudes[stop-1])],
            obs_lon_range=c.obs_lon_range, obs_delta_lat=spacing, obs_delta_lon=spacing)
        tensors.append(np.load(scratch / "stress_tensor_dep_15.00.npy"))
        for key in keys:
            collected[key].append(np.loadtxt(scratch / f"{key}_dep_15.00.csv", delimiter=",", ndmin=2))
    output.mkdir(exist_ok=True)
    np.save(output / "stress_tensor_dep_15.00.npy", np.concatenate(tensors))
    for key, arrays in collected.items():
        np.savetxt(output / f"{key}_dep_15.00.csv", np.concatenate(arrays), delimiter=",")
    return summarize(output / "cfs_static_dep_15.00.csv", "CFS, Pa")


def static_case(case, config, output, spacing):
    if case == "ludian":
        # The original nearest-neighbor table selects 4.75 km for a requested 5 km.
        nodes = nearest_nodes([config.fixed_obs_depth], config.static_obs_depth_list)
        c = build_static(config, output / "grn_s", nodes)
        compute_static_cfs_fix_depth(c)
        return {
            "receiver_library_depths_km": nodes,
            "cfs": summarize(Path(c.path_output_results_static)/"cfs_oop_static_dep_5.00.csv", "CFS, Pa"),
            "tensor": summarize(Path(c.path_output_results_static)/"stress_tensor_dep_5.00.npy", "NED stress, Pa"),
        }
    receivers = np.loadtxt(Path(config.path_input)/"obs_plane5.csv", delimiter=",")
    nodes = nearest_nodes(receivers[:, 2], config.static_obs_depth_list)
    c = build_static(config, output/"grn_s", nodes, distance_range=[0,800], distance_step=2)
    compute_static_cfs(c)
    snapshots = []
    for nt in range(0,80,2):
        cc = c.copy()
        cc.cut_stf = nt
        target = Path(c.path_output_results_static)/"time"/str(nt)
        target.mkdir(parents=True, exist_ok=True)
        cc.path_output_results_static = str(target)
        compute_static_cfs(cc)
        snapshots.append(summarize(target/"cfs_static_plane5.csv", f"cut_stf={nt}, Pa"))
    maps = {}
    for layered, label in [(True,"static_layer"), (False,"static_half")]:
        mc = build_static(config, output/f"grn_{label}", [15.],
            source_range=[0,30], source_step=1, distance_range=[0,1000], distance_step=1, layered=layered)
        target = output/"results"/label
        target.mkdir(parents=True, exist_ok=True)
        with reuse_focal_mechanisms():
            maps[label] = compute_map_batched(mc, target, spacing)
    return {"plane": summarize(Path(c.path_output_results_static)/"cfs_static_plane5.csv","CFS, Pa"),
            "receiver_library_depths_km": nodes, "snapshots": snapshots, "maps": maps,
            "map_spacing_note": "0.05 matches the supplied compute script; 0.01 matches the plotting INI"}


def dynamic_receivers(case, config):
    if case == "wenchuan":
        return np.loadtxt(Path(config.path_input)/"obs_plane5.csv", delimiter=",", ndmin=2)
    n_lat = cal_grid_num(config.obs_lat_range, config.obs_delta_lat)
    n_lon = cal_grid_num(config.obs_lon_range, config.obs_delta_lon)
    lat = np.linspace(*config.obs_lat_range, n_lat)
    lon = np.linspace(*config.obs_lon_range, n_lon)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return np.column_stack([lat_grid.ravel(), lon_grid.ravel(),
                            np.full(n_lat*n_lon, config.fixed_obs_depth), np.zeros((n_lat*n_lon,3))])


def required_jobs(case, config):
    """Keep the original interpolation nodes and original 100-distance input groups."""
    sources = read_source_array(config.source_inds, config.path_input)
    # Original synthesis skips STFs whose integral is zero. Thresholded moments
    # alone do not trigger that skip, so keep those source rows here.
    sources = sources[np.sum(sources[:, 10:], axis=1) != 0]
    receivers = dynamic_receivers(case, config)
    distances, _ = pairwise_spherical_dist_azimuth_km(
        sources[:,0], sources[:,1], receivers[:,0], receivers[:,1])
    distances = distances.reshape(len(sources), len(receivers))
    jobs = set()
    for i, source in enumerate(sources):
        src_low, src_high, _ = get_sorted_grid_params(source[2], config.event_depth_list)
        for rec_depth in np.unique(receivers[:,2]):
            rec_low, rec_high, _ = get_sorted_grid_params(rec_depth, config.receiver_depth_list)
            selected = distances[i, receivers[:,2] == rec_depth]
            fractional = (selected-config.grn_dist_range[0])/config.grn_delta_dist
            lower = np.maximum(0, np.floor(fractional).astype(int))
            upper = lower + (fractional-lower > 1e-4)
            groups = set((np.concatenate([lower,upper])//100).tolist())
            for sd in (src_low, src_high):
                for rd in (rec_low, rec_high):
                    for group in groups:
                        jobs.add((float(sd),float(rd),int(group)))
    return sorted(jobs), {
        "distance_min_km": float(distances.min()), "distance_max_km": float(distances.max()),
        "source_rows_with_nonzero_stf": len(sources), "receiver_count": len(receivers),
        "note": "No receiver, source or time grid is coarsened. Unqueried native jobs are omitted.",
    }


def dynamic_library(case, config, output):
    jobs, coverage = required_jobs(case, config)
    src_nodes = sorted(set(j[0] for j in jobs))
    rec_nodes = sorted(set(j[1] for j in jobs))
    c = config
    path = Path(c.path_green_dynamic)
    selection = {
        **coverage, "required_native_jobs": len(jobs), "jobs": jobs,
        "source_depth_nodes_km": src_nodes, "receiver_depth_nodes_km": rec_nodes,
        "original_source_nodes_km": c.event_depth_list,
        "original_receiver_nodes_km": c.receiver_depth_list,
        "original_distance_range_km": c.grn_dist_range,
        "distance_step_km": c.grn_delta_dist, "distances_per_native_group": 100,
    }
    save_json(output/"library_selection.json", selection)
    print(f"Preparing {len(jobs)} required native jobs, {len(src_nodes)} source and {len(rec_nodes)} receiver depths", flush=True)
    pre_process_qseis2025(
        processes_num=c.processes_num, path_green=c.path_green_dynamic,
        event_depth_list=src_nodes, receiver_depth_list=rec_nodes,
        dist_range=c.grn_dist_range, delta_dist=c.grn_delta_dist, N_each_group=100,
        time_window=c.time_window, sampling_interval=c.sampling_interval_cfs,
        output_observables=c.output_observables, slowness_int_algorithm=c.slowness_int_algorithm,
        eps_estimate_wavenumber=c.eps_estimate_wavenumber, source_radius_ratio=c.source_radius_ratio,
        slowness_window=c.slowness_window, time_reduction_velo=c.time_reduction_velo,
        wavenumber_sampling_rate=c.wavenumber_sampling_rate, anti_alias=c.anti_alias,
        free_surface=1-int(c.free_surface), wavelet_duration=c.wavelet_duration,
        wavelet_type=c.wavelet_type, flat_earth_transform=c.flat_earth_transform,
        path_nd=c.path_nd, earth_model_layer_num=c.earth_model_layer_num,
        check_finished_tpts_table=True,
    )
    groups = [[list(j) for j in jobs[i:i+c.processes_num]] for i in range(0,len(jobs),c.processes_num)]
    with (path/"group_list.pkl").open("wb") as stream:
        pickle.dump(groups, stream)
    create_grnlib_qseis2025_parallel(str(path), check_finished=True, convert_pd2bin=True, remove_pd=True)
    info = json.loads((path/"green_lib_info.json").read_text(encoding="utf-8"))
    for sd, rd, group in jobs:
        folder = path/f"{sd:.2f}"/f"{rd:.2f}"/f"{group}_0"
        n_dist = min(100, info["N_dist"]-group*100)
        for component in ("szz","szr","srr","stt","szt","srt"):
            n_sources = 2 if component in ("szt","srt") else 4
            file = folder/f"grn_{component}.bin"
            expected_size = n_dist*n_sources*c.sampling_num*4
            if not file.exists() or file.stat().st_size != expected_size:
                raise AssertionError(f"Incomplete native output: {file}")
            if not np.isfinite(np.fromfile(file,dtype=np.float32)).all():
                raise AssertionError(f"Nonfinite native output: {file}")
    return {k:v for k,v in selection.items() if k != "jobs"}


def dynamic_case(case, config, output):
    if case == "wenchuan":
        compute_dynamic_cfs_parallel(config)
        filename = "cfs_dynamic_plane5.csv"
    else:
        compute_dynamic_cfs_fix_depth_parallel(config)
        filename = "cfs_oop_dynamic_dep_5.00.csv"
    result = Path(config.path_output_results_dynamic)/filename
    return {"cfs": summarize(result, "CFS, Pa")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=["wenchuan","ludian"])
    parser.add_argument("--stage", choices=["static","dynamic-library","dynamic","all"], default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--processes", type=int, default=12)
    parser.add_argument("--map-spacing", type=float, default=0.01,
                        help="Wenchuan map degrees: 0.01 matches plotting INI, 0.05 matches compute script")
    parser.add_argument("--force-stage", action="store_true")
    args = parser.parse_args()
    if args.processes < 1 or args.map_spacing <= 0:
        parser.error("processes and map-spacing must be positive")
    output = (args.output_dir or ROOT/"docs/_build/cases"/args.case).resolve()
    if output == (ROOT/"examples"/args.case).resolve() or ROOT/"examples"/args.case in output.parents:
        parser.error("Use an output directory outside the original case data")
    config, report = prepare(args.case, output, args.processes, args.map_spacing)
    functions = {
        "static": lambda: static_case(args.case,config,output,args.map_spacing),
        "dynamic-library": lambda: dynamic_library(args.case,config,output),
        "dynamic": lambda: dynamic_case(args.case,config,output),
    }
    selected = list(functions) if args.stage == "all" else [args.stage]
    for stage in selected:
        record_stage(output,report,stage,functions[stage],force=args.force_stage)
    print(f"Run record: {output/'run.json'}", flush=True)


if __name__ == "__main__":
    main()
