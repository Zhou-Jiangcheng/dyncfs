"""Small static DynCFS calculation; run from a source checkout."""
import argparse
import configparser
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pygrnwang

from dyncfs import __version__
from dyncfs.configuration import CfsConfig
from dyncfs.cfs_static import create_static_lib, compute_static_cfs
from pygrnwang.geo import d2km


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/_build/quickstart")
    parser.add_argument("--prepare-only", action="store_true", help="Write inputs without running solvers")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new or empty output directory to keep calculations separate.")
    started = time.perf_counter()
    input_dir = output / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "examples/wenchuan/input/model.nd", input_dir / "model.nd")
    distances = np.array([30.0, 60.0, 90.0])
    # Patch center; static synthesis uses area and slip. STF is included for the file schema.
    source = [[0, 0, 10, 30, 45, 90, 1, 1, 1, 3.112616e16, 0, 0.5, 1, 0.5, 0]]
    receivers = np.array([[0, d / d2km, 5, 30, 45, 90] for d in distances])
    np.savetxt(input_dir / "source_plane1.csv", source, delimiter=",")
    np.savetxt(input_dir / "obs_plane1.csv", receivers, delimiter=",")
    ini = configparser.ConfigParser()
    ini.read(ROOT / "docs/examples/quickstart.ini", encoding="utf-8")
    ini["path"]["path_input"] = input_dir.as_posix()
    ini["path"]["path_output"] = output.as_posix()
    config_path = output / "quickstart.ini"
    with config_path.open("w", encoding="utf-8") as stream:
        ini.write(stream)
    print(f"Configuration: {config_path}")
    if args.prepare_only:
        return
    config = CfsConfig()
    config.read_config(str(config_path))
    create_static_lib(config)
    compute_static_cfs(config)
    result_dir = Path(config.path_output_results_static)
    stress = np.load(result_dir / "stress_tensor_plane1.npy")
    names = ["normal_stress_static", "shear_stress_static", "cfs_static"]
    values = np.column_stack([
        np.loadtxt(result_dir / f"{name}_plane1.csv", delimiter=",", ndmin=1)
        for name in names
    ])
    if stress.shape != (3, 6) or values.shape != (3, 3):
        raise AssertionError(f"Unexpected output shapes: {stress.shape}, {values.shape}")
    if not np.isfinite(stress).all() or not np.isfinite(values).all() or not np.any(values):
        raise AssertionError("Results must be finite and nonzero")
    np.testing.assert_allclose(values[:, 2], values[:, 1] + config.mu_f * values[:, 0],
                               rtol=1e-12, atol=1e-10)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for index, label in enumerate(["Normal stress", "Shear stress", "Coulomb stress"]):
        ax.plot(distances, values[:, index] / 1000, "o-", label=label)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(xlabel="Epicentral distance (km)", ylabel="Stress change (kPa)",
           title="DynCFS · a 1 km² patch with 1 m slip")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output / "static_cfs.png", dpi=160)
    plt.close(fig)
    report = {
        "dyncfs": __version__, "pygrnwang_source_version": pygrnwang.__version__,
        "pygrnwang_source": str(Path(pygrnwang.__file__).resolve()),
        "python": platform.python_version(), "platform": platform.platform(),
        "backend": "EDGRN2 + EDCMP2", "elapsed_seconds": time.perf_counter() - started,
        "stress_shape": list(stress.shape), "resolved_stress_shape": list(values.shape),
        "distance_km": distances.tolist(), "stress_unit": "Pa",
        "normal_shear_cfs_pa": values.tolist(),
        "checks": ["finite nonzero outputs", "expected array shapes", "CFS = shear + 0.4 * normal"],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
