# Output files and array layouts

## Directory structure

```text
path_output/
├── grn_s/                     # Static library, models and metadata
├── grn_d/
│   ├── qseis/                 # Layered dynamic library, or
│   ├── qssp/                  # Spherical dynamic library
│   └── results_each/          # Dynamic per-point arrays and job/cache metadata
└── results/
    ├── static/                # Static tensors and projected CSV files
    └── dynamic/               # Assembled dynamic CSV files
```

Configuration parsing creates library/result roots. Libraries contain their
own backend-specific files and `green_lib_info.json`; retain these together.

## Filename patterns

Let `n` be an observation plane ID and `z` a depth formatted to two decimals.

| Quantity/mode | Fault result | Fixed-depth result |
|---|---|---|
| Static tensor, all modes | `stress_tensor_plane<n>.npy` | `stress_tensor_dep_<z>.npy` |
| Static CFS, mode 0 | `cfs_static_plane<n>.csv` | `cfs_static_dep_<z>.csv` |
| Static CFS, mode 1 | `cfs_os_static_plane<n>.csv` | `cfs_os_static_dep_<z>.csv` |
| Static CFS, mode 2 | `cfs_oop_static_plane<n>.csv` | `cfs_oop_static_dep_<z>.csv` |
| Dynamic CFS, mode 0 | `cfs_dynamic_plane<n>.csv` | `cfs_dynamic_dep_<z>.csv` |
| Dynamic CFS, mode 1 | `cfs_os_dynamic_plane<n>.csv` | `cfs_os_dynamic_dep_<z>.csv` |
| Dynamic CFS, mode 2 | `cfs_oop_dynamic_plane<n>.csv` | `cfs_oop_dynamic_dep_<z>.csv` |

Other keys are `normal_vector`, `rupture_vector`, `normal_stress` and
`shear_stress`; optimal-rake mode adds `rake`.
OOP outputs use numbered quantities such as `normal_vector1` and
`normal_vector2`. Dynamic tensor CSVs use the key `stress_ned` and include
the mode marker, e.g. `stress_ned_os_dynamic_plane1.csv`.

Depth filenames have only two decimal places, so nearby depths can produce
the same name. Different receiver modes retain separate projected filenames,
but static tensor files are shared.

## Shapes

For N receivers and T time samples:

| Data | Shape |
|---|---|
| Static tensor NPY | `(N, 6)` |
| Static scalar CSV | `(N, 1)` |
| Static vector CSV | `(N, 3)` |
| Dynamic per-point tensor NPY | `(T, 6)` |
| Dynamic per-point vector NPY | `(T, 3)` |
| Dynamic per-point scalar NPY | `(T,)` |
| Dynamic scalar CSV | `(N, T)` |
| Dynamic vector CSV | `(3*N, T)` |
| Dynamic tensor CSV | `(6*N, T)` |

All CSVs omit headers and index columns. Dynamic vector/tensor rows are
**receiver-major**: all components of receiver 0, then receiver 1, etc.
Tensor order is `[NN, NE, ND, EE, ED, DD]`. Vectors use NED.
Stress is in Pa, orientation angles in degrees, unit vectors dimensionless.

```python
from pathlib import Path
import numpy as np

result = Path("my-output/results/dynamic")
raw = np.loadtxt(result / "stress_ned_dynamic_plane1.csv", delimiter=",", ndmin=2)
n_receivers = raw.shape[0] // 6
stress = raw.reshape(n_receivers, 6, raw.shape[1]).transpose(0, 2, 1)
# stress.shape == (n_receivers, time_samples, 6)
```

## Geographic grid order

Let `n_lat = cal_grid_num(obs_lat_range, obs_delta_lat)` and likewise for
longitude. The flattened index is `i_lat * n_lon + i_lon`.
Reshape a scalar snapshot to `(n_lat, n_lon)`.

Preserve the INI, input CSVs and row order with your outputs; result CSVs
do not carry receiver coordinates or a time column.
Parallel depth preparation additionally saves `obs_plane_<z>.npy`
under `grn_d/results_each/`.

## Per-point cache names

Dynamic filenames begin with latitude, longitude and depth, each to four
decimals, separated by underscores. The tensor cache uses
`*_stress_ned.npy` and `*_stress_ned.json`.
See [reuse rules](parallel.md) before treating existing files as complete.
