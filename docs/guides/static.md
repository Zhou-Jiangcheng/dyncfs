# Static calculations

## Build and compute on observation faults

```python
from pathlib import Path
from dyncfs.configuration import CfsConfig
from dyncfs.cfs_static import create_static_lib, compute_static_cfs

if __name__ == "__main__":
    config = CfsConfig()
    config.read_config(str(Path("case.ini").resolve()))
    create_static_lib(config)
    compute_static_cfs(config)
```

Use absolute input/output paths inside the INI. `create_static_lib`
prepares EDGRN, runs it, prepares EDCMP with stress output, runs it and
converts the tables to binary form. `check_finished` is passed to the
backend runners.

`compute_static_cfs` reads the selected patches, applies slip/STF
truncation, queries all source–receiver pairs and sums their tensors.
It saves NED tensors before resolving normal, shear and Coulomb stresses
according to `optimal_type`.

## Calculate a horizontal grid

```python
from pathlib import Path
from dyncfs.configuration import CfsConfig
from dyncfs.cfs_static import compute_static_cfs_fix_depth

if __name__ == "__main__":
    config = CfsConfig()
    config.read_config(str(Path("case.ini").resolve()))
    # The existing static library must cover these depths and distances.
    compute_static_cfs_fix_depth(
        config, obs_depth=10, optimal_type=0,
        receiver_mechanism=[30, 45, 90],
        obs_lat_range=[30.0, 30.2], obs_lon_range=[103.0, 103.2],
        obs_delta_lat=0.1, obs_delta_lon=0.1,
    )
```

This creates nine receivers with longitude varying fastest.
Depth is in km, angles and geographic spacing in degrees.
Omitted options use the configuration. If a mechanism is omitted,
modes 0/1 obtain one from the summed source moment tensor.

The direct grid function accepts an explicit depth independently of
`fixed_obs_depth_enabled()`. The CLI and `run_all_static` apply the
`fixed_obs_depth>0` gate.

## Reuse and half-space calculations

Skip `create_static_lib` when using an unchanged compatible library.
For homogeneous half-space EDCMP, set `default_config=False`,
`layered=False`, and Lamé `lam`/`mu` in Pa. The current wrapper still
performs its EDGRN preparation stage; follow the full wrapper's prerequisites.
The [Wenchuan example](../cases/wenchuan.md) compares a half-space and a
layered static map.

Use a new output directory after changing model, geometry or solver settings.
Static calculations allocate arrays proportional to the number of
source–receiver pairs, including the final bulk stress array. Geometry
chunking does not bound the memory of the complete static workflow.
