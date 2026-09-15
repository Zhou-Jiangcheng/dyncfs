# Static API

All high-level functions accept a populated `CfsConfig` named `config`, write files below its output paths, print progress and return None. [Static workflows](../guides/static.md) explain prerequisites.

```{py:function} dyncfs.cfs_static.create_static_lib(config)
Prepare and run EDGRN2/EDCMP2 and convert the static stress library. Uses configuration model, geometry, material, process and reuse settings. Requires native backend executables.
```

```{py:function} dyncfs.cfs_static.compute_static_cfs(config)
Compute static NED tensors and projected stresses at selected observation faults using an existing library. Outputs one tensor NPY and mode-specific CSV files per plane; see [layouts](../guides/outputs.md).
```

```{py:function} dyncfs.cfs_static.compute_static_cfs_fix_depth(config, obs_depth=None, optimal_type=None, receiver_mechanism=None, obs_lat_range=None, obs_lon_range=None, obs_delta_lat=None, obs_delta_lon=None)
`obs_depth` is km; `optimal_type` is 0/1/2; `receiver_mechanism` is `[strike,dip,rake]` in degrees. Geographic ranges are closed two-item lists in degrees, and increments are positive degrees. None uses the matching config attribute, except a missing receiver mechanism is derived from the moment-weighted source tensor in modes 0/1. Generates longitude-fast grid results. This direct call does not enforce the CLI's positive-depth gate.
```

```{py:function} dyncfs.cfs_static.run_all_static(config)
Create the library, compute selected observation faults, then compute the fixed-depth grid if enabled. This also rebuilds/prepares the library; use compute-only functions when appropriate.
```
