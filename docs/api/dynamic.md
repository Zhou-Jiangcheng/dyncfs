# Dynamic API

Import from `dyncfs.cfs_dynamic`. High-level functions take a populated
`config`, write files, report progress and return None.
The [dynamic guide](../guides/dynamic.md) explains library compatibility.

## Shared low-level arguments

| Argument | Meaning |
|---|---|
| `path_green` | Absolute root of the selected dynamic library |
| `source_array` | Source rows in the [CSV schema](../input-files.md); the three single-point CFS functions also accept a string path to an NPY |
| `obs_array_single_point` | Receiver coordinates in deg/deg/km; append strike/dip/rake for fixed mode, strike/dip for optimal rake |
| `srate_stf` | STF sampling rate, Hz |
| `static_stress` | Optional six-entry **ENU** static tensor, Pa, for correction |
| `max_slowness` | Optional s/km bound used for tail handling/correction |
| `green_info` | Library metadata dict; None loads `green_lib_info.json` |
| `use_spherical` | False for QSEIS2025, True for QSSP2020 |
| `mu_f`, `B_pore` | Dimensionless friction and pore-pressure coefficients |
| `path_results_each` | Existing directory for per-point outputs, or None to avoid those saves |
| `check_finish` | Reuse a matching stress tensor cache if True |
| `tectonic_stress_type`, `tectonic_stress` | Type 1 NED tensor in Pa, or type 2 ordered principal-axis angles in degrees |

Returned time sampling comes from library metadata. Tensor arrays are
`(T,6)` NED, vectors `(T,3)`, scalar series `(T,)`.
The spelling is `check_finish` in single-point APIs, while the configuration
uses `check_finished`.

```{py:function} dyncfs.cfs_dynamic.create_dynamic_lib(config)
Prepare and run the backend selected by `config.use_spherical`, convert native output to binary and remove converted text tables. Requires the corresponding executable; QSSP also computes spectra.
```

```{py:function} dyncfs.cfs_dynamic.compute_dynamic_cfs_sequential(config)
Synthesize and project each selected observation point serially, save per-point arrays and assemble plane CSVs. Requires an existing dynamic library.
```

```{py:function} dyncfs.cfs_dynamic.compute_dynamic_cfs_parallel(config)
Prepare source/job files, run spawned workers and assemble plane CSVs. The calling script requires a main guard.
```

```{py:function} dyncfs.cfs_dynamic.compute_dynamic_cfs_fix_depth_sequential(config, obs_depth=None, receiver_mechanism=None, obs_lat_range=None, obs_lon_range=None, obs_delta_lat=None, obs_delta_lon=None)
`obs_depth` is km. `receiver_mechanism` is `[strike,dip,rake]` in degrees; geographic ranges and increments are degrees. None uses configuration fields, except receiver mechanism is inferred from the source tensor in modes 0/1. Writes a longitude-fast grid.
```

```{py:function} dyncfs.cfs_dynamic.compute_dynamic_cfs_fix_depth_parallel(config, obs_depth=None, receiver_mechanism=None)
Parallel depth-grid computation. Explicit depth is km and mechanism is `[strike,dip,rake]` in degrees; None uses the same rules as the sequential function. Geographic ranges/increments come from `config`. Requires a main guard.
```

```{py:function} dyncfs.cfs_dynamic.run_all_dynamic(config)
Create a dynamic library, compute observation faults with the configured worker count, then compute the enabled fixed-depth grid. Does not calculate static tensors required by zero-frequency correction.
```

```{py:function} dyncfs.cfs_dynamic.synthesize_dynamic_stress(path_green, source_array, obs_array_single_point, srate_stf, static_stress=None, max_slowness=None, green_info=None, use_spherical=False)
Return the summed stress tensor `(T,6)` in Pa. Receiver input needs only the first three coordinates; `source_array` must be an array. Applies STF resampling, normalization, convolution and baseline/tail handling. Does not itself write a per-point cache.
```

```{py:function} dyncfs.cfs_dynamic.cal_cfs_dynamic_single_point_fm(path_green, source_array, obs_array_single_point, srate_stf, mu_f=0.4, B_pore=0, max_slowness=None, green_info=None, path_results_each=None, use_spherical=False, static_stress=None, check_finish=False)
Fixed mechanism. Receiver input has six entries. Return `(stress_ned,n,d,sigma,tau,cfs)`. When an output directory is supplied, save stress/cache metadata and projected arrays.
```

```{py:function} dyncfs.cfs_dynamic.cal_cfs_dynamic_single_point_opt_rake(path_green, source_array, obs_array_single_point, srate_stf, tectonic_stress, mu_f=0.4, B_pore=0, max_slowness=None, green_info=None, path_results_each=None, use_spherical=False, static_stress=None, check_finish=False)
Fix strike/dip, optimize rake using the full tectonic tensor and perturbation at each time. Receiver input needs latitude, longitude, depth, strike and dip (a sixth entry is ignored). Return `(stress_ned,n,d,sigma,tau,cfs,rake)`, with rake in degrees. Optional projected files use `_os_`.
```

```{py:function} dyncfs.cfs_dynamic.cal_cfs_dynamic_single_point_oop(path_green, source_array, obs_array_single_point, srate_stf, tectonic_stress_type, tectonic_stress, mu_f=0.4, B_pore=0, max_slowness=None, green_info=None, path_results_each=None, use_spherical=False, static_stress=None, check_finish=False)
Use receiver coordinates and tectonic input to construct optimal conjugate planes. Return `(stress_ned,n1,d1,sigma1,tau1,n2,d2,sigma2,tau2,cfs)`. Optional projected files use `_oop_`. The single returned CFS is evaluated on plane 2.
```
