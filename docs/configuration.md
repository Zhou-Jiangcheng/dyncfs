# Configuration reference

Create a `CfsConfig`, then call `read_config(path)`. A bare constructor
leaves most attributes as `None`; it is not a ready-to-run configuration.

The parser expects all general sections even for a static-only job.
Python-style lists are parsed with `ast.literal_eval`; booleans use
ConfigParser syntax. Paths are literal strings: use forward slashes on
Windows, without Python quotes, environment-variable placeholders or
`~`. Relative paths resolve against the **process working directory**,
not the INI's directory.

Reading a configuration creates the static/dynamic library and result
directories. It does not comprehensively validate input geometry.

## Paths and source/receiver settings

| Section/key | Meaning |
|---|---|
| `[path] path_input` | Directory containing the input files |
| `path_output` | Root for libraries, per-point files and results |
| `[input_addition] optimal_type` | 0 fixed mechanism, 1 optimal rake, 2 optimal planes |
| `tectonic_stress_type` | 1 full tensor, 2 principal-axis directions; read only for modes 1/2 |
| `tectonic_stress` | Six NED tensor entries in MPa (type 1), or six angles in degrees (type 2) |
| `mu_f`, `B_pore` | Dimensionless friction and Skempton coefficients |
| `source_inds`, `obs_inds` | Lists of selected CSV IDs |
| `source_shapes`, `obs_shapes` | One `[n_strike, n_dip]` per selected plane |
| `source_ref`, `obs_ref` | Reference latitude/longitude in degrees for geometry and plotting |
| `earth_model_layer_num` | Model selection count forwarded to preprocessing |
| `use_spherical` | False: QSEIS2025; True: QSSP2020 for dynamic stress |
| `slip_thresh` | Slip threshold in m; positive values zero smaller slip/moment |
| `cut_stf` | Number of STF samples retained; values ≤0 disable truncation |
| `correct_zero_freq` | Optional boolean, default False; use static tensors for dynamic correction |

Mode 1 requires `tectonic_stress_type=1`; type 2 raises `ValueError`.
Static calculations remain EDGRN/EDCMP regardless of `use_spherical`.

## Fixed-depth observation grid

All `[fixed_obs_depth]` fields are parsed even if the grid is disabled.

| Key | Meaning |
|---|---|
| `fixed_obs_depth` | Depth in km; CLI and complete workflows enable the grid only for values >0 |
| `obs_lat_range`, `obs_lon_range` | Closed `[minimum, maximum]` ranges, degrees |
| `obs_delta_lat`, `obs_delta_lon` | Positive target increments, degrees |

Grid counts are computed by `cal_grid_num`, then coordinates use
`linspace` including both endpoints. Choose ranges divisible by the
increments. Longitude varies fastest; see [output layouts](guides/outputs.md).
Direct low-level grid functions do not enforce the CLI's positive-depth gate.

If no explicit receiver mechanism is passed, modes 0/1 derive one from the
moment-weighted sum of the selected source mechanisms. The CLI has no
receiver-mechanism override; use the Python API for an explicit choice.

## Library coverage

| `[grn_region]` key | Meaning |
|---|---|
| `grn_source_depth_range` | Closed source-depth range, km |
| `grn_delta_source_depth` | Source-depth increment, km |
| `grn_obs_depth_range` | Closed receiver-depth range, km |
| `grn_delta_obs_depth` | Receiver-depth increment, km |
| `grn_dist_unit` | `km` or `deg` |
| `grn_dist_range` | Closed epicentral-distance range, in the selected unit |
| `grn_delta_dist` | Distance increment in the selected unit |

Degree distances are converted to km during parsing. Depth lists use
rounded interval counts and `linspace`. Select integral interval ratios,
cover every source/receiver pair, and allow margin around queries.
EDGRN requires at least two source depths; keep the minimum distance
positive for the shared introductory configuration.

Derived static attributes include `static_source_depth_range`,
`static_source_delta_depth`, `static_dist_range`, `static_delta_dist`
and `static_obs_depth_list`. Dynamic attributes include
`event_depth_list` and `receiver_depth_list`. Changing a general field
after parsing does not automatically recompute all derived attributes;
edit the INI and read it again when changing library geometry.

## Sampling and parallelism

| Section/key | Meaning |
|---|---|
| `[time_window] sampling_interval_stf` | STF sample interval, s |
| `sampling_interval_cfs` | Dynamic sample interval, s |
| `sampling_num` | Number of dynamic output samples |
| `max_frequency` | QSSP2020 maximum frequency, Hz; missing/unparseable value uses Nyquist |
| `[parallel] processes_num` | Positive process count; 1 selects sequential dynamic synthesis |
| `check_finished` | Reuse eligible backend outputs and matching dynamic stress caches |

The time window is `(sampling_num-1)*sampling_interval_cfs`.
`max_frequency` is not forwarded to the QSEIS2025 builder.

## Solver defaults

With `[default_config] default_config=True`, the parser calls
`set_default()` and ignores custom `[static]` and `[dynamic]` values.

| Setting | Default |
|---|---|
| Static `wavenumber_sampling_rate` | 12; also forwarded to QSEIS2025 |
| Static `layered` | True |
| `max_slowness` | None for QSEIS2025; `1/min(nonzero Vs)+0.1` s/km for QSSP2020 |
| `anti_alias`, `free_surface` | 0.01, True |
| `wavelet_duration` | 5 CFS samples; converted to seconds for QSSP |
| `output_observables` | QSEIS: `[0,0,0,1,0]`; QSSP: 11 entries with index 5 enabled |
| `slowness_int_algorithm` | 0 |
| `eps_estimate_wavenumber` | 1e-6 |
| `source_radius_ratio` | 0.05 |
| `slowness_window` | None |
| `time_reduction_velo` | 0 |
| `wavelet_type` | 2 |
| `flat_earth_transform` | True |
| QSSP `time_reduction` | -20 s |
| QSSP `source_radius` | 0 km |
| `turning_point_filter`, `turning_point_d1`, `turning_point_d2` | 0, 0, 0 |
| `gravity_fc`, `gravity_harmonic` | 0, 0 |
| `cal_sph`, `cal_tor` | 1, 1 |
| `min_harmonic`, `max_harmonic` | 6000, 25000 |
| `physical_dispersion` | 0 |
| Derived `spec_time_window` | Same as `time_window` |

These are implementation defaults, not universally converged scientific
settings. DynCFS translates its common `free_surface` boolean differently
for the two backend input conventions.

## Custom solver settings

With `default_config=False`, provide all settings shown in the bundled
`[static]` and `[dynamic]` sections, including those for the other dynamic
backend: the parser reads both groups.

For static `layered=False`, supply Lamé parameters `lam` and `mu` in Pa.
The library wrapper still prepares and invokes EDGRN before EDCMP.
For custom dynamics, `max_slowness=None` becomes None and
`slowness_window=[0,0,0,0]` becomes None. Keep the stress observable enabled
with the correct backend-specific list length.

The annotated bundled file is a detailed reference, but its paths and grid
sizes must be adapted:

```{literalinclude} ../examples/wenchuan/wenchuan.ini
:language: ini
:start-at: [static]
```

See [dynamic calculations](guides/dynamic.md) for zero-frequency correction
and [parallel execution](guides/parallel.md) for process behavior.
