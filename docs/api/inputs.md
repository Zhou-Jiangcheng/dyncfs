# Input conversion and utilities

Paths are filesystem paths. Create destination directories before
conversion. See [input files](../input-files.md) for column and row order.

```{py:function} dyncfs.convert_input_format.convert_usgs_basic2source_csvs(path_rup_model, path_input_dir, sampling_interval_stf)
`path_rup_model` names USGS `basic_inversion.param`; `path_input_dir` is the destination; `sampling_interval_stf` is seconds. Write one source CSV per segment with triangular STFs and return `[[n_strike,n_dip], ...]`. Converts slip cm to m and moment dyne cm to N m. Raises ValueError if no fault segments are found.
```

```{py:function} dyncfs.convert_input_format.convert_fsp2source_csvs(path_fsp, path_input_dir, sampling_interval_stf, rise_ratio=0.5)
Read single/multiple-segment FSP from `path_fsp`, write source CSVs into `path_input_dir` and return segment shapes. STF interval is seconds; `rise_ratio` in [0,1] allocates the total rise duration to the increasing branch. Required field and shape checks may raise ValueError.
```

```{py:function} dyncfs.convert_input_format.create_triangle_stfs(t_rup, t_ris, t_fal, m0, srate_stf)
Per-patch arrays `t_rup`, `t_ris`, `t_fal` are rupture delay, rise and fall times in seconds; `m0` is N m and `srate_stf` is Hz. Return moment-rate samples `(N,nt)` normalized by discrete integration to `m0`. Durations below one sample become an impulse carrying the moment. Inputs should have matching lengths and nonnegative times.
```

```{py:function} dyncfs.convert_input_format.write_source_plane_csv(path_csv, lat_lon_dep, strike, dip, rake, length_strike, length_dip, slip_m, m0, t_rup, t_ris, t_fal, srate_stf, nx, nz)
Write `path_csv` and return None. Coordinates are `(N,3)` in deg/deg/km; mechanism arrays are degrees; lengths are km; slip is m; moment is N m; STF times are seconds and rate is Hz. `nx` and `nz` are counts along strike/dip and must multiply to N. Input is dip-row-major; output is strike-major with dip varying fastest. The function creates triangular STFs and reorders all columns together.
```

```{py:function} dyncfs.convert_input_format.convert_source_csvs2coulomb3(config, obs_depth, possion_ratio=0.25, youngs_modulus=800000.0)
Use configured source geometry to write `config.path_input/coulomb3.inp`. `obs_depth` is km, `possion_ratio` is dimensionless and `youngs_modulus` is bar. Preserve the historical parameter spelling. Returns None; does not run Coulomb3.
```

```{py:function} dyncfs.utils.read_source_array(source_inds, path_input, shift2corner=False)
Read and concatenate numbered source CSVs in ID-list order. Drop rows with missing fields. `shift2corner=True` moves patch centers to a top-left corner approximation; normal DynCFS computation uses False. Returns the combined numeric array; selected files need equal column counts.
```

```{py:function} dyncfs.utils.read_nd(path_nd, with_Q=False)
Read numeric model rows from `path_nd`, skipping single-token labels. Return `(-1,4)` without Q or `(-1,6)` with Q. `with_Q` selects the expected file schema; it does not discard input columns.
```

```{py:function} dyncfs.utils.cal_grid_num(value_range, delta, decimals=8)
Return a grid count for the closed two-entry `value_range` using positive `delta`, rounding interval arithmetic to `decimals`. Use the same units for range and delta; grid consumers construct endpoints with linspace.
```

```{py:function} dyncfs.utils.ignore_slip_source_array(source_array, slip_thresh)
Return a copied source array with slip and moment zeroed wherever column 8 is less than `slip_thresh` in m. Rows and STF columns are retained.
```

```{py:function} dyncfs.utils.cut_stf_modify_source_array(source_array, cut_stf)
Return a copied source array with STF samples at and after the positive index `cut_stf` zeroed. Rescale slip and moment by the retained/original STF sum ratio; zero-integral rows get zero ratio. High-level callers skip this function for nonpositive cutoffs.
```
