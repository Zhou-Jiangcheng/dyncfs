# Plotting API

These functions create/save/show figures and normally return None.
They read existing input/result files; they do not compute new stress.
See [plotting workflows](../guides/plotting.md).

## Common parameters

| Parameters | Meaning |
|---|---|
| `path_input`, `path_output` | Source input directory or calculation output root |
| `ind_obs`, `obs_inds` | Single observation ID or ordered ID list |
| `obs_shape`, `obs_shapes` | One shape or matching list of `[n_strike,n_dip]` |
| `ind_source`, `plane_inds` | Source ID or list |
| `source_shape`, `plane_shapes` | Corresponding source grid shapes |
| `sub_length_strike_km`, `sub_length_dip_km` | Patch dimensions in km for 2D plots |
| `sub_length_strike`, `sub_length_dip` | Patch dimensions in km for 3D geometry |
| `color_saturation` | Symmetric stress limit in Pa for CFS plots; slip scale in m for slip plots; None chooses from data |
| `tick_interval` | Tick stride in plotted grid samples for fault-plane maps |
| `zoom_strike`, `zoom_dip`, `zoom_lat`, `zoom_lon` | Plot interpolation factors, not physical grid refinement |
| `show`, `save` | Display interactively and/or save the figure |
| `nt`, `nt_list` | Zero-based CFS sample index or index list |
| `nt_cut`, `nt_cut_list` | STF sample cutoff or cutoff list |
| `sampling_interval_cfs`, `sampling_interval_stf` | Seconds per sample for labels and source-time handling |
| `elev`, `azim` | Camera angles in degrees |
| `obs_depth` | Receiver depth in km |
| `obs_lat_range`, `obs_lon_range` | Two-entry geographic ranges in degrees |
| `obs_delta_lat`, `obs_delta_lon` | Geographic grid spacing in degrees |
| `delta_tick` | Map tick spacing in degrees, automatic if None |
| `slip_thresh` | Slip threshold in m |
| `sub_stress` | Explicit scalar stress array in Pa |

Map range/spacing arguments must be supplied despite their None defaults.
Fault plotters read fixed-mechanism filenames; use an explicit array to plot
optimized results. Some plotting modules set global Matplotlib rcParams on import.

```{py:function} dyncfs.plot_cfs_static_2d.plot_cfs_static_2d(path_output, ind_obs, obs_shape, sub_length_strike_km, sub_length_dip_km, color_saturation=None, tick_interval=5, zoom_strike=1, zoom_dip=1, show=True, save=True)
Static fault-plane CFS map, displayed in MPa.
```

```{py:function} dyncfs.plot_cfs_static_2d.plot_cfs_static_fix_depth(path_output, sub_stress, obs_depth, obs_lat_range=None, obs_lon_range=None, obs_delta_lat=None, obs_delta_lon=None, color_saturation=None, zoom_lat=1, zoom_lon=1, show=True, save=True, delta_tick=None)
Plot the supplied scalar array on a latitude–longitude grid.
```

```{py:function} dyncfs.plot_cfs_static_3d.plot_staic_coulomb_stress_3d(elev, azim, path_input, path_output, obs_inds, obs_shapes, sub_length_strike, sub_length_dip, color_saturation=None, show=True, save=True)
Static CFS on 3D receiver patches. The spelling `staic` is the actual exported function name.
```

```{py:function} dyncfs.plot_cfs_dynamic_2d.plot_cfs_dynamic_2d_nt(path_output, nt, sampling_interval_cfs, ind_obs, obs_shape, sub_length_strike_km, sub_length_dip_km, color_saturation=None, tick_interval=5, zoom_strike=1, zoom_dip=1, show=True, save=True)
Dynamic CFS snapshot at sample `nt`.
```

```{py:function} dyncfs.plot_cfs_dynamic_2d.plot_cfs_dynamic_2d_series(path_output, nt_list, sampling_interval_cfs, ind_obs, obs_shape, sub_length_strike_km, sub_length_dip_km, color_saturation=None, tick_interval=5, zoom_strike=1, zoom_dip=1, show=True, save=True)
Series of fault-plane dynamic CFS snapshots.
```

```{py:function} dyncfs.plot_cfs_dynamic_2d.plot_cfs_dynamic_fix_depth_one_time_point(path_output, nt, sampling_interval_cfs, obs_depth, obs_lat_range=None, obs_lon_range=None, obs_delta_lat=None, obs_delta_lon=None, color_saturation=None, zoom_lat=1, zoom_lon=1, show=True, save=True, delta_tick=None)
Geographic dynamic snapshot, reading fixed-mechanism depth output.
```

```{py:function} dyncfs.plot_cfs_dynamic_3d.plot_dynamic_coulomb_stress_3d_nt(elev, azim, nt, sampling_interval_cfs, path_input, path_output, obs_inds, obs_shapes, sub_length_strike, sub_length_dip, color_saturation=None, show=True, save=True)
3D dynamic snapshot. Patch lengths may be scalars, per-plane entries or per-patch arrays; use entries matching the selected plane's patch count.
```

```{py:function} dyncfs.plot_slip_2d.plot_slip_2d(path_input, nt_cut, sampling_interval_stf, ind_source, source_shape, sub_length_strike_km, sub_length_dip_km, slip_thresh=0, color_saturation=None, tick_interval=5, zoom_strike=1, zoom_dip=1, show=True, save=True)
2D source slip map at an STF cutoff.
```

```{py:function} dyncfs.plot_slip_2d.plot_slip_2d_series(path_input, nt_cut_list, sampling_interval_stf, ind_source, source_shape, sub_length_strike_km, sub_length_dip_km, color_saturation=None, tick_interval=5, zoom_strike=1, zoom_dip=1, show=True, save=True)
Series of 2D source slip maps.
```

```{py:function} dyncfs.plot_slip_3d.plot_slip_3d(elev, azim, nt_cut, sampling_interval_stf, path_input, plane_inds, plane_shapes, slip_thresh=0, color_saturation=None, save=True, show=True)
3D source slip geometry from the selected source CSVs.
```

```{py:function} dyncfs.plot_gif.images_to_gif(image_files, output_file, duration=0.2, loop=0)
`image_files` is an ordered path list and `output_file` the GIF destination. `duration` is forwarded unchanged to imageio's writer; verify its timing convention. `loop=0` requests indefinite looping. Requires the optional `imageio` package and returns None.
```
