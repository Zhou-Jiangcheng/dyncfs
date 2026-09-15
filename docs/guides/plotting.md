# Plotting and export

DynCFS includes 2D fault maps, geographic depth maps, 3D fault views and
slip plots. They use Matplotlib. Use `show=False` for batch work and
`save=True` to write figures. Stress color limits are supplied in **Pa**;
the built-in stress plots display MPa.

## Plot a fault

```python
from dyncfs.plot_cfs_static_2d import plot_cfs_static_2d

plot_cfs_static_2d(
    path_output="my-output", ind_obs=1, obs_shape=[12, 8],
    sub_length_strike_km=2, sub_length_dip_km=2,
    color_saturation=1e5, show=False,
)
```

The shape and patch dimensions must match that receiver grid.
This plotter reads fixed-mechanism `cfs_static_plane1.csv`.
It does not select an optimal-rake or OOP filename automatically.

For a dynamic snapshot, use `plot_cfs_dynamic_2d_nt`.
`nt` is a zero-based sample index, not seconds; provide the CFS sampling
interval for the time label. A series function accepts `nt_list`.

## Plot a geographic depth slice

```python
import numpy as np
from dyncfs.plot_cfs_static_2d import plot_cfs_static_fix_depth

cfs = np.loadtxt("my-output/results/static/cfs_static_dep_10.00.csv", delimiter=",")
plot_cfs_static_fix_depth(
    path_output="my-output", sub_stress=cfs, obs_depth=10,
    obs_lat_range=[30, 30.2], obs_lon_range=[103, 103.2],
    obs_delta_lat=0.1, obs_delta_lon=0.1, show=False,
)
```

The static depth plot accepts an explicit array, so it can display an
optimized-mode result by loading the corresponding file.
Supply all geographic ranges and increments even though their API defaults
are None. Zoom parameters interpolate the image; they do not refine the
physical calculation.

## 3D views and slip

The static 3D entry point is spelled
`plot_staic_coulomb_stress_3d` in the source. Preserve that spelling when
importing it. Dynamic 3D uses `plot_dynamic_coulomb_stress_3d_nt`.
Provide camera elevation/azimuth, plane shapes and patch dimensions.

Slip routines read source CSVs. `nt_cut` is an STF sample cutoff, and
`sampling_interval_stf` supplies the time scale. Plotting settings do not
change saved calculation results.

## Animation

`dyncfs.plot_gif.images_to_gif` combines an explicitly ordered list of
images and requires `imageio`. It forwards `duration` unchanged to
`imageio.v3.imwrite`; timing units can depend on the selected writer.
Check the resulting playback speed instead of assuming the historical
seconds wording applies to every plugin.

The [plotting API](../api/plotting.md) lists exact signatures.
