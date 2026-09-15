# Dynamic calculations

## Select the backend

| Setting | Backend | Stress output selection |
|---|---|---|
| `use_spherical=False` | QSEIS2025, layered model | `[0,0,0,1,0]` |
| `use_spherical=True` | QSSP2020, spherical model | 11 entries, index 5 set to 1 |

`default_config=True` chooses the matching list. For custom settings,
keep this output enabled. QSEIS06 is not used by the current DynCFS
dispatcher. Old Green's-function libraries are not automatically migrated.

## Build and synthesize

```python
from pathlib import Path
from dyncfs.configuration import CfsConfig
from dyncfs.cfs_dynamic import create_dynamic_lib, compute_dynamic_cfs_sequential

if __name__ == "__main__":
    config = CfsConfig()
    config.read_config(str(Path("case.ini").resolve()))
    create_dynamic_lib(config)
    compute_dynamic_cfs_sequential(config)
```

For multiple workers, replace the last call with
`compute_dynamic_cfs_parallel(config)`. The complete
`run_all_dynamic(config)` builds the library, computes observation faults
and, if enabled, calculates the fixed-depth grid.

Both builders convert native tables to binary and request removal of
converted text tables. QSSP creation also requests spectrum calculation.
Retain metadata and model files with the library.

Synthesis resamples each source STF, normalizes it to seismic moment,
queries the selected stress Green's functions, subtracts the pre-P mean,
convolves, sums and converts ENU to NED. The resulting tensor has shape
`(sampling_num, 6)` for one receiver.

## Fixed-depth receivers

Use `compute_dynamic_cfs_fix_depth_sequential` or
`compute_dynamic_cfs_fix_depth_parallel`.
Both accept an explicit depth and receiver mechanism.
The sequential function additionally accepts geographic range/spacing
overrides. For the parallel function, put those settings in the configuration.

The default receiver mechanism for modes 0/1 comes from the summed source
moment tensor. Choose it explicitly through Python when that is not the
intended receiver.

## Static zero-frequency correction

Set `correct_zero_freq=True` only after computing static tensors for the
same sources and receivers. Keep their row order, geometry and source
normalization consistent. The loader checks receiver count, not the full
scientific configuration.

The required files are:

- Faults: `results/static/stress_tensor_plane<n>.npy`.
- Depth grids: `results/static/stress_tensor_dep_<depth to 2 decimals>.npy`.

A missing tensor raises `FileNotFoundError`; a receiver-count mismatch
raises `ValueError`. The high-level loader converts NED to ENU for the
correction stage.

A finite `max_slowness` is also required. For QSEIS2025, its default is
None, so `correct_zero_freq=True` alone does not apply the correction.
The loader warns, but still requires the static file.
Select a justified slowness bound in custom settings or assign
`config.max_slowness` after parsing.

The correction interval is derived from earliest P time, maximum
source–receiver distance, source duration and STF length, then clipped to
the output window. Correction requires at least two interval samples.
It differentiates stress, corrects the stress-rate integral toward the
static tensor and integrates back.

Without a static tensor, a finite slowness bound may replace the late tail
with a post-cutoff mean. Inspect the final time series and provide enough
window length; the setting is not a simple late-time zeroing switch.

## Scientific checks

Compare multiple sampling intervals, source/grid spacings and integration
settings for your geometry. Check for aliasing, a sufficient time window,
finite stress tensors and stable baselines. A static match after explicit
correction is not independent evidence of dynamic accuracy.

The local documentation validation exercises the static tutorial.
These dynamic recipes describe the current code and have not been
claimed as new end-to-end dynamic validation.
