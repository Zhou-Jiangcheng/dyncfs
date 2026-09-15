# Input files

The input directory contains `model.nd`, `source_plane<m>.csv` and
`obs_plane<n>.csv`. Integer IDs match `source_inds` and `obs_inds`.
CSV files contain numbers only, with **no header and no index column**.

## Source patches

One row describes the **center** of a subfault.

| Zero-based column | Quantity | Unit |
|---|---|---|
| 0, 1 | Latitude, longitude | Degrees; north/east positive |
| 2 | Depth | km, positive downward |
| 3, 4, 5 | Strike, dip, rake | Degrees |
| 6, 7 | Patch length along strike and dip | km |
| 8 | Final slip | m |
| 9 | Scalar seismic moment | N m |
| 10 onward | Source time function samples | Relative shape, or moment rate |

Selected source planes need the same number of STF columns. Pad shorter
functions with zeros. The reader drops rows containing missing values, so a
malformed row may disappear without an explicit schema error.

Dynamic synthesis resamples each STF to the CFS rate and rescales its
discrete integral to column 9. A zero-integral function is skipped.
Rupture delay is encoded by leading zero samples; the final CSV has no
separate rupture-time column.

Static synthesis uses `slip × patch area × local rigidity`. Keep moment
consistent with those values before comparing static and dynamic results.
The original STF amplitude is not an additional moment multiplier.

### Row order

For shape `[n_strike, n_dip]`, dip varies fastest:

```text
row = i_strike * n_dip + i_dip
```

The row count must equal the product of these dimensions. Calculations
often operate directly on rows, but plotting reshapes them, so a wrong
shape may only become apparent during visualization.

The low-level `write_source_plane_csv` converter accepts the opposite
grid traversal: it reshapes `(n_dip, n_strike)`, transposes, then flattens.
Do not transpose its output again.

## Observation faults

Every row contains six columns:

```text
latitude, longitude, depth, strike, dip, rake
```

Coordinates follow the source conventions. Keep six columns for all modes:
mode 0 uses the full mechanism; mode 1 uses strike/dip; mode 2 determines
orientation from stress.

Rows follow `obs_shapes`, with dip varying fastest. Observation IDs need
not match source IDs. Fixed-depth functions construct their own geographic
grid, whereas `run_all_*` also calculates the selected observation faults.

## Earth model

The bundled `.nd` files use:

```text
depth_km  vp_km_per_s  vs_km_per_s  density_g_per_cm3  Qp  Qs
```

Single-word interface labels such as `mantle` appear in the supplied
format. Repeated depths describe discontinuities. Preserve the full model
needed by the chosen solver and travel-time calculation;
`earth_model_layer_num` controls selection during preprocessing.

```{literalinclude} ../examples/wenchuan/input/model.nd
:language: text
:lines: 1-10
```

The utility `read_nd(path, with_Q=True)` expects six numeric columns.
Its default `with_Q=False` expects four; it does not remove Q columns
automatically.

## Convert a finite-fault model

```python
from pathlib import Path
from dyncfs.convert_input_format import convert_fsp2source_csvs

input_dir = Path("my-case/input").resolve()
input_dir.mkdir(parents=True, exist_ok=True)
shapes = convert_fsp2source_csvs(
    path_fsp="complete_inversion.fsp",
    path_input_dir=str(input_dir),
    sampling_interval_stf=0.5,
    rise_ratio=0.5,
)
print(shapes)  # Copy into source_shapes.
```

FSP conversion requires LAT, LON, Z, SLIP, TRUP, RISE and SF_MOMENT.
RAKE may fall back to the mechanism header. `rise_ratio` splits RISE
between a triangular STF's rising and falling parts; its range is [0, 1].

For USGS `basic_inversion.param`, use `convert_usgs_basic2source_csvs`.
It converts cm to m and dyne cm to N m. Both converters write source
CSVs and return segment shapes. Prepare receivers and the Earth model
separately.

The [Wenchuan](cases/wenchuan.md) and [Ludian](cases/ludian.md) examples
describe the supplied case files and their location-specific paths.
