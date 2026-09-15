# Quickstart: static Coulomb stress

This calculation uses **one 1 km × 1 km source patch with 1 m slip** and receivers at **30, 60 and 90 km** epicentral distance. The patch is 10 km deep; receivers are 5 km deep. Both mechanisms are strike 30°, dip 45°, rake 90°. Friction is 0.4 and `B_pore=0`.

The script copies `examples/wenchuan/input/model.nd` and uses its first 24 numeric rows for the static backend. This illustrative geometry is not a Wenchuan reproduction. The coarse library checks the workflow; it is not a convergence study.

## 1. Run the calculation

Follow [installation](installation.md), then run from the DynCFS repository root:

```bash
python docs/examples/quickstart.py
```

On Windows with Conda:

```powershell
conda run -n cfs python docs/examples/quickstart.py
```

Files are written below `docs/_build/quickstart/`. The script requires a new or empty output directory. To repeat:

```bash
python docs/examples/quickstart.py --output-dir docs/_build/quickstart-repeat
```

## 2. Inspect the inputs and library

The script writes source/receiver CSV files and an INI with absolute paths, then calls `create_static_lib` and `compute_static_cfs`.

The library has source depths **10 and 11 km**, receiver depth **5 km**, and distances **1–121 km at 10 km spacing**. EDGRN needs at least two source depths. Queries stay inside the library range.

To prepare inputs and run the equivalent CLI steps:

```bash
python docs/examples/quickstart.py --prepare-only --output-dir docs/_build/prepared
python -m dyncfs.main --config docs/_build/prepared/quickstart.ini --create-static-lib --compute-static-cfs
```

The CLI writes numerical results; plotting and the summary are steps in the complete Python script.

## 3. Read the results

| File below the output directory | Meaning |
|---|---|
| `quickstart.ini` | Configuration with resolved paths |
| `input/` | Source, receivers and copied Earth model |
| `grn_s/` | Static library and metadata |
| `results/static/stress_tensor_plane1.npy` | Shape `(3, 6)`, Pa, NED order |
| `results/static/cfs_static_plane1.csv` | Three rows, one column, Pa |
| `static_cfs.png` | Normal, shear and Coulomb stress in kPa |
| `summary.json` | Environment, timing, shapes and checks |

```{figure} _static/quickstart.png
:alt: Normal, shear and Coulomb stress changes at three distances for the small static example.
:width: 100%

Local quickstart output. CSV values remain in pascals; this figure displays kilopascals.
```

The script checks dimensions, finite nonzero values and
`CFS = shear_stress + 0.4 * normal_stress`.
See the [validation record](validation.md) for what was run.

```python
from pathlib import Path
import numpy as np

result = Path("docs/_build/quickstart/results/static")
stress = np.load(result / "stress_tensor_plane1.npy")
cfs = np.loadtxt(result / "cfs_static_plane1.csv", delimiter=",", ndmin=1)
print(stress.shape)  # (3, 6)
print(cfs / 1e6)     # MPa
```

## 4. Continue to dynamic stress

Follow the [dynamic workflow](guides/dynamic.md) and provide a physically appropriate STF. Dynamic synthesis scales the STF to moment; static synthesis uses area and slip. Verify their consistency before comparing them.

The introductory INI includes time fields required by the parser, but its dynamic settings are not a validated dynamic tutorial.

## Complete script and configuration

```{literalinclude} examples/quickstart.py
:language: python
:linenos:
```

```{literalinclude} examples/quickstart.ini
:language: ini
```
