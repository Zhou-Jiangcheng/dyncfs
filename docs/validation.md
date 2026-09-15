# Local validation record

Validation date: **2026-09-14**. This record describes the local checkout.

## Static tutorial

Executed from the repository root:

```powershell
conda run -n pygrnwang python docs/examples/quickstart.py
```

| Item | Observed value |
|---|---|
| Platform | Windows 11, x86-64 |
| Python | 3.12.12 |
| DynCFS source version | 3.0.0 |
| Companion pygrnwang source version | 3.0.0 |
| NumPy / SciPy / pandas | 1.26.4 / 1.17.0 / 2.1.4 |
| Matplotlib | 3.10.8 |
| Backends | EDGRN2 + EDCMP2 |
| Source patches / receivers | 1 / 3 |
| Stress tensor shape | (3, 6) |
| Resolved normal/shear/CFS array | (3, 3) |
| Calculation and export elapsed time | Approximately 5.9 s |

The environment loads the adjacent pygrnwang source checkout. Its installed
distribution metadata still reports 2.1.5, while the imported source reports
3.0.0. This run validates those local sources and available native binaries;
it is not a clean PyPI installation or wheel compatibility test.

Checks passed: expected array shapes, finite nonzero results, and the
Coulomb identity with friction 0.4 and no pore term.

| Distance (km) | Normal stress (Pa) | Shear stress (Pa) | CFS (Pa) |
|---|---|---|---|
| 30 | 122.858813 | -92.986806 | -43.843281 |
| 60 | 13.145526 | -18.439058 | -13.180847 |
| 90 | 4.702104 | -6.278490 | -4.397648 |

Full-precision output and environment details are in the locally generated
`docs/_build/quickstart/summary.json`. The selected figure is copied to
`docs/_static/quickstart.png` for documentation builds.

## Earthquake examples

The [Wenchuan](cases/wenchuan.md) and [Ludian](cases/ludian.md) examples
completed all three stages locally. The script
`docs/examples/audit_case_results.py` checks input and core-source hashes,
output dimensions, finite values and the saved Coulomb-stress identities.
For the Wenchuan half-space library it also requires every stress value
beyond 10 km to stay below 10 μ/r³. These checks do not compare against
earlier numerical results, which are not available. Fourteen independently
repeated native QSEIS2025 jobs produced 84 bitwise-identical
Green's-function binary files.

## Documentation checks

The API/CLI checks passed for 49 documented signatures and 10 CLI flags.
The HTML build completed for 31 source pages with `-W --keep-going` and
no warnings. It checks internal references and includes executable example sources.
All 33 generated documentation pages (including search and index) were checked
for local file/anchor targets: 2,378 link and asset references resolved.
The theme's unrendered `_static/webpack-macros.html` template is an asset,
so it is excluded from page parsing. Referenced assets are still checked.
All six public example scripts and ten embedded Python snippets compiled.
The tutorial and regenerated case figures were visually inspected.
An earlier local HTTP homepage check returned 200 with the expected content.
Browser page inspection could not be completed because the browser connection
failed; no browser layout or interaction check is claimed.
See [development](development.md) for repeatable commands.

## Scope

The small tutorial checks a static workflow; the separate earthquake
record covers the local case reruns, including QSEIS2025 dynamics.
These checks do not establish numerical convergence, validate QSSP2020,
compare all optimized modes or test multi-node execution.
External reference URLs and clean installations on other platforms were
not part of this validation.
