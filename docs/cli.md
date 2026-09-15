# Command-line interface

Use the installed entry point or its module equivalent:

```bash
dyncfs --help
python -m dyncfs.main --help
dyncfs --config /absolute/path/case.ini --create-static-lib --compute-static-cfs
```

`--config` is required for calculations. Numerical settings come from the INI.

| Flag | Action |
|---|---|
| `--create-static-lib` | Build EDGRN2/EDCMP2 stress library |
| `--compute-static-cfs` | Static results on observation faults |
| `--compute-static-cfs-fix-depth` | Static results on the depth grid |
| `--run-static` | Build library, compute faults, then enabled depth grid |
| `--create-dynamic-lib` | Build QSEIS2025 or QSSP2020 stress library |
| `--compute-dynamic-cfs` | Dynamic results on observation faults |
| `--compute-dynamic-cfs-fix-depth` | Dynamic results on the depth grid |
| `--run-dynamic` | Build library, compute faults, then enabled depth grid |
| `--run-all` | Complete static workflow, then complete dynamic workflow |

Flags run in this table's order, regardless of command-line order. They are not mutually exclusive: combining a `--run-*` flag with its constituent flags repeats work.

Dynamic computation is sequential for `processes_num=1`, otherwise it uses spawned workers. Fixed-depth CLI flags and complete workflows skip the grid for `fixed_obs_depth<=0`.

## Reuse a prepared library

```bash
dyncfs --config /absolute/path/case.ini --compute-dynamic-cfs
```

Verify model, coverage, backend, sampling and output observables before reuse.
`check_finished` is an INI field, not a CLI flag.

Zero-frequency correction needs static tensors for the same receiver set.
`--run-dynamic` does not create those tensors; `--run-all` orders static work first.

Using only `--config` parses the INI and creates directories without calculating.
It is not comprehensive input validation. `python -m dyncfs` is not an entry point.
