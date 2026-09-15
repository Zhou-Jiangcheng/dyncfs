# Troubleshooting

| Symptom | Check or action |
|---|---|
| INI section/key error | Use a complete template; all general sections are parsed even for static work |
| Input file not found | Resolve paths against the process directory; use absolute paths without surrounding quotes |
| Missing solver executable | Check the active environment and the matching pygrnwang installation |
| Numerical DLL error on Windows | Activate Conda or use `conda run -n ...`; do not call an inactive interpreter directly |
| Multiprocessing bootstrap RuntimeError | Add a main guard and run from a script |
| Fixed-depth CLI does no work | The configured depth must be greater than zero |
| Missing static tensor during dynamics | Compute matching static results first when `correct_zero_freq=True` |
| Correction enabled but not applied | Set a justified finite `max_slowness` and a sufficient correction window |
| Full-tensor stress appears scaled by a million | INI uses MPa; direct Python APIs use Pa |
| Static and dynamic amplitudes disagree | Compare local rigidity × area × slip against supplied moment, then examine STF and baseline settings |
| Unexpectedly missing source patches | Check CSV missing values, slip threshold and STF cutoff |
| Source concatenation or grid reshape failure | Check equal STF lengths, row counts and `[n_strike,n_dip]` shapes |
| Wrong or repeated dynamic receiver output | Check collisions in four-decimal per-point names and concurrent writes |
| Reused results after changing the model | Use a fresh library/output directory; cache signatures do not hash all native files |
| Excess memory use | Reduce source–receiver pairs, grid points, time samples or workers; final arrays remain fully assembled |
| Optimized CFS not found by a plotter | Load its `_os` or `_oop` filename explicitly and use an array-based plot |
| GIF import failure | Install the optional `imageio` package |
| Small tutorial refuses to run again | Select a new or empty `--output-dir` |

## Useful evidence for a bug report

Record the source revision, versions, platform, exact command, full error
trace, INI and a small input subset. Preserve native solver logs and
`green_lib_info.json`. Reproduce with one receiver or a small grid when
possible.

Documentation checks establish that the pages build and reference real
signatures. They do not establish numerical convergence or validate all
backend/receiver combinations.
