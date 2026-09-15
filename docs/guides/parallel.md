# Parallel execution and reuse

## Spawned worker processes

DynCFS dynamic parallel routines use a local **spawn** context on Windows,
Linux and macOS. Protect the calling script:

```python
from dyncfs.configuration import CfsConfig
from dyncfs.cfs_dynamic import compute_dynamic_cfs_parallel

if __name__ == "__main__":
    config = CfsConfig()
    config.read_config("case.ini")
    compute_dynamic_cfs_parallel(config)
```

Use a positive `processes_num`, bounded by available CPU and memory.
Worker count is capped at the number of jobs. Workers are reused across
chunks; `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` are set to 1 while
workers start, then their original values are restored in the parent.

This does not retroactively resize numerical thread pools already loaded
in the parent. Avoid concurrent independent runs writing to the same
library or result directory.

The CLI selects sequential synthesis for 1 process. Calling an explicitly
parallel Python function still uses the parallel path even with a count of 1.
Static library parallelism is implemented by the companion backend runners.

## Cache behavior

With `check_finished=True`, per-point dynamic stress can be reused when
both its `*_stress_ned.npy` and `*_stress_ned.json` exist and the signature
matches. It includes:

- Absolute Green's-function root and hashed library metadata.
- Source array contents and shape.
- Receiver position, STF rate, backend and slowness bound.
- Static tensor supplied for correction.

Receiver mechanism, friction and pore coefficient are not part of the
stress-tensor signature: the cached tensor can be projected again for a
different mechanism. CFS and orientation results are recomputed.

The signature does not hash every native library file or the implementation.
Replacing binary tables in place without updating metadata can leave a
matching cache. Use a fresh output root when changing numerical inputs
or the backend implementation.

Per-point names round latitude, longitude and depth to **four decimals**.
Distinct receivers with identical rounded names collide; repeated
positions with different mechanisms can also overwrite projection files.
Avoid such duplicates within a batch, particularly in parallel jobs.

## Intermediate storage and memory

Parallel preparation writes a shared `source_array.npy` and job lists
below `grn_d/results_each/`. The pickled job lists are internal files,
not portable input formats.

Final dynamic CSV assembly allocates complete arrays for all receivers
and samples. Increase receiver count and time-window length gradually;
more workers do not remove the memory cost of final assembly.
