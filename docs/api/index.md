# API reference

Import entry points from their modules; the package root exposes its version
and does not re-export the full API.

```python
from dyncfs.configuration import CfsConfig
from dyncfs.cfs_static import compute_static_cfs
from dyncfs.cfs_dynamic import compute_dynamic_cfs_parallel
```

This guide documents **49 classes, methods and functions**, selected for
configuration, calculations, scientific projection, conversion and plotting.
Internal preparation, aggregation and cache helpers are not all included;
a name without an underscore is not automatically a documented entry point.

The [API inventory](public-api.json) is checked against local source during
every HTML build. The check verifies names, argument order and defaults,
without importing numerical modules. Explanations are maintained here
because some historical source docstrings have stale units or descriptions.

```{toctree}
:maxdepth: 1

configuration
static
dynamic
science
inputs
plotting
```
