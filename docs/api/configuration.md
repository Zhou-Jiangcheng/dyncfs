# Configuration API

Import the class from `dyncfs.configuration`. General units and INI keys are in the [configuration reference](../configuration.md).

```{py:class} dyncfs.configuration.CfsConfig()
Create an uninitialized configuration. Call `read_config` before a calculation; most attributes initially contain None.
```

```{py:method} dyncfs.configuration.CfsConfig.read_config(path_conf)
Read the INI at `path_conf` (str/path). Creates library and result directories, parses settings, converts degree distances to km and full tectonic stress from MPa to Pa, then derives backend fields. Returns None. Missing files/sections and invalid values surface as configuration or conversion errors; this is not a full geometry validator.
```

```{py:method} dyncfs.configuration.CfsConfig.fixed_obs_depth_enabled()
Return True only when `fixed_obs_depth` is not None and is greater than zero. Used by CLI and complete workflows.
```

```{py:method} dyncfs.configuration.CfsConfig.set_default()
Assign the solver defaults listed in the configuration reference. Uses `use_spherical` and may read `path_nd` to derive QSSP slowness; call only after basic settings are available. Returns None.
```

```{py:method} dyncfs.configuration.CfsConfig.get_obs_region()
Read selected source and receiver CSVs. Set geographic ranges around the receivers, pad by the library distance step converted to degrees, and update source/receiver reference points to a geographic centroid. Return `(reference_point, obs_points)`, with coordinates in degrees/km. Requires nonempty selected sources and observations; changes the configuration in place.
```

```{py:method} dyncfs.configuration.CfsConfig.copy(deep=True)
Return a deep copy by default. With `deep=False`, lists and other mutable objects are shared, while NumPy array attributes are copied. Use a deep copy before changing nested shapes or ranges.
```
