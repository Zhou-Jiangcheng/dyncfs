# Stress projection API

Direct calls use Pa, NED tensor order `[NN,NE,ND,EE,ED,DD]` and degrees. These functions operate on supplied stress and do not need a Green's-function library. See [scientific conventions](../conventions.md).

```{py:function} dyncfs.cfs_static.cal_coulomb_failure_stress(norm_stress, shear_stress, mu_f=0.4)
`norm_stress` is tension-positive normal stress in Pa; `shear_stress` is signed shear in Pa; `mu_f` is dimensionless effective friction. Return `shear_stress + mu_f * norm_stress`, preserving NumPy broadcasting.
```

```{py:function} dyncfs.cfs_static.cal_coulomb_failure_stress_poroelasticity(norm_stress, shear_stress, mean_stress, mu_f=0.6, B_pore=0)
Normal, shear and mean stresses are in Pa; `mean_stress` is tensor trace/3. Dimensionless `mu_f` and `B_pore` give `shear + mu_f*(normal-B_pore*mean)`. Returns a scalar or broadcast array. Note its friction default is 0.6; the single-point wrappers default to 0.4.
```

```{py:function} dyncfs.cfs_static.cal_cfs_static_single_point_fix_fm(obs_fm, stress, mu_f=0.4, B_pore=0.0)
`obs_fm` is `[strike,dip,rake]`; `stress` is a six-entry perturbation tensor. Return `(n,d,sigma,tau,cfs)`: two `(3,)` NED unit vectors and three scalar stresses in Pa. `mu_f` and `B_pore` select the Coulomb formula.
```

```{py:function} dyncfs.cfs_static.cal_cfs_static_single_point_opt_rake(obs_strike, obs_dip, stress, tectonic_stress, mu_f=0.4, B_pore=0.0)
`obs_strike`/`obs_dip` fix the plane. `stress` and `tectonic_stress` are six-entry NED tensors in Pa. Optimize rake using total traction, then resolve perturbation stress. Return `(n,d,sigma,tau,cfs,rake)`, with vectors `(3,)`, stresses in Pa and rake in degrees.
```

```{py:function} dyncfs.cfs_static.cal_cfs_static_single_point_opt_plane(stress, tectonic_stress_type, tectonic_stress, mu_f=0.4, B_pore=0.0)
`stress` is the perturbation tensor in Pa. Type 1 uses a full `tectonic_stress` tensor in Pa; type 2 uses ordered principal-axis azimuth/plunge pairs in degrees. Invalid types raise ValueError. Return `([n1,d1,sigma1,tau1], [n2,d2,sigma2,tau2], cfs)`. The one CFS value is evaluated on plane 2. Units follow the other projection functions.
```

```{py:function} dyncfs.cfs_dynamic.cal_stress_vector_ned_dynamic(stress_ned, n)
`stress_ned` has shape `(T,6)` in Pa and `n` is a three-component fixed NED normal. Return traction vectors `(T,3)` in Pa. This helper does no synthesis or disk I/O.
```

```{py:function} dyncfs.signal_process.correct_zero_frequency(data, srate, A0, f_c, tc1, tc2, ratio_interp=0)
`data` is a 1D series, normally stress rate in Pa/s; `srate` is Hz; `A0` is the desired integral (Pa for stress rate). `f_c` counts frequency bins, not Hz. `tc1`/`tc2` are start/inclusive and stop/exclusive sample indices clipped to the data. Positive `ratio_interp` resamples before FFT; 0 disables it. Return a same-length corrected array without mutating input. Outside the corrected window the returned series is zero; a window shorter than two samples instead returns an unchanged copy.
```

```{py:function} dyncfs.utils.static_stress_ned2enz(stress_ned)
Convert a stress array `(6,)` or `(N,6)` from NED to ENU component order, exchanging horizontal axes and reversing vertical-shear signs. Return a new same-shaped array with unchanged units.
```
