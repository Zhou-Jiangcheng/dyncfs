# Scientific conventions

## Stress coordinates and units

DynCFS stores stress in **north, east, down (NED)** order:

```text
[σNN, σNE, σND, σEE, σED, σDD]
```

The symmetric matrix is

$$
\boldsymbol{\sigma} =
\begin{bmatrix}
\sigma_{NN} & \sigma_{NE} & \sigma_{ND} \\
\sigma_{NE} & \sigma_{EE} & \sigma_{ED} \\
\sigma_{ND} & \sigma_{ED} & \sigma_{DD}
\end{bmatrix}.
$$

Stress arrays and CSV files use **Pa**. Divide by $10^6$ for MPa.
The INI full `tectonic_stress` uses **MPa**; `read_config` converts
it to Pa. Direct Python stress functions expect **Pa** and do not
perform that conversion.

Pygrnwang's rotated stress uses east, north, up:
`[EE, EN, EU, NN, NU, UU]`. DynCFS exchanges horizontal axes and
changes vertical shear signs. The low-level dynamic `static_stress`
argument for correction is **ENU**; high-level workflows convert saved
NED static tensors automatically.

## Traction and Coulomb stress

Normal stress is **tension positive**. For unit normal $\mathbf n$ and
unit slip direction $\mathbf d$:

$$
\Delta\sigma_n = \mathbf n^T\Delta\boldsymbol\sigma\mathbf n,
\qquad
\Delta\tau = \mathbf d^T\Delta\boldsymbol\sigma\mathbf n.
$$

Shear is signed along the selected slip direction. For `B_pore=0`:

$$
\Delta CFS = \Delta\tau + \mu_f\Delta\sigma_n.
$$

For nonzero `B_pore`:

$$
\Delta CFS = \Delta\tau +
\mu_f(\Delta\sigma_n-B_{pore}\Delta\sigma_m),
\qquad
\Delta\sigma_m=\mathrm{tr}(\Delta\boldsymbol\sigma)/3.
$$

Positive CFS increases failure tendency under this receiver and sign
convention. With `B_pore=0`, `mu_f` may represent effective friction.
For explicit pore pressure, choose the two coefficients consistently.

## Receiver modes

| `optimal_type` | Orientation | Tectonic input |
|---|---|---|
| 0 | Fixed strike, dip, rake | None |
| 1 | Fixed strike/dip, rake follows total shear traction | Full tensor, type 1 |
| 2 | Two optimally oriented conjugate planes | Full tensor (type 1) or axes (type 2) |

With a full tectonic tensor, orientation comes from **tectonic plus
earthquake-induced stress**, while reported changes are resolved from
the **earthquake perturbation**. The background tensor is not added
directly to the reported CFS.

Type 2 input is
`[azimuth1, plunge1, azimuth2, plunge2, azimuth3, plunge3]` in degrees,
ordered from smallest to largest principal stress, tension positive.
Supply mutually orthogonal directions. No magnitudes are supplied.

In optimized dynamic modes, orientation may change at every sample.
OOP routines return two normal/shear sets and one CFS array.
The implementation evaluates the returned CFS on the second conjugate
plane; there are not two separately saved CFS histories.

## Source normalization

Static synthesis scales with patch area and slip; dynamic synthesis
normalizes the STF to scalar moment. Use consistent values for
$M_0=\mu A D$, with area in m² and slip in m.

The STF and CFS intervals are separate, both in seconds. STF normalization
uses the discrete sum divided by sampling rate. Leading zeros encode
rupture delay; a zero-integral STF contributes no dynamic stress.

`cut_stf=k` retains indices `0 ... k-1`, zeros later samples, and
rescales slip and moment by the retained integral fraction.
A positive `slip_thresh` zeros smaller slip and moment values without
deleting rows.

## Time and dynamic baselines

The window is `(sampling_num - 1) * sampling_interval_cfs`.
CSV columns contain samples, without a time column. For zero-reduction
QSEIS2025, use `arange(sampling_num) * sampling_interval_cfs` relative
to the source/STF origin. Recheck alignment after changing backend
time-reduction settings; index zero is not automatically the P arrival.

Synthesis removes each source's mean pre-P baseline, convolves its STF,
and retains the requested samples. With `max_slowness`, it may replace
the late tail with a mean or correct zero frequency using static stress.
It does **not** simply zero every late sample, despite historical
docstring wording. See [dynamic calculations](guides/dynamic.md).
