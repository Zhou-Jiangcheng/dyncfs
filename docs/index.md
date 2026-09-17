# Coulomb stress, from rupture to receiver

```{raw} html
<p class="hero-kicker">DynCFS · computational seismology</p>
<p class="hero-copy">Compute static and dynamic Coulomb failure stress changes from finite-fault models, with explicit source, receiver, time and stress conventions.</p>
```

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Run a small calculation
:link: quickstart
:link-type: doc

Prepare one source patch and three receivers, build a static library and plot CFS.
:::

:::{grid-item-card} Configure your model
:link: configuration
:link-type: doc

Understand the INI sections, receiver modes, library coverage and dynamic backends.
:::

:::{grid-item-card} 中文入门
:link: zh/index
:link-type: doc

安装程序、运行小算例，并核对应力单位和结果目录。
:::
::::

## A complete workflow

1. **Describe the rupture.** Supply patch geometry, slip, seismic moment and source time functions.
2. **Define the receivers.** Use observation faults or a geographic grid at a fixed depth.
3. **Build a library.** Match the Earth model, depth and distance coverage to the calculation.
4. **Resolve the stress.** Use a fixed mechanism, an optimized rake, or optimally oriented planes.
5. **Inspect the result.** Check tensor components, units, sampling and static–dynamic consistency.

DynCFS calls [pygrnwang](https://github.com/Zhou-Jiangcheng/pygrnwang) for Green's-function preparation and synthesis. The current source uses **EDGRN2/EDCMP2** for static stress, **QSEIS2025** for layered dynamic calculations and **QSSP2020** for spherical dynamic calculations.

These pages describe the **3.0.0** source tree on the `main` branch. Python support is declared as 3.9 or newer; the documentation build uses Python 3.12. Read the [scientific conventions](conventions.md) before interpreting stress signs or comparing outputs.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
zh/index
```

```{toctree}
:maxdepth: 2
:caption: User guide

configuration
input-files
conventions
guides/index
cli
```

```{toctree}
:maxdepth: 2
:caption: Earthquake examples

cases/wenchuan
cases/ludian
```

```{toctree}
:maxdepth: 2
:caption: Reference and development

api/index
validation
development
references
```
