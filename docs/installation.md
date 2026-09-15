# Installation

## Package and numerical dependencies

The local metadata declares Python **3.9+**, `pygrnwang>=3.0.0` and `matplotlib>=3.9.2`. NumPy, SciPy, pandas, ObsPy and tqdm are used through the numerical dependency stack. The GIF helper additionally imports `imageio`, which DynCFS does not declare; install it separately for that helper.

```bash
python -m pip install dyncfs
python -c "import dyncfs; print(dyncfs.__version__)"
dyncfs --help
```

Import and CLI help do not exercise the solvers. Use the [quickstart](quickstart.md) for a complete static calculation.

Native executables come from **pygrnwang**. A matching wheel can avoid compilation; a source build requires a working Fortran toolchain. Wheel availability depends on the release, platform and Python version.

## Work from source

Use a compatible pygrnwang 3.x checkout alongside DynCFS. From the directory containing both repositories:

```bash
conda create -n cfs -c conda-forge python=3.12 numpy scipy pandas obspy tqdm matplotlib
conda activate cfs
conda install -c conda-forge gfortran
python -m pip install -e ./pygrnwang
python -m pip install -e ./dyncfs
cd dyncfs
```

Obtain missing checkouts from the [DynCFS repository](https://github.com/Zhou-Jiangcheng/dyncfs) and [pygrnwang repository](https://github.com/Zhou-Jiangcheng/pygrnwang).

Follow the installation instructions in the matching pygrnwang checkout for compiler details. Its build hooks supply the executables. The current DynCFS setup hook itself creates an executable directory and checks for Java; it does not compile the solvers.

Use **absolute paths** for calculation inputs and outputs. Native solvers can change the working directory. Short paths without spaces avoid historical native filename limits; the bundled INI comments recommend fewer than 80 characters for input/output roots.

## Windows and Conda

Keep the environment activated while running numerical code. For automation:

```powershell
conda run -n cfs python docs/examples/quickstart.py
conda run -n cfs python -m dyncfs.main --help
```

Do not call an inactive Conda environment's `python.exe` directly. Activation supplies numerical-library and compiler DLL directories.

Check that `edgrn2.exe`, `edcmp2.exe` and the required dynamic solver are available in the active environment. On Unix, the companion bulk runners use `.bin` names. If a wheel imports but bulk creation cannot find a solver, inspect pygrnwang's executable layout and use its documented editable installation.

## Travel times and parallel execution

The companion pygrnwang 3.0 source uses Java subprocesses when the JDK and TauP JAR are available, with an ObsPy fallback for general travel-time queries. Keep both `java` and `javac` on PATH if choosing that backend.

DynCFS's high-level dynamic execution uses local multiprocessing. Its CLI has no multi-node MPI option; `mpi4py` is not required for this workflow.

## Build only the documentation

The HTML build needs no installed numerical package. It checks API signatures by parsing local source without importing it. See [development](development.md) for build and preview commands.
