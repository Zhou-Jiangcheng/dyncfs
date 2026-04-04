# Introduction

This Python package serves as the frontend for computing static and dynamic Coulomb Failure Stress changes in layerd media. The backend relies on programs developed by Rongjiang Wang and modified by Jiangcheng Zhou for calculating synthetic seismograms, specifically EDGRN/EDCMP, [QSEIS_2006_STRESS](https://github.com/Zhou-Jiangcheng/QSEIS_2006_STRESS), and QSSP. The generation and retrieval of Green's function libraries are handled by the [pygrnwang](https://github.com/Zhou-Jiangcheng/pygrnwang) package, while travel-time calculations and geographic coordinate transformations are implemented using [obspy](https://github.com/obspy/obspy).

# Installation

1. For user mode

```
pip install dyncfs
```

2. For developer mode.

```
conda create -n cfs python=3.13
conda activate cfs
conda install gfortran obspy tqdm -c conda-forge
conda install geographiclib mpi4py -c conda-forge # optional
git clone https://github.com/Zhou-Jiangcheng/pygrnwang
cd pygrnwang
pip install -e .
cd ..
git clone https://github.com/Zhou-Jiangcheng/dyncfs.git
cd dyncfs
pip install -e .
```

# Usage

Fill in all parameters in the .ini file, and prepare the input files as described in `example.ini`, including `source_plane[m].csv` and `obs_plane[n].csv` under the input directory, as well as `model.nd` representing the Earth model.

1. Command-line usage

```
dyncfs --help
```

```
usage: dyncfs [-h] --config CONFIG [--create-static-lib] [--compute-static-cfs] [--compute-static-cfs-fix-depth] [--run-static] [--create-dynamic-lib] [--compute-dynamic-cfs] [--compute-dynamic-cfs-fix-depth] [--run-dynamic] [--run-all]

dyncfs command line tool

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the configuration file
  --create-static-lib   Create static stress library
  --compute-static-cfs  Compute static dCFS on obs faults
  --compute-static-cfs-fix-depth
                        Compute static dCFS at fixed depth
  --run-static          Create static stress library and Compute static dCFS on obs faults and Compute static dCFS at fixed depth
  --create-dynamic-lib  Create dynamic stress library
  --compute-dynamic-cfs
                        Compute dynamic dCFS on obs faults
  --compute-dynamic-cfs-fix-depth
                        Compute dynamic dCFS at fixed depth
  --run-dynamic         Create dynamic stress library and Compute dynamic dCFS on obs faults and Compute dynamic dCFS at fixed depth
  --run-all             Create static and dynamic stress library and Compute static and dynamic dCFS
```

2. Import and use classes and functions in .py files

```
from dyncfs.cfs_static import *

if __name__ == "__main__":
    config = CfsConfig()
    config.read_config("example.ini")
    create_static_lib(config)
    compute_static_cfs_parallel(config)
```

```
from dyncfs.cfs_dynamic import *

if __name__ == "__main__":
    config = CfsConfig()
    config.read_config("example.ini")
    create_dynamic_lib(config)
    compute_dynamic_cfs_parallel(config)
```

# References:
Wang, R. (1999). A simple orthonormalization method for stable and efficient computation of Green’s functions.  *Bulletin of the Seismological Society of America* ,  *89* (3), 733–741. https://doi.org/10.1785/BSSA0890030733

Wang, R. (2003). Computation of deformation induced by earthquakes in a multi-layered elastic crust—FORTRAN programs EDGRN/EDCMP. Computers & Geosciences, 29(2), 195–207. https://doi.org/10.1016/S0098-3004(02)00111-5

Wang, R., & Wang, H. (2007). A fast converging and anti-aliasing algorithm for green’s functions in terms of spherical or cylindrical harmonics. Geophysical Journal International, 170(1), 239–248. https://doi.org/10.1111/j.1365-246X.2007.03385.x

Wang, R., Heimann, S., Zhang, Y., Wang, H., & Dahm, T. (2017). Complete synthetic seismograms based on a spherical self-gravitating earth model with an atmosphere–ocean–mantle–core structure. Geophysical Journal International, 210(3), 1739–1764. https://doi.org/10.1093/gji/ggx259

Zhou, J., Wang, R., & Zhang, Y. (2026). DynCFS: a program for modeling dynamic coulomb failure stress changes in layered elastic media. Geophysical Journal International, ggaf534. https://doi.org/10.1093/gji/ggaf534

# Acknowledgment
Thanks to Haitao Qin from Tongji University for reporting a bug, which has been fixed in the latest version.