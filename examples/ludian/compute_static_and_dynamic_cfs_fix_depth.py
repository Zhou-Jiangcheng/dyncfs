from dyncfs.cfs_static import *
from dyncfs.cfs_dynamic import *

if __name__ == "__main__":
    config = CfsConfig()
    config.read_config("ludian.ini")
    create_static_lib(config)
    compute_static_cfs_fix_depth(config)
    create_dynamic_lib(config)
    compute_dynamic_cfs_fix_depth_parallel(config)
