from dyncfs.cfs_dynamic import *

if __name__ == "__main__":
    config = CfsConfig()
    config.read_config("wenchuan.ini")
    create_dynamic_lib(config)
    compute_dynamic_cfs_parallel(config)
