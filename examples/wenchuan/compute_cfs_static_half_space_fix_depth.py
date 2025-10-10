from dyncfs.cfs_static import *

if __name__ == "__main__":
    config = CfsConfig()
    config.read_config("/home/zjc/python_works/dyncfs/test/wenchuan/wenchuan.ini")
    config.layered = False
    config.lam = 25.211679999999994 * 1e9
    config.mu = 31.126160000000002 * 1e9

    config.check_finished = False
    config.obs_delta_lat = 0.01
    config.obs_delta_lon = 0.01
    compute_static_cfs_fix_depth(
        config=config,
        obs_depth=15,
        optimal_type=0,
        receiver_mechanism=[223, 47, 131],
        obs_lat_range=config.obs_lat_range,
        obs_lon_range=config.obs_lon_range,
        obs_delta_lat=config.obs_delta_lat,
        obs_delta_lon=config.obs_delta_lon,
    )
