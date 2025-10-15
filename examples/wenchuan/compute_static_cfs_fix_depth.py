from dyncfs.cfs_static import *

if __name__ == "__main__":
    config = CfsConfig()
    config.read_config("wenchuan.ini")
    config.source_inds = [1,2,3,4,5]
    config.source_shapes = [[[22,9],[6,9],[8,9],[62,9],[17,6]]]
    config.static_dist_range = [0, 1000]
    config.static_delta_dist = 1
    config.static_source_depth_range = [0, 30]
    config.static_source_delta_depth = 1
    config.static_obs_depth_list = [_ for _ in range(31)]
    config.check_finished = False

    create_static_lib(config)
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
