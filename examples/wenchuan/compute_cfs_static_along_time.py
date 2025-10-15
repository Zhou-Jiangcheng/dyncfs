from dyncfs.cfs_static import *


def cal_cfs_static_one_time_point(nt, config: CfsConfig):
    config = config.copy()
    config.cut_stf = nt
    config.check_finished = False
    config.path_output_results_static = str(
        os.path.join(config.path_output_results_static, "time", "%d" % nt)
    )
    os.makedirs(config.path_output_results_static, exist_ok=True)
    compute_static_cfs(config)


if __name__ == "__main__":
    config = CfsConfig()
    config.read_config("wenchuan.ini")
    config.processes_num = 4

    config.optimal_type = 0
    config.source_inds = [4]
    config.source_shapes = [[62, 9]]
    config.obs_inds = [5]
    config.obs_shapes = [[17, 6]]
    config.slip_thresh = 1
    config.check_finished = False
    config.static_dist_range = [0, 800]
    create_static_lib(config)

    nt_list = [_ for _ in range(0, 80, 2)]
    for nt in nt_list:
        print(nt)
        cal_cfs_static_one_time_point(nt, config)
