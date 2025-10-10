import os
import platform
import copy
from typing import Union

if platform.system() == "Windows":
    platform_exec = "exe"
else:
    platform_exec = "bin"
import configparser
import ast

import numpy as np
import pandas as pd

from .utils import read_nd, bool2int
from .geo import d2km, geographic_centroid


class CfsConfig(object):
    def __init__(self):
        # set by user
        self.path_input: str = None  # type: ignore
        self.path_output: str = None  # type: ignore

        self.optimal_type: int = None  # type: ignore
        self.calc_rup2unrup: bool = None  # type: ignore
        self.rup_velocity: float = None  # type: ignore
        self.hypocentre: list[float] = None  # type: ignore
        self.correct_zero_freq: bool = None  # type: ignore
        self.tectonic_stress_type: int = None  # type: ignore
        self.tectonic_stress: Union[list[float], np.ndarray] = None  # type: ignore
        self.mu_f: float = None  # type: ignore
        self.B_pore: float = None  # type: ignore
        self.source_inds: list[int] = None  # type: ignore
        self.source_shapes: list[list[int]] = None  # type: ignore
        self.source_ref: list[int] = None  # type: ignore
        self.obs_inds: list[int] = None  # type: ignore
        self.obs_shapes: list[list[int]] = None  # type: ignore
        self.obs_ref: list[int] = None  # type: ignore
        self.earth_model_layer_num: int = None  # type: ignore
        self.use_spherical: bool = None  # type: ignore
        self.slip_thresh: float = None  # type: ignore
        self.cut_stf: int = None  # type: ignore

        self.fixed_obs_depth: float = None  # type: ignore
        self.obs_lat_range: list[float] = None  # type: ignore
        self.obs_lon_range: list[float] = None  # type: ignore
        self.obs_delta_lat: float = None  # type: ignore
        self.obs_delta_lon: float = None  # type: ignore

        self.grn_source_depth_range: list[float] = None  # type: ignore
        self.grn_delta_source_depth: float = None  # type: ignore
        self.grn_obs_depth_range: list[float] = None  # type: ignore
        self.grn_delta_obs_depth: float = None  # type: ignore
        self.grn_dist_unit: str = None  # type: ignore
        self.grn_dist_range: list[float] = None  # type: ignore
        self.grn_delta_dist: float = None  # type: ignore

        self.sampling_interval_stf: float = None  # type: ignore
        self.sampling_interval_cfs: float = None  # type: ignore
        self.sampling_num: int = None  # type: ignore
        self.max_frequency: float = None  # type: ignore

        self.processes_num: int = None  # type: ignore
        self.check_finished: bool = None  # type: ignore

        # path
        self.path_nd: str = None  # type: ignore
        self.path_green_staic: str = None  # type: ignore
        self.path_green_dynamic: str = None  # type: ignore
        self.path_output_results_static: str = None  # type: ignore
        self.path_output_results_dynamic: str = None  # type: ignore
        self.path_bin_edgrn: str = None  # type: ignore
        self.path_bin_edcmp: str = None  # type: ignore
        self.path_bin_qseis: str = None  # type: ignore
        self.path_bin_qssp: str = None  # type: ignore

        # edgrn2
        self.static_source_depth_range: Union[list[float],np.ndarray] = None  # type: ignore
        self.static_source_delta_depth: float = None  # type: ignore
        self.static_dist_range: Union[list[float],np.ndarray] = None  # type: ignore
        self.static_delta_dist: float = None  # type: ignore
        self.static_obs_depth_list: Union[list[float],np.ndarray] = None  # type: ignore
        # the following variables will be set by func self.set_default()
        # the value in qseis is the same as here
        self.wavenumber_sampling_rate: int = None  # type: ignore

        # edcmp2
        # the following variables will be set by func self.set_default()
        self.layered: bool = None  # type: ignore
        self.lam: float = None  # type: ignore
        self.mu: float = None  # type: ignore

        # qseis2025
        self.event_depth_list: list[float] = None  # type: ignore # also in qssp
        self.receiver_depth_list: list[float] = None  # type: ignore   # also in qssp
        self.time_window: float = None  # type: ignore  # also in qssp
        # the following variables will be set by func self.set_default()
        self.max_slowness: float = None  # type: ignore # also in qssp
        self.anti_alias: float = None  # type: ignore # also in qssp
        self.free_surface: bool = None  # type: ignore # also in qssp
        self.wavelet_duration: int = None  # type: ignore # also in qssp
        self.physical_dispersion: int = None  # type: ignore # also in qssp
        self.output_observables: list[int] = None  # type: ignore # also in qssp
        self.slowness_int_algorithm: int = None  # type: ignore
        self.eps_estimate_wavenumber: float = None  # type: ignore
        self.source_radius_ratio: float = None  # type: ignore
        self.slowness_window = None  # type: ignore
        self.time_reduction_velo: float = None  # type: ignore
        self.wavelet_type: int = None  # type: ignore
        self.flat_earth_transform: bool = None  # type: ignore

        # qssp2020
        # the following variables will be set by func self.set_default()
        self.time_reduction: float = None  # type: ignore
        self.spec_time_window: float = None  # type: ignore
        self.source_radius: float = None  # type: ignore
        self.turning_point_filter: int = None  # type: ignore
        self.turning_point_d1: float = None  # type: ignore
        self.turning_point_d2: float = None  # type: ignore
        self.gravity_fc: float = None  # type: ignore
        self.gravity_harmonic: int = None  # type: ignore
        self.cal_sph: int = None  # type: ignore
        self.cal_tor: int = None  # type: ignore
        self.min_harmonic: int = None  # type: ignore
        self.max_harmonic: int = None  # type: ignore

        self.default_config: bool = None  # type: ignore
        self.multiprocessing_flag = None  # type: ignore

    def read_config(self, path_conf):
        config = configparser.ConfigParser()
        config.read(path_conf)
        # [path]
        self.path_input = config["path"]["path_input"]
        self.path_output = config["path"]["path_output"]

        # other path
        self.path_nd = os.path.join(self.path_input, "model.nd")
        self.path_green_staic = os.path.join(self.path_output, "grn_s")
        os.makedirs(self.path_green_staic, exist_ok=True)
        if self.use_spherical:
            self.path_green_dynamic = os.path.join(self.path_output, "grn_d", "qssp")
        else:
            self.path_green_dynamic = os.path.join(self.path_output, "grn_d", "qseis")
        os.makedirs(self.path_green_dynamic, exist_ok=True)
        self.path_output_results_static = os.path.join(self.path_output, "results", "static")
        os.makedirs(self.path_output_results_static, exist_ok=True)
        self.path_output_results_dynamic = os.path.join(self.path_output, "results", "dynamic")
        os.makedirs(self.path_output_results_dynamic, exist_ok=True)
        self.path_bin_edgrn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "exec",
            "edgrn2.%s" % platform_exec,
        )
        self.path_bin_edcmp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "exec",
            "edcmp2.%s" % platform_exec,
        )
        self.path_bin_qseis = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "exec",
            "qseis2025.%s" % platform_exec,
        )
        self.path_bin_qssp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "exec",
            "qssp2020.%s" % platform_exec,
        )

        # [input_addition]
        self.optimal_type = int(config["input_addition"]["optimal_type"])
        if self.optimal_type == 1 or self.optimal_type == 2:
            self.tectonic_stress_type = int(
                config["input_addition"]["tectonic_stress_type"].strip())
            self.tectonic_stress = ast.literal_eval(
                config["input_addition"]["tectonic_stress"].strip()
            )
            if self.tectonic_stress_type == 1:
                for i in range(6):
                    self.tectonic_stress[i] = self.tectonic_stress[i] * 1e6
            if self.optimal_type == 2 and self.tectonic_stress_type == 2:
                raise ValueError('Can not optimal rake when full tectonic_stress is not provided!')
        self.mu_f = float(config["input_addition"]["mu_f"])
        self.B_pore = float(config["input_addition"]["B_pore"])
        # self.calc_rup2unrup = config["input_addition"].getboolean("calc_rup2unrup")
        # if self.calc_rup2unrup:
        #     self.rup_velocity = float(config["input_addition"]["rup_velocity"])
        # self.hypocentre = ast.literal_eval(
        #     config["input_addition"]["hypocentre"].strip()
        # )
        # if self.calc_rup2unrup:
        #     self.obs_inds = self.source_inds
        #     self.obs_shapes = self.source_shapes
        # else:
        #     self.obs_inds = ast.literal_eval(
        #         config["input_addition"]["obs_inds"].strip()
        #     )
        #     self.obs_shapes = ast.literal_eval(
        #         config["input_addition"]["obs_shapes"].strip()
        #     )
        self.source_inds = ast.literal_eval(
            config["input_addition"]["source_inds"].strip()
        )
        self.source_shapes = ast.literal_eval(
            config["input_addition"]["source_shapes"].strip()
        )
        self.source_ref = ast.literal_eval(config["input_addition"]["source_ref"].strip())
        self.obs_inds = ast.literal_eval(
            config["input_addition"]["obs_inds"].strip()
        )
        self.obs_shapes = ast.literal_eval(
            config["input_addition"]["obs_shapes"].strip()
        )
        self.obs_ref = ast.literal_eval(config["input_addition"]["obs_ref"].strip())
        self.earth_model_layer_num = int(
            config["input_addition"]["earth_model_layer_num"]
        )
        self.use_spherical = config["input_addition"].getboolean("use_spherical")
        self.cut_stf = int(config["input_addition"]["cut_stf"].strip())
        self.slip_thresh = float(config["input_addition"]["slip_thresh"])

        # [fixed_obs_depth]
        self.fixed_obs_depth = float(
            config['fixed_obs_depth']['fixed_obs_depth'].strip()
        )
        self.obs_lat_range = ast.literal_eval(
            config["fixed_obs_depth"]["obs_lat_range"].strip()
        )
        self.obs_lon_range = ast.literal_eval(
            config["fixed_obs_depth"]["obs_lon_range"].strip()
        )
        self.obs_delta_lat = float(config["fixed_obs_depth"]["obs_delta_lat"])
        self.obs_delta_lon = float(config["fixed_obs_depth"]["obs_delta_lon"])

        # [grn_region]
        self.grn_source_depth_range = ast.literal_eval(
            config["grn_region"]["grn_source_depth_range"].strip()
        )
        self.grn_delta_source_depth = float(config["grn_region"]["grn_delta_source_depth"])
        self.grn_obs_depth_range = ast.literal_eval(
            config["grn_region"]["grn_obs_depth_range"].strip()
        )
        self.grn_delta_obs_depth = float(config["grn_region"]["grn_delta_obs_depth"])
        self.grn_dist_unit = str(config['grn_region']['grn_dist_unit'].strip())
        self.grn_dist_range = ast.literal_eval(config["grn_region"]["grn_dist_range"].strip())
        self.grn_delta_dist = float(config["grn_region"]["grn_delta_dist"])
        if self.grn_dist_unit == 'deg':
            self.grn_dist_range[0] = self.grn_dist_range[0] * d2km
            self.grn_dist_range[1] = self.grn_dist_range[1] * d2km
            self.grn_delta_dist = self.grn_delta_dist * d2km

        # [time_window]
        self.sampling_interval_stf = float(
            config["time_window"]["sampling_interval_stf"]
        )
        self.sampling_interval_cfs = float(
            config["time_window"]["sampling_interval_cfs"]
        )
        self.sampling_num = int(config["time_window"]["sampling_num"])
        self.time_window = (self.sampling_num - 1) * self.sampling_interval_cfs
        try:
            self.max_frequency = float(config["time_window"]["max_frequency"])
        except:
            self.max_frequency = self.sampling_interval_cfs / 2

        # [parallel]
        self.processes_num = int(config["parallel"]["processes_num"])
        self.check_finished = config["parallel"].getboolean("check_finished")

        # depth_list from range and delta_dep
        event_depth_list = np.linspace(
            self.grn_source_depth_range[0],
            self.grn_source_depth_range[1],
            round(
                (self.grn_source_depth_range[1] - self.grn_source_depth_range[0])
                / self.grn_delta_source_depth
                + 1
            ),
        ).tolist()
        obs_depth_list = np.linspace(
            self.grn_obs_depth_range[0],
            self.grn_obs_depth_range[1],
            round(
                (self.grn_obs_depth_range[1] - self.grn_obs_depth_range[0])
                / self.grn_delta_obs_depth
                + 1
            ),
        ).tolist()
        # edgrn2
        self.static_source_depth_range = self.grn_source_depth_range
        self.static_source_delta_depth = self.grn_delta_source_depth
        self.static_dist_range = self.grn_dist_range
        self.static_delta_dist = self.grn_delta_dist
        self.static_obs_depth_list = obs_depth_list

        # qseis2025/qssp2020
        self.event_depth_list = event_depth_list
        self.receiver_depth_list = obs_depth_list

        if config["default_config"].getboolean("default_config"):
            self.default_config = True
            self.set_default()
        else:
            self.default_config = False

            # edgrn2
            self.wavenumber_sampling_rate = int(
                config["static"]["wavenumber_sampling_rate"]
            )
            # edcmp2
            # the following variables will be set by func self.set_default()
            self.layered = config["static"].getboolean("layered")
            if not self.layered:
                self.lam = float(config["static"]["lam"])
                self.mu = float(config["static"]["mu"])

            # both in qseis2025 and qssp2020
            try:
                self.max_slowness = float(config["dynamic"]["max_slowness"])
            except:
                self.max_slowness = None
            self.anti_alias = float(config["dynamic"]["anti_alias"])
            self.free_surface = config["dynamic"].getboolean("free_surface")
            self.wavelet_duration = int(config["dynamic"]["wavelet_duration"])
            self.output_observables = ast.literal_eval(
                config["dynamic"]["output_observables"])

            # qseis2025
            self.slowness_int_algorithm = int(
                config["dynamic"]["slowness_int_algorithm"]
            )
            self.eps_estimate_wavenumber = float(
                config["dynamic"]["eps_estimate_wavenumber"])
            self.source_radius_ratio = float(
                config["dynamic"]["source_radius_ratio"])
            self.slowness_window = ast.literal_eval(
                config["dynamic"]["slowness_window"]
            )
            if self.slowness_window == [0, 0, 0, 0]:
                self.slowness_window = None
            self.time_reduction_velo = float(config["dynamic"]["time_reduction_velo"])

            self.wavelet_type = int(config["dynamic"]["wavelet_type"])
            self.flat_earth_transform = config["dynamic"].getboolean("flat_earth_transform")

            # qssp2020
            self.time_reduction = float(config["dynamic"]["time_reduction"])
            self.spec_time_window = self.time_window
            self.source_radius = float(config["dynamic"]["source_radius"])
            self.turning_point_filter = bool2int(config['dynamic'].getboolean('turning_point_filter'))
            self.turning_point_d1 = int(config["dynamic"]["turning_point_d1"])
            self.turning_point_d2 = int(config["dynamic"]["turning_point_d2"])
            self.gravity_fc = float(config["dynamic"]["gravity_fc"])
            self.gravity_harmonic = int(config["dynamic"]["gravity_harmonic"])
            self.cal_sph = bool2int(config["dynamic"].getboolean("cal_sph"))
            self.cal_tor = bool2int(config["dynamic"].getboolean("cal_sph"))
            self.min_harmonic = int(config["dynamic"]["min_harmonic"])
            self.max_harmonic = int(config["dynamic"]["max_harmonic"])
            self.physical_dispersion = bool2int(
                config["dynamic"].getboolean("physical_dispersion")
            )

    def get_obs_region(self):
        # calculate reference point
        source_points = None
        for ind_src in range(len(self.source_inds)):
            source_plane = pd.read_csv(
                str(
                    os.path.join(
                        self.path_input,
                        f"source_plane{self.source_inds[ind_src]}.csv"
                    )
                ),
                index_col=False,
                header=None,
            ).to_numpy()
            if ind_src == 0:
                source_points = source_plane[:, :3]
            else:
                source_points = np.concatenate(
                    [source_points, source_plane[:, :3]], axis=0
                )

        obs_points = None
        for ind_obs in range(len(self.obs_inds)):
            obs_plane = pd.read_csv(
                str(
                    os.path.join(
                        self.path_input, f"obs_plane{self.obs_inds[ind_obs]}.csv"
                    )
                ),
                index_col=False,
                header=None,
            ).to_numpy()
            if ind_obs == 0:
                obs_points = obs_plane[:, :3]
            else:
                obs_points = np.concatenate([obs_points, obs_plane[:, :3]], axis=0)

        points = np.concatenate([source_points[:, :2], obs_points[:, :2]])
        ref_point = geographic_centroid(points)
        obs_x_min = float(np.min(obs_points[:, 0])) - self.grn_delta_dist
        obs_x_max = float(np.max(obs_points[:, 0])) + self.grn_delta_dist
        obs_y_min = float(np.min(obs_points[:, 1])) - self.grn_delta_dist
        obs_y_max = float(np.max(obs_points[:, 1])) + self.grn_delta_dist
        self.obs_lat_range = [obs_x_min, obs_x_max]
        self.obs_lon_range = [obs_y_min, obs_y_max]
        self.obs_delta_lat = self.grn_delta_dist / d2km
        self.obs_delta_lon = self.grn_delta_dist / d2km
        self.source_ref = list(ref_point)
        self.obs_ref = list(ref_point)
        return ref_point, obs_points

    def set_default(self):
        # edgrn2
        self.wavenumber_sampling_rate = 12

        # edcmp2
        self.layered = True
        if not self.layered:
            self.lam = 30516224000
            self.mu = 33701888000

        # both qseis2025 and qssp2020
        if self.use_spherical:
            nd_model = read_nd(self.path_nd, with_Q=True)
            # vp_max = np.max(nd_model[:, 1])  # km/s
            vs_min = np.min(nd_model[nd_model[:, 2] != 0][:, 2])  # km/s
            self.max_slowness = 1 / vs_min + 0.1  # s/km
        else:
            self.max_slowness = None
        self.anti_alias = 0.01
        self.free_surface = True
        self.wavelet_duration = 5
        self.physical_dispersion = 0
        if self.use_spherical:
            self.output_observables = [0 for _ in range(11)]
            self.output_observables[5] = 1
        else:
            self.output_observables = [0 for _ in range(5)]
            self.output_observables[3] = 1

        # qseis2025
        self.slowness_int_algorithm = 0
        self.eps_estimate_wavenumber = 1e-6
        self.source_radius_ratio = 0.05
        self.slowness_window = None
        self.time_reduction_velo = 0
        self.wavelet_type = 2
        self.flat_earth_transform = True

        # qssp2020
        self.time_reduction = -20
        self.spec_time_window = self.time_window
        self.source_radius = 0
        self.turning_point_filter = 0
        self.turning_point_d1 = 0
        self.turning_point_d2 = 0
        self.gravity_fc = 0
        self.gravity_harmonic = 0
        self.cal_sph = 1
        self.cal_tor = 1
        self.min_harmonic = 6000
        self.max_harmonic = 25000

    def __copy__(self):
        """
        Shallow copy of the configuration.
        Mutable objects are shared with the original, except numpy arrays,
        which are copied to avoid accidental in-place modifications.
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(new_obj, k, v.copy())
            else:
                setattr(new_obj, k, v)
        return new_obj

    def __deepcopy__(self, memo):
        """
        Deep copy of the configuration.
        Recursively copies all attributes; numpy arrays use .copy() for efficiency.
        The memo dict prevents infinite recursion on cyclic references.
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(new_obj, k, v.copy())
            else:
                setattr(new_obj, k, copy.deepcopy(v, memo))
        return new_obj

    def copy(self, deep: bool = True):
        """
        Convenience method to copy the config.
        Args:
            deep (bool): If True, returns a deep copy; otherwise a shallow copy.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)



if __name__ == "__main__":
    pass
