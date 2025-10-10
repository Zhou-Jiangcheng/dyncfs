import datetime
import os
from typing import Union
import pickle
import json
import multiprocessing as mp
from multiprocessing import get_context

# import warnings
# warnings.filterwarnings('error', category=RuntimeWarning)

import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from .configuration import CfsConfig
from .obspy_geo import gps2dist_azimuth
from .create_qssp2020_bulk import (
    pre_process_qssp2020,
    create_grnlib_qssp2020_parallel,
)
from .read_qssp2020 import seek_qssp2020
from .create_qseis2025_bulk import (
    pre_process_qseis2025,
    create_grnlib_qseis2025_parallel,
)
from .read_qseis2025 import seek_qseis2025
from .focal_mechanism import plane2nd, check_convert_fm, mt2plane
from .signal_process import resample, correct_zero_frequency
from .cfs_static import (
    cal_coulomb_failure_stress,
    cal_coulomb_failure_stress_poroelasticity,
    cal_cfs_static_single_point_opt_plane,
)
from .utils import read_source_array, bool2int, ignore_slip_source_array, cut_stf_modify_source_array

d2km = 111.19492664455874


def create_dynamic_lib(config: CfsConfig):
    s = datetime.datetime.now()
    if config.use_spherical:
        pre_process_qssp2020(
            processes_num=config.processes_num,
            path_green=config.path_green_dynamic,
            path_bin=config.path_bin_qssp,
            event_depth_list=config.event_depth_list,
            receiver_depth_list=config.receiver_depth_list,
            spec_time_window=config.spec_time_window,
            sampling_interval=config.sampling_interval_cfs,
            max_frequency=config.max_frequency,
            max_slowness=config.max_slowness,
            anti_alias=config.anti_alias,
            turning_point_filter=config.turning_point_filter,
            turning_point_d1=config.turning_point_d1,
            turning_point_d2=config.turning_point_d2,
            free_surface_filter=bool2int(config.free_surface),
            gravity_fc=config.gravity_fc,
            gravity_harmonic=config.gravity_harmonic,
            cal_sph=config.cal_sph,
            cal_tor=config.cal_tor,
            min_harmonic=config.min_harmonic,
            max_harmonic=config.max_harmonic,
            source_radius=config.source_radius,
            source_duration=config.wavelet_duration / config.sampling_interval_cfs,
            output_observables=config.output_observables,
            time_window=config.time_window,
            time_reduction=config.time_reduction,
            dist_range=config.grn_dist_range,
            delta_dist=config.grn_delta_dist,
            path_nd=config.path_nd,
            earth_model_layer_num=config.earth_model_layer_num,
            physical_dispersion=config.physical_dispersion,
        )
        create_grnlib_qssp2020_parallel(
            path_green=config.path_green_dynamic,
            cal_spec=True,
            check_finished=config.check_finished,
            convert_pd2bin=True,
            remove_pd=True,
        )
    else:
        pre_process_qseis2025(
            processes_num=config.processes_num,
            path_green=config.path_green_dynamic,
            path_bin=config.path_bin_qseis,
            event_depth_list=config.event_depth_list,
            receiver_depth_list=config.receiver_depth_list,
            dist_range=config.grn_dist_range,
            delta_dist=config.grn_delta_dist,
            N_each_group=100,  # less than nrmax in qsglobal.h in qseis_stress_src
            time_window=config.time_window,
            sampling_interval=config.sampling_interval_cfs,
            output_observables=config.output_observables,
            slowness_int_algorithm=config.slowness_int_algorithm,
            eps_estimate_wavenumber=config.eps_estimate_wavenumber,
            source_radius_ratio=config.source_radius_ratio,
            slowness_window=config.slowness_window,
            time_reduction_velo=config.time_reduction_velo,
            wavenumber_sampling_rate=config.wavenumber_sampling_rate,
            anti_alias=config.anti_alias,
            free_surface=config.free_surface,
            wavelet_duration=config.wavelet_duration,
            wavelet_type=config.wavelet_type,
            flat_earth_transform=config.flat_earth_transform,
            path_nd=config.path_nd,
            earth_model_layer_num=config.earth_model_layer_num,
        )
        create_grnlib_qseis2025_parallel(
            path_green=config.path_green_dynamic,
            check_finished=config.check_finished,
            convert_pd2bin=True,
            remove_pd=True,
        )
    e = datetime.datetime.now()
    print("run time:", e - s)


def synthesize_dynamic_stress(
        path_green,
        source_array,
        obs_array_single_point,
        srate_stf,
        static_stress=None,
        max_slowness=None,
        green_info=None,
        use_spherical=False,
):
    """
    :param path_green: Root directory of Green's function library created
                       by create_grnlib_qssp2020_*.
    :param source_array: 2D numpy array, each line contains
                        [lat(deg), lon(deg), depth(km), strike(deg), dip(deg), rake(deg),
                        length_strike(km), length_dip(km), slip(m), m0(Nm),
                        stf_array(dimensionless)]
                        The stf at the end will be normalized by m0.
    :param obs_array_single_point: 1D numpy array,
                        [lat(deg), lon(deg), depth(km),
                        strike(deg), dip(deg), rake(deg)]
    :param srate_stf: Sampling rate of stf in Hz.
    :param static_stress: Static stress tensor used for correcting zero-frequency values.
    :param max_slowness: Seismograms after dist(km)*max_slowness(s/km) will be set to 0.
    :param green_info:
    :param use_spherical:
    :return: stress_ned, shape is (sampling_num, 6)
    """
    if green_info is None:
        with open(os.path.join(path_green, "green_lib_info.json"), "r") as fr:
            green_info = json.load(fr)
    sampling_num = round(green_info["sampling_num"])
    srate_cfs = 1 / green_info["sampling_interval"]

    # lat lon dep len_strike len_dip slip m0 strike dip rake stf
    stress_enz = np.zeros((sampling_num, 6))
    sub_faults_source = source_array[:, :3]
    sub_fms = source_array[:, 3:6]
    sub_m0s = source_array[:, 9]
    sub_stfs = source_array[:, 10:]
    tp_min = np.inf
    dist_m_max = -np.inf
    for i in range(sub_faults_source.shape[0]):
        # print(i, sub_faults_source[i])
        sub_stf = resample(
            sub_stfs[i],
            srate_old=srate_stf,
            srate_new=srate_cfs,
            zero_phase=True,
        )
        m0_sub_stf = (np.sum(sub_stf) / srate_cfs)
        if m0_sub_stf != 0:
            sub_stf = sub_stf / m0_sub_stf * sub_m0s[i]
        else:
            continue
        dist_in_m, az_in_deg, baz_in_deg = gps2dist_azimuth(
            lat1=sub_faults_source[i][0],
            lon1=sub_faults_source[i][1],
            lat2=obs_array_single_point[0],
            lon2=obs_array_single_point[1],
        )
        focal_mechanism = sub_fms[i]
        if use_spherical:
            (
                stress_enz_1source,
                tpts_table,
                _,
                _,
                grn_dep_source,
                grn_dep_receiver,
                grn_dist,
            ) = seek_qssp2020(
                path_green=path_green,
                event_depth_km=sub_faults_source[i, 2],
                receiver_depth_km=obs_array_single_point[2],
                az_deg=az_in_deg,
                dist_km=dist_in_m / 1e3,
                focal_mechanism=focal_mechanism,
                srate=srate_cfs,
                before_p=None,
                pad_zeros=True,
                shift=False,
                rotate=True,
                only_seismograms=False,
                output_type="stress",
                model_name=green_info["path_nd_without_Q"],
                green_info=green_info,
            )
        else:
            (
                stress_enz_1source,
                tpts_table,
                _,
                _,
                grn_dep_source,
                grn_dep_receiver,
                grn_dist,
            ) = seek_qseis2025(
                path_green=path_green,
                event_depth_km=sub_faults_source[i, 2],
                receiver_depth_km=obs_array_single_point[2],
                az_deg=az_in_deg,
                dist_km=dist_in_m / 1e3,
                focal_mechanism=focal_mechanism,
                srate=srate_cfs,
                before_p=None,
                pad_zeros=True,
                shift=False,
                rotate=True,
                only_seismograms=False,
                output_type="stress",
                model_name=green_info["path_nd_without_Q"],
                green_info=green_info,
            )
        tp_i = tpts_table["p_onset"]
        tp_ic = max(1, round((tp_i - 1) * srate_cfs))
        before_p_mean = np.array([np.mean(stress_enz_1source[:, :tp_ic], axis=1)]).T
        stress_enz_1source = stress_enz_1source - before_p_mean
        conv_result = (
                signal.convolve(
                    stress_enz_1source.T, sub_stf[:, None], mode="full", method="auto"
                )
                / srate_cfs
        )
        stress_enz_1source = conv_result[:sampling_num, :]
        stress_enz = stress_enz + stress_enz_1source
        if tp_min > tp_i:
            tp_min = tp_i
        if dist_m_max < dist_in_m:
            dist_m_max = dist_in_m

    if use_spherical:
        wavelet_duration = round(green_info["source_duration"] / srate_cfs)
    else:
        wavelet_duration = green_info["wavelet_duration"]
    if (static_stress is not None) and (max_slowness is not None):
        tc1 = max(1, round(tp_min * srate_cfs - 1))
        tc2 = round(
            dist_m_max / 1e3 * max_slowness
            + 1.5 * wavelet_duration
            + sub_stfs.shape[1] * srate_cfs / srate_stf
        )
        stress_rate_enz = (
                signal.convolve(
                    stress_enz, np.array([1, -1])[:, None], mode="same", method="auto"
                )
                / srate_cfs
        )
        for i_cor in range(6):
            stress_rate_enz[:, i_cor] = correct_zero_frequency(
                data=stress_rate_enz[:, i_cor],
                srate=srate_cfs,
                A0=static_stress[i_cor],
                f_c=min(4, tc2 - tc1),
                tc1=tc1,
                tc2=tc2,
                ratio_interp=0,
            )
            stress_enz[:, i_cor] = np.sum(stress_rate_enz[:, i_cor]) / srate_cfs
    elif max_slowness is not None:
        tc2 = round(
            dist_m_max / 1e3 * max_slowness
            + 1.5 * wavelet_duration
            + sub_stfs.shape[1] * srate_cfs / srate_stf
        )
        final_values = np.mean(
            stress_enz[tc2 + round(10 * srate_cfs): tc2 + round(20 * srate_cfs), :],
            axis=0,
        )
        stress_enz[tc2:, :] = np.array([final_values]) * np.ones_like(
            stress_enz[tc2:, :]
        )

    # sigma_ee, sigma_en, sigma_ez, sigma_nn, sigma_nz, sigma_zz
    # sigma_nn, sigma_ne, sigma_nd, sigma_ee, sigma_ed, sigma_dd
    stress_ned = np.zeros_like(stress_enz)
    stress_ned[:, 0] = stress_enz[:, 3]
    stress_ned[:, 1] = stress_enz[:, 1]
    stress_ned[:, 2] = -stress_enz[:, 4]
    stress_ned[:, 3] = stress_enz[:, 0]
    stress_ned[:, 4] = -stress_enz[:, 2]
    stress_ned[:, 5] = stress_enz[:, 5]
    return stress_ned


def cal_stress_vector_ned_dynamic(stress_ned, n):
    stress_tensor_ned = np.array(
        [
            [stress_ned[:, 0], stress_ned[:, 1], stress_ned[:, 2]],
            [stress_ned[:, 1], stress_ned[:, 3], stress_ned[:, 4]],
            [stress_ned[:, 2], stress_ned[:, 4], stress_ned[:, 5]],
        ]
    ).T  # shape (n, 3, 3)
    sigma_vector = np.einsum("ijk,k->ij", stress_tensor_ned, n.flatten())
    return sigma_vector


def cal_cfs_dynamic_single_point_fm(
        path_green: str,
        source_array: Union[np.ndarray, str],
        obs_array_single_point: np.ndarray,
        srate_stf: float,
        mu_f: float = 0.4,
        B_pore: float = 0,
        max_slowness: float = None,
        green_info: dict = None,
        path_results_each: str = None,
        use_spherical: bool = False,
        static_stress=None,
        check_finish: bool = False,
):
    """
    :param path_green: Root directory of Green's function library created
                       by create_grnlib_qssp2020_*.
    :param source_array: 2D numpy array, each line contains
                        [lat(deg), lon(deg), depth(km), strike(deg), dip(deg), rake(deg),
                        length_strike(km), length_dip(km), slip(m), m0(Nm),
                        stf_array(dimensionless)]
                        The stf at the end will be normalized by m0.
    :param obs_array_single_point: 1D numpy array,
                        [lat(deg), lon(deg), depth(km),
                        strike(deg), dip(deg), rake(deg)]
    :param srate_stf: Sampling rate of stf in Hz.
    :param mu_f: Coefficient or effective coefficient of friction.
    :param B_pore: Skempton's coefficient. If zero, the mu_f means effective coefficient of friction.
    :param max_slowness: Seismograms after dist(km)*max_slowness(s/km) will be set to 0.
    :param green_info:
    :param path_results_each:
    :param use_spherical:
    :param static_stress: Static stress tensor used for correcting zero-frequency values.
    :param check_finish: If True, read the stress_ned.npy file if it already exists

    Note: The sampling interval for all return values
          is the same as green_info['sampling_interval']
    :return: (
        stress_ned,
        n,
        d,
        sigma,
        tau,
        cfs
    )
    """
    print('obs point:', obs_array_single_point)
    file_name = "%.4f_%.4f_%.4f" % (
        float(obs_array_single_point[0]),
        float(obs_array_single_point[1]),
        float(obs_array_single_point[2]),
    )

    if isinstance(source_array, str):
        source_array = np.load(source_array)

    if green_info is None:
        with open(os.path.join(path_green, "green_lib_info.json"), "r") as fr:
            green_info = json.load(fr)

    if path_results_each is not None:
        path_stress_ned = str(
            os.path.join(
                path_results_each,
                file_name + "_stress_ned.npy",
            )
        )
    else:
        path_stress_ned = ''
    if check_finish and os.path.exists(path_stress_ned):
        stress_ned = np.load(path_stress_ned)
    else:
        stress_ned = synthesize_dynamic_stress(
            path_green=path_green,
            source_array=source_array,
            obs_array_single_point=obs_array_single_point,
            srate_stf=srate_stf,
            static_stress=static_stress,
            max_slowness=max_slowness,
            green_info=green_info,
            use_spherical=use_spherical,
        )

    n_obs, d_obs = plane2nd(*obs_array_single_point[3:])
    n = np.array([n_obs.flatten()]).T
    d = np.array([d_obs.flatten()]).T
    sigma_vector = cal_stress_vector_ned_dynamic(stress_ned, n)
    sigma = np.dot(sigma_vector, n).flatten()
    tau = np.dot(sigma_vector, d).flatten()
    mean_stress = (stress_ned[:, 0] + stress_ned[:, 3] + stress_ned[:, 5]) / 3
    if B_pore != 0:
        cfs = cal_coulomb_failure_stress(
            norm_stress=sigma, shear_stress=tau, mu_f=mu_f
        )
    else:
        cfs = cal_coulomb_failure_stress_poroelasticity(
            norm_stress=sigma,
            shear_stress=tau,
            mean_stress=mean_stress,
            mu_f=mu_f,
            B_pore=B_pore,
        )

    if path_results_each is not None:
        results = (
            stress_ned,
            np.ones((len(stress_ned), 3)) * n.flatten(),
            np.ones((len(stress_ned), 3)) * d.flatten(),
            sigma,
            tau,
            cfs
        )
        np.save(
            str(
                os.path.join(
                    path_results_each,
                    file_name + "_stress_ned.npy",
                )
            ),
            stress_ned,
        )
        np.save(
            str(
                os.path.join(
                    path_results_each,
                    file_name + "_cfs.npy",
                )
            ),
            cfs,
        )
        for j in range(len(result_name_list)):
            np.save(
                str(
                    os.path.join(
                        path_results_each,
                        file_name + "_%s.npy" % result_name_list[j],
                    )
                ),
                results[j + 1],
            )
    return (
        stress_ned,
        np.ones((len(stress_ned), 3)) * n.flatten(),
        np.ones((len(stress_ned), 3)) * d.flatten(),
        sigma,
        tau,
        cfs
    )


def cal_cfs_dynamic_single_point_opt_rake(
        path_green: str,
        source_array: Union[np.ndarray, str],
        obs_array_single_point: np.ndarray,
        srate_stf: float,
        tectonic_stress: Union[list[float], np.ndarray],
        mu_f: float = 0.4,
        B_pore: float = 0,
        max_slowness: float = None,
        green_info: dict = None,
        path_results_each: str = None,
        use_spherical: bool = False,
        static_stress=None,
        check_finish: bool = False,
):
    """
    :param path_green: Root directory of Green's function library created
                       by create_grnlib_qssp2020_*.
    :param source_array: 2D numpy array, each line contains
                        [lat(deg), lon(deg), depth(km), strike(deg), dip(deg), rake(deg),
                        length_strike(km), length_dip(km), slip(m), m0(Nm),
                        stf_array(dimensionless)]
                        The stf at the end will be normalized by m0.
    :param obs_array_single_point: 1D numpy array,
                        [lat(deg), lon(deg), depth(km),
                        strike(deg), dip(deg), rake(deg)]
    :param srate_stf: Sampling rate of stf in Hz.
    :param tectonic_stress: Tectonic stress tensor in NED axis,
            [s_nn, s_ne, s_nd, s_ee, s_ed, s_dd] (Pa);
    :param mu_f: Coefficient or effective coefficient of friction.
    :param B_pore: Skempton's coefficient. If zero, the mu_f means effective coefficient of friction.
    :param max_slowness: Seismograms after dist(km)*max_slowness(s/km) will be set to 0.
    :param green_info:
    :param path_results_each:
    :param use_spherical:
    :param static_stress: Static stress tensor used for correcting zero-frequency values.
    :param check_finish: If True, read the stress_ned.npy file if it already exists

    Note: The sampling interval for all return values
          is the same as green_info['sampling_interval']
    :return: (
        stress_ned,
        n,
        d,
        sigma,
        tau,
        cfs,
        rake
    )
    """
    print('obs point:', obs_array_single_point)
    file_name = "%.4f_%.4f_%.4f" % (
        float(obs_array_single_point[0]),
        float(obs_array_single_point[1]),
        float(obs_array_single_point[2]),
    )

    if isinstance(source_array, str):
        source_array = np.load(source_array)

    if green_info is None:
        with open(os.path.join(path_green, "green_lib_info.json"), "r") as fr:
            green_info = json.load(fr)

    if path_results_each is not None:
        path_stress_ned = str(
            os.path.join(
                path_results_each,
                file_name + "_stress_ned.npy",
            )
        )
    else:
        path_stress_ned = ''
    if check_finish and os.path.exists(path_stress_ned):
        stress_ned = np.load(path_stress_ned)
    else:
        stress_ned = synthesize_dynamic_stress(
            path_green=path_green,
            source_array=source_array,
            obs_array_single_point=obs_array_single_point,
            srate_stf=srate_stf,
            static_stress=static_stress,
            max_slowness=max_slowness,
            green_info=green_info,
            use_spherical=use_spherical,
        )

    # to be continued


def cal_cfs_dynamic_single_point_oop(
        path_green: str,
        source_array: Union[np.ndarray, str],
        obs_array_single_point: np.ndarray,
        srate_stf: float,
        tectonic_stress_type: int,
        tectonic_stress: Union[list[float], np.ndarray],
        mu_f: float = 0.4,
        B_pore: float = 0,
        max_slowness: float = None,
        green_info: dict = None,
        path_results_each: str = None,
        use_spherical: bool = False,
        check_finish: bool = False
):
    """
    :param path_green: Root directory of Green's function library created
                       by create_grnlib_qssp2020_*.
    :param source_array: 2D numpy array, each line contains
                        [lat(deg), lon(deg), depth(km), strike(deg), dip(deg), rake(deg),
                        length_strike(km), length_dip(km), slip(m), m0(Nm),
                        stf_array(dimensionless)]
                        The stf at the end will be normalized by m0.
    :param obs_array_single_point: 1D numpy array,
                        [lat(deg), lon(deg), depth(km),
                        strike(deg), dip(deg), rake(deg)]
    :param srate_stf: Sampling rate of stf in Hz.
    :param tectonic_stress_type: 1 for stress, 2 for orientations
    :param tectonic_stress:
            When tectonic_stress_type=1, tectonic stress tensor in NED axis,
            [s_nn, s_ne, s_nd, s_ee, s_ed, s_dd] (Pa);
            when tectonic_stress_type=2, tectonic stress tensor in NED axis,
            [azimuth1, plunge1, azimuth2, plunge2, azimuth3, plunge3] (deg);
    :param mu_f: Coefficient or effective coefficient of friction.
    :param B_pore: Skempton's coefficient. If zero, the mu_f means effective coefficient of friction.
    :param max_slowness: Seismograms after dist(km)*max_slowness(s/km) will be set to 0.
    :param green_info:
    :param path_results_each:
    :param use_spherical:
    :param check_finish: If True, read the stress_ned.npy file if it already exists

    Note: The sampling interval for all return values is the same as green'info ['sampling_interval ']
    :return: (
        stress_ned,
        n1,
        d1,
        sigma1,
        tau1,
        n2,
        d2,
        sigma2,
        tau2,
        cfs,
    )

    """
    print('obs point:', obs_array_single_point)
    file_name = "%.4f_%.4f_%.4f" % (
        float(obs_array_single_point[0]),
        float(obs_array_single_point[1]),
        float(obs_array_single_point[2]),
    )

    if isinstance(source_array, str):
        source_array = np.load(source_array)

    if green_info is None:
        with open(os.path.join(path_green, "green_lib_info.json"), "r") as fr:
            green_info = json.load(fr)

    if path_results_each is not None:
        path_stress_ned = str(
            os.path.join(
                path_results_each,
                file_name + "_stress_ned.npy",
            )
        )
    else:
        path_stress_ned = ''
    if check_finish and os.path.exists(path_stress_ned):
        stress_ned = np.load(path_stress_ned)
    else:
        stress_ned = synthesize_dynamic_stress(
            path_green=path_green,
            source_array=source_array,
            obs_array_single_point=obs_array_single_point,
            srate_stf=srate_stf,
            max_slowness=max_slowness,
            green_info=green_info,
            use_spherical=use_spherical,
        )
    sampling_num = len(stress_ned)

    n1 = np.zeros((sampling_num, 3))
    d1 = np.zeros((sampling_num, 3))
    sigma1 = np.zeros(sampling_num)
    tau1 = np.zeros(sampling_num)

    n2 = np.zeros((sampling_num, 3))
    d2 = np.zeros((sampling_num, 3))
    sigma2 = np.zeros(sampling_num)
    tau2 = np.zeros(sampling_num)

    cfs = np.zeros(sampling_num)

    for i in range(sampling_num):
        (
            [n1i, d1i, sigma1i, tau1i],
            [n2i, d2i, sigma2i, tau2i],
            cfsi
        ) = cal_cfs_static_single_point_opt_plane(
            stress=stress_ned[i],
            tectonic_stress_type=tectonic_stress_type,
            tectonic_stress=tectonic_stress,
            mu_f=mu_f,
            B_pore=B_pore,
        )
        n1[i] = n1i
        d1[i] = d1i
        sigma1[i] = sigma1i
        tau1[i] = tau1i

        n2[i] = n2i
        d2[i] = d2i
        sigma2[i] = sigma2i
        tau2[i] = tau2i
        cfs[i] = cfsi

    if path_results_each is not None:
        results = (
            stress_ned,
            [n1, d1, sigma1, tau1],
            [n2, d2, sigma2, tau2],
            cfs,
        )
        np.save(
            str(
                os.path.join(
                    path_results_each,
                    file_name + "_stress_ned.npy",
                )
            ),
            stress_ned,
        )
        np.save(
            str(
                os.path.join(
                    path_results_each,
                    file_name + "_cfs.npy",
                )
            ),
            cfs,
        )
        for jj in range(2):
            for j in range(len(result_name_list)):
                np.save(
                    str(
                        os.path.join(
                            path_results_each,
                            file_name + "_%s%d.npy" % (result_name_list[j], jj + 1),
                        )
                    ),
                    results[jj + 1][j],
                )

    return (
        stress_ned,
        n1,
        d1,
        sigma1,
        tau1,
        n2,
        d2,
        sigma2,
        tau2,
        cfs,
    )


def prepare_compute_cfs(config: CfsConfig):
    source_array = read_source_array(
        source_inds=config.source_inds,
        path_input=config.path_input,
    )
    if config.slip_thresh > 0:
        source_array = ignore_slip_source_array(
            source_array, config.slip_thresh)
    if config.cut_stf > 0:
        source_array = cut_stf_modify_source_array(
            source_array, config.cut_stf)
    with open(
            os.path.join(config.path_green_dynamic, "green_lib_info.json"), "r"
    ) as fr:
        green_info = json.load(fr)
    path_results_each = os.path.join(config.path_output, "grn_d", "results_each")
    os.makedirs(path_results_each, exist_ok=True)
    np.save(os.path.join(path_results_each, 'source_array.npy'), source_array)
    for ind_obs in config.obs_inds:
        obs_plane = pd.read_csv(
            str(os.path.join(config.path_input,
                             "obs_plane%d.csv" % ind_obs)),
            index_col=False,
            header=None,
        ).to_numpy()
        if config.optimal_type:
            inp_list = []
            for i in range(len(obs_plane)):
                inp = [
                    config.path_green_dynamic,
                    obs_plane[i, :],
                    1 / config.sampling_interval_stf,
                    config.mu_f,
                    config.B_pore,
                    config.max_slowness,
                    green_info,
                    path_results_each,
                    config.use_spherical,
                ]
                inp_list.append(inp)
            with open(
                    os.path.join(path_results_each, "group_list_%d.pkl" % ind_obs), "wb"
            ) as fw:
                pickle.dump(inp_list, fw)  # type: ignore
        else:
            inp_list = []
            for i in range(len(obs_plane)):
                inp = [
                    config.path_green_dynamic,
                    obs_plane[i, :],
                    1 / config.sampling_interval_stf,
                    config.tectonic_stress_type,
                    config.tectonic_stress,
                    config.mu_f,
                    config.B_pore,
                    config.max_slowness,
                    green_info,
                    path_results_each,
                    config.use_spherical,
                ]
                inp_list.append(inp)
            with open(
                    os.path.join(path_results_each, "group_list_%d.pkl" % ind_obs), "wb"
            ) as fw:
                pickle.dump(inp_list, fw)  # type: ignore


def compute_dynamic_cfs_parallel(config: CfsConfig):
    s = datetime.datetime.now()
    if config.multiprocessing_flag is None:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        config.multiprocessing_flag = 1
    mp.set_start_method('spawn', force=True)
    ctx = get_context("spawn")

    prepare_compute_cfs(config)
    path_results_each = str(os.path.join(config.path_output, "grn_d", "results_each"))
    path_source_array = str(os.path.join(path_results_each, 'source_array.npy'))
    for ind_obs in config.obs_inds:
        print('Parallel computing cfs on Plane No.%d' % ind_obs)
        print('This process may take a long time.')
        with open(os.path.join(path_results_each, f"group_list_{ind_obs}.pkl"), "rb") as fr:
            group_list = pickle.load(fr)

        jobs = []
        for args in group_list:
            # [path_green, source_array, ...]
            args = [args[0], path_source_array] + args[1:] + [config.check_finished]
            jobs.append(args)

        target = cal_cfs_dynamic_single_point_fm if config.optimal_type \
            else cal_cfs_dynamic_single_point_oop

        with ctx.Pool(processes=config.processes_num, maxtasksperchild=1) as pool:
            pool.starmap(target, jobs)

        obs_plane = pd.read_csv(
            os.path.join(config.path_input, f"obs_plane{ind_obs}.csv"),
            index_col=False, header=None,
        ).to_numpy()
        read_stress_results(
            obs_plane=obs_plane,
            ind_obs=ind_obs,
            path_output=config.path_output,
            sampling_num=config.sampling_num,
            specified_orientation=config.optimal_type,
        )

    e = datetime.datetime.now()
    print("run time:", e - s)


def compute_dynamic_cfs_sequential(config: CfsConfig):
    s = datetime.datetime.now()
    if config.multiprocessing_flag is not None:
        os.environ['OMP_NUM_THREADS'] = ''
        os.environ['MKL_NUM_THREADS'] = ''
        os.environ['OPENBLAS_NUM_THREADS'] = ''
        config.multiprocessing_flag = None
    source_array = read_source_array(
        source_inds=config.source_inds,
        path_input=config.path_input,
    )
    if config.slip_thresh > 0:
        source_array = ignore_slip_source_array(
            source_array, config.slip_thresh)
    if config.cut_stf > 0:
        source_array = cut_stf_modify_source_array(
            source_array, config.cut_stf)
    with open(
            os.path.join(config.path_green_dynamic, "green_lib_info.json"), "r"
    ) as fr:
        green_info = json.load(fr)
    path_results_each = os.path.join(config.path_output, "grn_d", "results_each")
    os.makedirs(path_results_each, exist_ok=True)
    for ind_obs in config.obs_inds:
        obs_plane = pd.read_csv(
            str(os.path.join(config.path_input, "obs_plane%d.csv" % ind_obs)),
            index_col=False,
            header=None,
        ).to_numpy()
        if config.correct_zero_freq:
            path_static_stress_tensor = str(
                os.path.join(
                    config.path_output,
                    "results",
                    "static",
                    "stress_tensor_plane%d.npy" % ind_obs,
                )
            )
            static_stress_obs = np.load(path_static_stress_tensor)  # ned
        else:
            static_stress_obs = None

        if config.optimal_type:
            for i in tqdm(
                    range(len(obs_plane)),
                    desc="Computing dynamic Coulomb Failure Stress change at No.%d plane"
                         % ind_obs,
            ):
                if static_stress_obs is not None:
                    static_stress = np.array(
                        [
                            static_stress_obs[i, 3],
                            static_stress_obs[i, 1],
                            -static_stress_obs[i, 4],
                            static_stress_obs[i, 0],
                            -static_stress_obs[i, 2],
                            static_stress_obs[i, 5],
                        ]
                    )  # enz
                else:
                    static_stress = None
                cal_cfs_dynamic_single_point_fm(
                    path_green=config.path_green_dynamic,
                    source_array=source_array,
                    obs_array_single_point=obs_plane[i, :],
                    srate_stf=1 / config.sampling_interval_stf,
                    mu_f=config.mu_f,
                    B_pore=config.B_pore,
                    static_stress=static_stress,
                    max_slowness=config.max_slowness,
                    green_info=green_info,
                    path_results_each=path_results_each,
                    use_spherical=config.use_spherical,
                    check_finish=config.check_finished
                )
        else:
            for i in tqdm(
                    range(len(obs_plane)),
                    desc="Computing dynamic Coulomb Failure Stress change at No.%d plane"
                         % ind_obs,
            ):
                cal_cfs_dynamic_single_point_oop(
                    path_green=config.path_green_dynamic,
                    source_array=source_array,
                    obs_array_single_point=obs_plane[i, :],
                    tectonic_stress_type=config.tectonic_stress_type,
                    tectonic_stress=config.tectonic_stress,
                    srate_stf=1 / config.sampling_interval_stf,
                    mu_f=config.mu_f,
                    max_slowness=config.max_slowness,
                    green_info=green_info,
                    path_results_each=path_results_each,
                    use_spherical=config.use_spherical,
                    check_finish=config.check_finished
                )
        read_stress_results(
            obs_plane=obs_plane,
            ind_obs=ind_obs,
            path_output=config.path_output,
            sampling_num=config.sampling_num,
            specified_orientation=config.optimal_type,
        )
    e = datetime.datetime.now()
    print("run time:", e - s)


result_name_list = [
    "normal_vector",
    "rupture_vector",
    "normal_stress",
    "shear_stress",
]


def read_stress_results(
        obs_plane, ind_obs, path_output, sampling_num, specified_orientation
):
    print("Outputting results for No.%d obs_plane to csv files" % ind_obs)
    path_results_each = os.path.join(path_output, "grn_d", "results_each")
    path_output_results = os.path.join(path_output, "results", "dynamic")
    os.makedirs(path_output_results, exist_ok=True)

    if specified_orientation:
        file_info = [[item, "_%s.npy" % item, 1] for item in result_name_list]
        file_info[0][2] = 3
        file_info[1][2] = 3
    else:
        file_info1 = [[item + '1', "_%s1.npy" % item, 1] for item in result_name_list]
        file_info2 = [[item + '2', "_%s2.npy" % item, 1] for item in result_name_list]
        file_info1[0][2] = 3
        file_info1[1][2] = 3
        file_info2[0][2] = 3
        file_info2[1][2] = 3
        file_info = file_info1 + file_info2
    file_info = [["stress_ned", "_stress_ned.npy", 6], ["cfs", "_cfs.npy", 1]] + file_info
    data = {
        key: np.zeros((len(obs_plane) * col_num, sampling_num))
        for key, _, col_num in file_info
    }
    for i in range(len(obs_plane)):
        lat, lon, dep = obs_plane[i, 0:3]
        fname = "%.4f_%.4f_%.4f" % (lat, lon, dep)
        for key, suffix, col_num in file_info:
            arr = np.load(os.path.join(path_results_each, fname + suffix))
            data[key][i * col_num: (i + 1) * col_num] = arr.T

    for key, arr in data.items():
        if specified_orientation:
            out_name = "%s_dynamic_plane%d.csv" % (key, ind_obs)
        else:
            out_name = "%s_dynamic_plane%d.csv" % (key + "_oop", ind_obs)
        pd.DataFrame(arr).to_csv(
            os.path.join(path_output_results, out_name), header=False, index=False
        )


def prepare_compute_cfs_fix_depth(
        config: CfsConfig, obs_depth: float = None, receiver_mechanism=None):
    if obs_depth is None:
        obs_depth = config.fixed_obs_depth
    source_array = read_source_array(
        source_inds=config.source_inds,
        path_input=config.path_input,
    )
    if config.slip_thresh > 0:
        source_array = ignore_slip_source_array(
            source_array, config.slip_thresh)
    if config.cut_stf > 0:
        source_array = cut_stf_modify_source_array(
            source_array, config.cut_stf)
    if config.optimal_type and receiver_mechanism is None:
        mt_mean = np.zeros(6)
        for i in range(len(source_array)):
            mt_i = check_convert_fm(source_array[i, 3:6])
            mt_i = np.array(mt_i) * source_array[i, 9]
            mt_mean = mt_mean + mt_i
        receiver_mechanism = mt2plane(mt=mt_mean)[0]
        print(
            "receiver_mechanism is (strike, dip, rake)=(%.2f, %.2f, %.2f) deg."
            % (receiver_mechanism[0], receiver_mechanism[1], receiver_mechanism[2])
        )
    with open(
            os.path.join(config.path_green_dynamic, "green_lib_info.json"), "r"
    ) as fr:
        green_info = json.load(fr)
    path_results_each = os.path.join(config.path_output, "grn_d", "results_each")
    os.makedirs(path_results_each, exist_ok=True)

    Nx = int(np.ceil((config.obs_lat_range[1] - config.obs_lat_range[0]) / config.obs_delta_lat) + 1)
    Ny = int(np.ceil((config.obs_lon_range[1] - config.obs_lon_range[0]) / config.obs_delta_lon) + 1)

    obs_plane = np.zeros((Nx * Ny, 6))
    obs_plane[:, 2] = obs_plane[:, 2] + obs_depth
    lat_array = np.linspace(config.obs_lat_range[0], config.obs_lat_range[1], Nx)
    lon_array = np.linspace(config.obs_lon_range[0], config.obs_lon_range[1], Ny)
    for i in range(Nx):
        for j in range(Ny):
            ind = j + i * Ny
            obs_plane[ind, :2] = np.array([lat_array[i], lon_array[j]])
    if config.optimal_type:
        obs_plane[:, 3] = obs_plane[:, 3] + receiver_mechanism[0]
        obs_plane[:, 4] = obs_plane[:, 4] + receiver_mechanism[1]
        obs_plane[:, 5] = obs_plane[:, 5] + receiver_mechanism[2]

    with open(
            os.path.join(config.path_green_dynamic, "green_lib_info.json"), "r"
    ) as fr:
        green_info = json.load(fr)
    path_results_each = os.path.join(config.path_output, "grn_d", "results_each")
    os.makedirs(path_results_each, exist_ok=True)
    np.save(os.path.join(path_results_each, 'source_array.npy'), source_array)
    np.save(os.path.join(path_results_each, 'obs_plane_%.2f.npy' % obs_depth),
            obs_plane)

    if config.optimal_type:
        inp_list = []
        for i in range(len(obs_plane)):
            inp = [
                config.path_green_dynamic,
                obs_plane[i, :],
                1 / config.sampling_interval_stf,
                config.mu_f,
                config.B_pore,
                config.max_slowness,
                green_info,
                path_results_each,
                config.use_spherical,
            ]
            inp_list.append(inp)
        with open(
                os.path.join(path_results_each, "group_list_%.2f.pkl" % obs_depth), "wb"
        ) as fw:
            pickle.dump(inp_list, fw)  # type: ignore
    else:
        inp_list = []
        for i in range(len(obs_plane)):
            inp = [
                config.path_green_dynamic,
                obs_plane[i, :],
                1 / config.sampling_interval_stf,
                config.tectonic_stress_type,
                config.tectonic_stress,
                config.mu_f,
                config.B_pore,
                config.max_slowness,
                green_info,
                path_results_each,
                config.use_spherical,
            ]
            inp_list.append(inp)
        with open(
                os.path.join(path_results_each, "group_list_%.2f.pkl" % obs_depth), "wb"
        ) as fw:
            pickle.dump(inp_list, fw)  # type: ignore


def compute_dynamic_cfs_fix_depth_parallel(
        config: CfsConfig, obs_depth: float = None, receiver_mechanism=None):
    s = datetime.datetime.now()
    if obs_depth is None:
        obs_depth = config.fixed_obs_depth
    if config.multiprocessing_flag is None:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
    mp.set_start_method('spawn', force=True)
    ctx = get_context("spawn")

    prepare_compute_cfs_fix_depth(config, obs_depth, receiver_mechanism)
    path_results_each = str(os.path.join(config.path_output, "grn_d", "results_each"))
    path_source_array = str(os.path.join(path_results_each, 'source_array.npy'))
    print('Parallel computing cfs on  %.2f km depth' % obs_depth)
    print('This process may take a long time.')
    with open(os.path.join(path_results_each, f"group_list_%.2f.pkl" % obs_depth), "rb") as fr:
        group_list = pickle.load(fr)

    jobs = []
    for args in group_list:
        # [path_green, source_array, ...]
        args = [args[0], path_source_array] + args[1:] + [config.check_finished]
        jobs.append(args)

    target = cal_cfs_dynamic_single_point_fm if config.optimal_type \
        else cal_cfs_dynamic_single_point_oop

    with ctx.Pool(processes=config.processes_num, maxtasksperchild=1) as pool:
        pool.starmap(target, jobs)

    obs_plane = np.load(os.path.join(path_results_each, 'obs_plane_%.2f.npy' % obs_depth))
    read_stress_results_fix_depth(
        obs_plane=obs_plane,
        obs_depth=obs_depth,
        path_output=config.path_output,
        sampling_num=config.sampling_num,
        specified_orientation=config.optimal_type,
    )

    e = datetime.datetime.now()
    print("run time:", e - s)


def compute_dynamic_cfs_fix_depth_sequential(
        config: CfsConfig,
        obs_depth: float = None,
        receiver_mechanism: list = None,
        obs_lat_range: list = None,
        obs_lon_range: list = None,
        obs_delta_lat: float = None,
        obs_delta_lon: float = None,
):
    """
    Compute static Coulomb failure stress change at fixed depth.

    :param config:
    :param obs_depth: Observation depth, Default equals to config.fixed_obs_depth, unit km.
    :param receiver_mechanism: [strike, dip, rake] of receiver fault, if None, set as the
                               mean focal mechanism of the source faults.
    :param obs_lat_range: Default equals to config.obs_x_range, unit deg.
    :param obs_lon_range: Default equals to config.obs_y_range, unit deg.
    :param obs_delta_lat: Default equals to config.obs_delta_x, unit deg.
    :param obs_delta_lon: Default equals to config.obs_delta_y, unit deg.
    """
    if obs_depth is None:
        obs_depth = config.fixed_obs_depth
    if config.multiprocessing_flag is not None:
        os.environ['OMP_NUM_THREADS'] = ''
        os.environ['MKL_NUM_THREADS'] = ''
        os.environ['OPENBLAS_NUM_THREADS'] = ''
        config.multiprocessing_flag = None

    if obs_lat_range is None:
        obs_lat_range = config.obs_lat_range
    if obs_lon_range is None:
        obs_lon_range = config.obs_lon_range
    if obs_delta_lat is None:
        obs_delta_lat = config.obs_delta_lat
    if obs_delta_lon is None:
        obs_delta_lon = config.obs_delta_lon

    s = datetime.datetime.now()

    source_array = read_source_array(
        source_inds=config.source_inds,
        path_input=config.path_input,
    )
    if config.slip_thresh > 0:
        source_array = ignore_slip_source_array(
            source_array, config.slip_thresh)
    if config.cut_stf > 0:
        source_array = cut_stf_modify_source_array(
            source_array, config.cut_stf)
    if config.optimal_type and receiver_mechanism is None:
        mt_mean = np.zeros(6)
        for i in range(len(source_array)):
            mt_i = check_convert_fm(source_array[i, 3:6])
            mt_i = np.array(mt_i) * source_array[i, 9]
            mt_mean = mt_mean + mt_i
        receiver_mechanism = mt2plane(mt=mt_mean)[0]
        print(
            "receiver_mechanism is (strike, dip, rake)=(%.2f, %.2f, %.2f) deg."
            % (receiver_mechanism[0], receiver_mechanism[1], receiver_mechanism[2])
        )
    with open(
            os.path.join(config.path_green_dynamic, "green_lib_info.json"), "r"
    ) as fr:
        green_info = json.load(fr)
    path_results_each = os.path.join(config.path_output, "grn_d", "results_each")
    os.makedirs(path_results_each, exist_ok=True)

    Nx = int(np.ceil((obs_lat_range[1] - obs_lat_range[0]) / obs_delta_lat) + 1)
    Ny = int(np.ceil((obs_lon_range[1] - obs_lon_range[0]) / obs_delta_lon) + 1)

    obs_plane = np.zeros((Nx * Ny, 6))
    obs_plane[:, 2] = obs_plane[:, 2] + obs_depth
    lat_array = np.linspace(obs_lat_range[0], obs_lat_range[1], Nx)
    lon_array = np.linspace(obs_lon_range[0], obs_lon_range[1], Ny)
    for i in range(Nx):
        for j in range(Ny):
            ind = j + i * Ny
            obs_plane[ind, :2] = np.array([lat_array[i], lon_array[j]])
    if config.optimal_type:
        obs_plane[:, 3] = obs_plane[:, 3] + receiver_mechanism[0]
        obs_plane[:, 4] = obs_plane[:, 4] + receiver_mechanism[1]
        obs_plane[:, 5] = obs_plane[:, 5] + receiver_mechanism[2]

    for i in tqdm(
            range(len(obs_plane)),
            desc="Computing dynamic Coulomb Failure Stress change at depth %f" % obs_depth,
    ):
        if config.optimal_type:
            cal_cfs_dynamic_single_point_fm(
                path_green=config.path_green_dynamic,
                source_array=source_array,
                obs_array_single_point=obs_plane[i, :],
                srate_stf=1 / config.sampling_interval_stf,
                mu_f=config.mu_f,
                B_pore=config.B_pore,
                max_slowness=config.max_slowness,
                green_info=green_info,
                path_results_each=path_results_each,
                use_spherical=config.use_spherical,
                check_finish=config.check_finished
            )
        else:
            cal_cfs_dynamic_single_point_oop(
                path_green=config.path_green_dynamic,
                source_array=source_array,
                obs_array_single_point=obs_plane[i, :],
                tectonic_stress_type=config.tectonic_stress_type,
                tectonic_stress=config.tectonic_stress,
                srate_stf=1 / config.sampling_interval_stf,
                mu_f=config.mu_f,
                max_slowness=config.max_slowness,
                green_info=green_info,
                path_results_each=path_results_each,
                use_spherical=config.use_spherical,
                check_finish=config.check_finished
            )
    read_stress_results_fix_depth(
        obs_plane=obs_plane,
        obs_depth=obs_depth,
        path_output=config.path_output,
        sampling_num=config.sampling_num,
        specified_orientation=config.optimal_type,
    )
    e = datetime.datetime.now()
    print("run time:", e - s)


def read_stress_results_fix_depth(
        obs_plane, obs_depth, path_output, sampling_num, specified_orientation
):
    print("Outputting results for depth %f km to csv files" % obs_depth)
    path_results_each = os.path.join(path_output, "grn_d", "results_each")
    path_output_results = os.path.join(path_output, "results", "dynamic")
    os.makedirs(path_output_results, exist_ok=True)

    if specified_orientation:
        file_info = [[item, "_%s.npy" % item, 1] for item in result_name_list]
        file_info[0][2] = 3
        file_info[1][2] = 3
    else:
        file_info1 = [[item + '1', "_%s1.npy" % item, 1] for item in result_name_list]
        file_info2 = [[item + '2', "_%s2.npy" % item, 1] for item in result_name_list]
        file_info1[0][2] = 3
        file_info1[1][2] = 3
        file_info2[0][2] = 3
        file_info2[1][2] = 3
        file_info = file_info1 + file_info2
    file_info = [["stress_ned", "_stress_ned.npy", 6], ["cfs", "_cfs.npy", 1]] + file_info

    data = {
        key: np.zeros((len(obs_plane) * col_num, sampling_num))
        for key, _, col_num in file_info
    }
    for i in range(len(obs_plane)):
        lat, lon, dep = obs_plane[i, 0:3]
        fname = "%.4f_%.4f_%.4f" % (lat, lon, dep)
        for key, suffix, col_num in file_info:
            arr = np.load(os.path.join(path_results_each, fname + suffix))
            data[key][i * col_num: (i + 1) * col_num] = arr.T

    for key, arr in data.items():
        if specified_orientation:
            out_name = "%s_dynamic_dep_%.2f.csv" % (key, obs_depth)
        else:
            out_name = "%s_dynamic_dep_%.2f.csv" % (key + "_oop", obs_depth)
        pd.DataFrame(arr).to_csv(
            os.path.join(path_output_results, out_name), header=False, index=False
        )


def run_all_dynamic(config: CfsConfig):
    create_dynamic_lib(config)
    compute_dynamic_cfs_sequential(config)


if __name__ == "__main__":
    pass
