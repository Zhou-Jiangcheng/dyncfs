import contextlib
import datetime
import hashlib
import math
import os
from typing import Union
import pickle
import json
from multiprocessing import get_context
import warnings

# import warnings
# warnings.filterwarnings('error', category=RuntimeWarning)

import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from pygrnwang.create_qssp2020_bulk import (
    pre_process_qssp2020,
    create_grnlib_qssp2020_parallel,
)
from pygrnwang.read_qssp2020 import seek_qssp2020
from pygrnwang.create_qseis2025_bulk import (
    pre_process_qseis2025,
    create_grnlib_qseis2025_parallel,
)
from pygrnwang.read_qseis2025 import seek_qseis2025
from pygrnwang.focal_mechanism import plane2nd, check_convert_fm, mt2plane
from pygrnwang.signal_process import resample

from .configuration import CfsConfig
from .signal_process import correct_zero_frequency
from .cfs_static import (
    cal_coulomb_failure_stress,
    cal_coulomb_failure_stress_poroelasticity,
)
from .utils import (
    read_source_array,
    bool2int,
    ignore_slip_source_array,
    cut_stf_modify_source_array,
    spherical_dist_azimuth_km,
    static_stress_ned2enz,
    cal_grid_num,
    orient_principal_axes,
)


def create_dynamic_lib(config: CfsConfig):
    s = datetime.datetime.now()
    if config.use_spherical:
        pre_process_qssp2020(
            processes_num=config.processes_num,
            path_green=config.path_green_dynamic,
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
            source_duration=config.wavelet_duration * config.sampling_interval_cfs,
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
            free_surface=1-bool2int(config.free_surface),
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
                        [lat(deg), lon(deg), depth(km)]
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
    dist_km_max = -np.inf
    for i in range(sub_faults_source.shape[0]):
        # print(i, sub_faults_source[i])
        sub_stf = resample(
            sub_stfs[i],
            srate_old=srate_stf,
            srate_new=srate_cfs,
            zero_phase=True,
        )
        m0_sub_stf = np.sum(sub_stf) / srate_cfs
        if m0_sub_stf != 0:
            sub_stf = sub_stf / m0_sub_stf * sub_m0s[i]
        else:
            continue
        dist_in_km, az_in_deg = spherical_dist_azimuth_km(
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
                dist_km=dist_in_km,
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
                interpolate_type=1,
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
                dist_km=dist_in_km,
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
                interpolate_type=1,
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
        if dist_km_max < dist_in_km:
            dist_km_max = dist_in_km

    # wavelet_duration in samples
    if use_spherical:
        wavelet_duration = round(green_info["source_duration"] * srate_cfs)
    else:
        wavelet_duration = green_info["wavelet_duration"]
    if max_slowness is not None and np.isfinite(tp_min):
        # all terms in samples
        tc2 = round(
            dist_km_max * max_slowness * srate_cfs
            + 1.5 * wavelet_duration
            + sub_stfs.shape[1] * srate_cfs / srate_stf
        )
        tc2 = min(tc2, sampling_num)
    if (static_stress is not None) and (max_slowness is not None) and np.isfinite(tp_min):
        tc1 = max(1, round(tp_min * srate_cfs - 1))
        stress_rate_enz = (
            signal.convolve(
                stress_enz, np.array([1, -1])[:, None], mode="same", method="auto"
            )
            * srate_cfs
        )
        if tc2 - tc1 > 1:
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
                stress_enz[:, i_cor] = np.cumsum(stress_rate_enz[:, i_cor]) / srate_cfs
    elif max_slowness is not None and np.isfinite(tp_min) and tc2 < sampling_num:
        i0 = min(tc2 + round(10 * srate_cfs), sampling_num - 1)
        i1 = min(tc2 + round(20 * srate_cfs), sampling_num)
        final_values = np.mean(stress_enz[i0:i1, :], axis=0)
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


def stress_cache_signature(
    path_green,
    source_array,
    obs_array_single_point,
    srate_stf,
    static_stress,
    max_slowness,
    green_info,
    use_spherical,
):
    """
    Parameters that determine the synthesized dynamic stress of one obs point.
    The cached *_stress_ned.npy is only reused when its signature is identical.
    """
    source_array = np.ascontiguousarray(source_array, dtype=np.float64)
    return {
        "path_green": os.path.abspath(path_green),
        "green_info_sha1": hashlib.sha1(
            json.dumps(green_info, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "source_array_sha1": hashlib.sha1(source_array.tobytes()).hexdigest()
        + "_%dx%d" % source_array.shape,
        "obs_point": [float(v) for v in np.asarray(obs_array_single_point)[:3]],
        "srate_stf": float(srate_stf),
        "static_stress": None
        if static_stress is None
        else [float(v) for v in np.asarray(static_stress).ravel()],
        "max_slowness": None if max_slowness is None else float(max_slowness),
        "use_spherical": bool(use_spherical),
    }


def load_or_synthesize_stress_ned(
    path_green,
    source_array,
    obs_array_single_point,
    srate_stf,
    static_stress,
    max_slowness,
    green_info,
    use_spherical,
    path_results_each,
    file_name,
    check_finish,
):
    """
    Synthesize the dynamic stress in NED axis [nn, ne, nd, ee, ed, dd], or load it
    from path_results_each when check_finish is True and the cached result was
    computed with the same parameters (see stress_cache_signature).
    The synthesized stress and its signature are saved to path_results_each.
    """
    if path_results_each is None:
        return synthesize_dynamic_stress(
            path_green=path_green,
            source_array=source_array,
            obs_array_single_point=obs_array_single_point,
            srate_stf=srate_stf,
            static_stress=static_stress,
            max_slowness=max_slowness,
            green_info=green_info,
            use_spherical=use_spherical,
        )
    signature = stress_cache_signature(
        path_green,
        source_array,
        obs_array_single_point,
        srate_stf,
        static_stress,
        max_slowness,
        green_info,
        use_spherical,
    )
    path_stress_ned = os.path.join(path_results_each, file_name + "_stress_ned.npy")
    path_signature = os.path.join(path_results_each, file_name + "_stress_ned.json")
    if check_finish and os.path.exists(path_stress_ned) and os.path.exists(path_signature):
        try:
            with open(path_signature, "r") as fr:
                cached_signature = json.load(fr)
        except (OSError, ValueError):
            cached_signature = None
        if cached_signature == signature:
            return np.load(path_stress_ned)
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
    np.save(path_stress_ned, stress_ned)
    with open(path_signature, "w") as fw:
        json.dump(signature, fw)
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
    print("obs point:", obs_array_single_point)
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

    stress_ned = load_or_synthesize_stress_ned(
        path_green=path_green,
        source_array=source_array,
        obs_array_single_point=obs_array_single_point,
        srate_stf=srate_stf,
        static_stress=static_stress,
        max_slowness=max_slowness,
        green_info=green_info,
        use_spherical=use_spherical,
        path_results_each=path_results_each,
        file_name=file_name,
        check_finish=check_finish,
    )

    n_obs, d_obs = plane2nd(*obs_array_single_point[3:])
    n = np.array([n_obs.flatten()]).T
    d = np.array([d_obs.flatten()]).T
    sigma_vector = cal_stress_vector_ned_dynamic(stress_ned, n)
    sigma = np.dot(sigma_vector, n).flatten()
    tau = np.dot(sigma_vector, d).flatten()
    if B_pore == 0:
        cfs = cal_coulomb_failure_stress(norm_stress=sigma, shear_stress=tau, mu_f=mu_f)
    else:
        mean_stress = (stress_ned[:, 0] + stress_ned[:, 3] + stress_ned[:, 5]) / 3
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
            cfs,
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
        cfs,
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
    print("obs point:", obs_array_single_point)
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

    # Synthesize (or load cached) dynamic stress in NED axis: [nn, ne, nd, ee, ed, dd]
    stress_ned = load_or_synthesize_stress_ned(
        path_green=path_green,
        source_array=source_array,
        obs_array_single_point=obs_array_single_point,
        srate_stf=srate_stf,
        static_stress=static_stress,
        max_slowness=max_slowness,
        green_info=green_info,
        use_spherical=use_spherical,
        path_results_each=path_results_each,
        file_name=file_name,
        check_finish=check_finish,
    )

    N = stress_ned.shape[0]
    tectonic_stress = np.asarray(tectonic_stress, dtype=float).reshape(6)
    # Total stress at each time = dynamic + tectonic (Voigt 6 components)
    stress_total = stress_ned + tectonic_stress

    # Get plane normal n and two in-plane orthonormal directions via plane2nd:
    # - rake = 0° -> strike-parallel slip direction (basis u)
    # - rake = 90° -> down-dip slip direction (basis v)
    n0, d_r0 = plane2nd(
        float(obs_array_single_point[3]),  # strike (deg)
        float(obs_array_single_point[4]),  # dip (deg)
        0.0,  # rake=0°
    )
    _, d_r90 = plane2nd(
        float(obs_array_single_point[3]),
        float(obs_array_single_point[4]),
        90.0,  # rake=90°
    )
    n = n0.flatten()  # (3,)
    u = d_r0.flatten()  # (3,) - along strike
    v = d_r90.flatten()  # (3,) - along dip

    # Optimal slip direction from the total stress (dynamic + tectonic):
    # shear traction on the plane s = t - (n·t)*n, with t = S_total * n
    sigma_vec_total = cal_stress_vector_ned_dynamic(stress_total, n)  # (N,3)
    sigma_n_total = np.einsum("ij,j->i", sigma_vec_total, n)  # (N,)
    s_vec = sigma_vec_total - sigma_n_total[:, None] * n[None, :]  # (N,3)
    s_norm = np.linalg.norm(s_vec, axis=1)  # (N,)
    # Avoid division by zero
    eps = 1e-20
    d_opt = s_vec / (s_norm[:, None] + eps)  # (N,3)

    # Stress changes (earthquake-induced only) projected on n and d_opt,
    # consistent with cal_cfs_static_single_point_opt_rake
    sigma_vec = cal_stress_vector_ned_dynamic(stress_ned, n)  # (N,3)
    sigma_n = np.einsum("ij,j->i", sigma_vec, n)  # (N,)
    tau = np.einsum("ij,ij->i", sigma_vec, d_opt)  # (N,)

    if B_pore == 0:
        cfs = cal_coulomb_failure_stress(
            norm_stress=sigma_n, shear_stress=tau, mu_f=mu_f
        )
    else:
        mean_stress = (stress_ned[:, 0] + stress_ned[:, 3] + stress_ned[:, 5]) / 3.0
        cfs = cal_coulomb_failure_stress_poroelasticity(
            norm_stress=sigma_n,
            shear_stress=tau,
            mean_stress=mean_stress,
            mu_f=mu_f,
            B_pore=B_pore,
        )

    # Optimal rake angle (deg): d_opt = cos(rake)*u + sin(rake)*v  =>  rake = atan2(d·v, d·u)
    du = np.einsum("ij,j->i", d_opt, u)  # (N,)
    dv = np.einsum("ij,j->i", d_opt, v)  # (N,)
    rake_rad = np.arctan2(dv, du)
    rake_deg = np.rad2deg(rake_rad)

    if path_results_each is not None:
        results = (
            stress_ned,  # keep the raw dynamic part for _stress_ned.npy
            np.ones((N, 3)) * n[None, :],  # normal_vector
            d_opt,  # rupture_vector (time-varying, optimal)
            sigma_n,  # normal_stress
            tau,  # shear_stress
        )
        np.save(
            str(
                os.path.join(
                    path_results_each,
                    file_name + "_os_cfs.npy",
                )
            ),
            cfs,
        )
        for j in range(len(result_name_list)):
            np.save(
                str(
                    os.path.join(
                        path_results_each,
                        file_name + "_os_%s.npy" % result_name_list[j],
                    )
                ),
                results[j + 1],
            )
        np.save(os.path.join(path_results_each, file_name + "_os_rake.npy"), rake_deg)

    return (
        stress_ned,
        np.ones((N, 3)) * n[None, :],
        d_opt,
        sigma_n,
        tau,
        cfs,
        rake_deg,
    )


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
                        [lat(deg), lon(deg), depth(km)]
    :param srate_stf: Sampling rate of stf in Hz.
    :param static_stress: Static stress tensor used for correcting zero-frequency values.
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
    print("obs point:", obs_array_single_point)
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

    stress_ned = load_or_synthesize_stress_ned(
        path_green=path_green,
        source_array=source_array,
        obs_array_single_point=obs_array_single_point,
        srate_stf=srate_stf,
        static_stress=static_stress,
        max_slowness=max_slowness,
        green_info=green_info,
        use_spherical=use_spherical,
        path_results_each=path_results_each,
        file_name=file_name,
        check_finish=check_finish,
    )

    N = stress_ned.shape[0]

    # --- Stack (N,6) components [nn,ne,nd,ee,ed,dd] into symmetric tensors (N,3,3) ---
    # sigma_nn, sigma_ne, sigma_nd, sigma_ee, sigma_ed, sigma_dd  (N,6)
    s_nn = stress_ned[:, 0]
    s_ne = stress_ned[:, 1]
    s_nd = stress_ned[:, 2]
    s_ee = stress_ned[:, 3]
    s_ed = stress_ned[:, 4]
    s_dd = stress_ned[:, 5]

    sigma = np.empty((N, 3, 3), dtype=stress_ned.dtype)
    sigma[:, 0, 0] = s_nn
    sigma[:, 0, 1] = s_ne
    sigma[:, 0, 2] = s_nd
    sigma[:, 1, 0] = s_ne
    sigma[:, 1, 1] = s_ee
    sigma[:, 1, 2] = s_ed
    sigma[:, 2, 0] = s_nd
    sigma[:, 2, 1] = s_ed
    sigma[:, 2, 2] = s_dd

    # Build tectonic stress tensor T (3x3) or principal-axis matrix R
    if tectonic_stress_type == 1:
        # tectonic_stress = [s_nn, s_ne, s_nd, s_ee, s_ed, s_dd] (Pa)
        t = np.asarray(tectonic_stress, dtype=float)
        T = np.array(
            [[t[0], t[1], t[2]], [t[1], t[3], t[4]], [t[2], t[4], t[5]]], dtype=float
        )
        # Batched eigendecomposition for S = sigma + T
        S = sigma + T  # broadcast to (N,3,3)
        # S is symmetric: batched eigh returns real, ascending eigenvalues
        evals, evecs = np.linalg.eigh(S)  # evals: (N,3), evecs: (N,3,3)
        # Descending order; columns are principal directions. Remove the arbitrary
        # eigenvector signs so that plane 1 / plane 2 are labeled reproducibly.
        R = orient_principal_axes(evecs[:, :, ::-1])
    elif tectonic_stress_type == 2:
        # tectonic_stress = [az1, pl1, az2, pl2, az3, pl3] (deg)
        ts = np.asarray(tectonic_stress, dtype=float)
        R_const = np.zeros((3, 3), dtype=float)
        for i in range(3):
            phi = np.deg2rad(ts[2 * i])  # azimuth
            delta = np.deg2rad(ts[2 * i + 1])  # plunge
            R_const[:, i] = [
                np.cos(phi) * np.cos(delta),
                np.sin(phi) * np.cos(delta),
                np.sin(delta),
            ]
        R_const = R_const[
            :, ::-1
        ]  # keep the same column order convention as the static implementation
        R = np.broadcast_to(R_const, (N, 3, 3)).copy()
    else:
        raise ValueError("tectonic_stress_type must be 1 or 2")

    # Two candidate fault planes in principal-stress coordinates
    # closed-form as in the static version
    theta = (np.pi / 4.0) if (mu_f == 0) else 0.5 * np.arctan(1.0 / mu_f)
    n_prin1 = np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=float)
    d_prin1 = np.array([np.sin(theta), 0.0, -np.cos(theta)], dtype=float)
    n_prin2 = np.array([np.cos(theta), 0.0, -np.sin(theta)], dtype=float)
    d_prin2 = np.array([np.sin(theta), 0.0, np.cos(theta)], dtype=float)

    # Rotate to NED coordinates in batch
    # (N,3,3) @ (3,) -> (N,3)
    n1 = R @ n_prin1
    d1 = R @ d_prin1
    n2 = R @ n_prin2
    d2 = R @ d_prin2

    # If the z-component of normal is positive (upward), flip both normal and slip to keep the same orientation rule as the static version
    flip1 = n1[:, 2] > 0.0
    n1[flip1] = -n1[flip1]
    d1[flip1] = -d1[flip1]
    flip2 = n2[:, 2] > 0.0
    n2[flip2] = -n2[flip2]
    d2[flip2] = -d2[flip2]

    # Batched stress projections onto normal/slip directions
    sigvec1 = np.einsum("nij,nj->ni", sigma, n1)  # (N,3)
    sigvec2 = np.einsum("nij,nj->ni", sigma, n2)

    sigma1 = np.einsum("ni,ni->n", sigvec1, n1)  # (N,)
    tau1 = np.einsum("ni,ni->n", sigvec1, d1)  # (N,)

    sigma2 = np.einsum("ni,ni->n", sigvec2, n2)
    tau2 = np.einsum("ni,ni->n", sigvec2, d2)

    if B_pore == 0:
        cfs = cal_coulomb_failure_stress(
            norm_stress=sigma2, shear_stress=tau2, mu_f=mu_f
        )
    else:
        mean_stress = (sigma[:, 0, 0] + sigma[:, 1, 1] + sigma[:, 2, 2]) / 3.0
        cfs = cal_coulomb_failure_stress_poroelasticity(
            norm_stress=sigma2,
            shear_stress=tau2,
            mean_stress=mean_stress,
            mu_f=mu_f,
            B_pore=B_pore,
        )

    if path_results_each is not None:
        results = (
            stress_ned,
            [n1, d1, sigma1, tau1],
            [n2, d2, sigma2, tau2],
            cfs,
        )
        np.save(os.path.join(path_results_each, file_name + "_oop_cfs.npy"), cfs)
        for jj in range(2):
            for j, key in enumerate(result_name_list):
                np.save(
                    os.path.join(
                        path_results_each, f"{file_name}_oop_{key}{jj + 1}.npy"
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


def load_static_stress_enz(config: CfsConfig, file_name: str, n_points: int):
    """
    Load the static stress tensors computed by cfs_static (NED axis) and convert
    them to ENZ axis for the zero-frequency correction.
    Return None if config.correct_zero_freq is False.
    """
    if not config.correct_zero_freq:
        return None
    if config.max_slowness is None:
        warnings.warn(
            "correct_zero_freq is True but max_slowness is None, "
            "the zero-frequency correction will not be applied."
        )
    path_static_stress = os.path.join(config.path_output_results_static, file_name)
    if not os.path.exists(path_static_stress):
        raise FileNotFoundError(
            "%s not found, compute static stress before the dynamic stress "
            "when correct_zero_freq is True." % path_static_stress
        )
    stress_ned = np.load(path_static_stress)
    if stress_ned.shape[0] != n_points:
        raise ValueError(
            "%s has %d points, but %d obs points are used in dynamic computing."
            % (path_static_stress, stress_ned.shape[0], n_points)
        )
    return static_stress_ned2enz(stress_ned)


def prepare_compute_cfs(config: CfsConfig):
    source_array = read_source_array(
        source_inds=config.source_inds,
        path_input=config.path_input,
    )
    if config.slip_thresh > 0:
        source_array = ignore_slip_source_array(source_array, config.slip_thresh)
    if config.cut_stf > 0:
        source_array = cut_stf_modify_source_array(source_array, config.cut_stf)
    with open(
        os.path.join(config.path_green_dynamic, "green_lib_info.json"), "r"
    ) as fr:
        green_info = json.load(fr)
    path_results_each = os.path.join(config.path_output, "grn_d", "results_each")
    os.makedirs(path_results_each, exist_ok=True)
    np.save(os.path.join(path_results_each, "source_array.npy"), source_array)
    for ind_obs in config.obs_inds:
        obs_plane = pd.read_csv(
            str(os.path.join(config.path_input, "obs_plane%d.csv" % ind_obs)),
            index_col=False,
            header=None,
        ).to_numpy()
        static_stress_obs = load_static_stress_enz(
            config, "stress_tensor_plane%d.npy" % ind_obs, len(obs_plane)
        )
        if config.optimal_type == 0:
            inp_list = []
            for i in range(len(obs_plane)):
                inp = [
                    config.path_green_dynamic,
                    obs_plane[i, :6],
                    1 / config.sampling_interval_stf,
                    config.mu_f,
                    config.B_pore,
                    config.max_slowness,
                    green_info,
                    path_results_each,
                    config.use_spherical,
                    None if static_stress_obs is None else static_stress_obs[i],
                ]
                inp_list.append(inp)
            with open(
                os.path.join(path_results_each, "group_list_%d.pkl" % ind_obs), "wb"
            ) as fw:
                pickle.dump(inp_list, fw)  # type: ignore
        elif config.optimal_type == 1:
            inp_list = []
            for i in range(len(obs_plane)):
                inp = [
                    config.path_green_dynamic,
                    obs_plane[i, :5],
                    1 / config.sampling_interval_stf,
                    config.tectonic_stress,
                    config.mu_f,
                    config.B_pore,
                    config.max_slowness,
                    green_info,
                    path_results_each,
                    config.use_spherical,
                    None if static_stress_obs is None else static_stress_obs[i],
                ]
                inp_list.append(inp)
            with open(
                os.path.join(path_results_each, "group_list_%d.pkl" % ind_obs), "wb"
            ) as fw:
                pickle.dump(inp_list, fw)  # type: ignore
        elif config.optimal_type == 2:
            inp_list = []
            for i in range(len(obs_plane)):
                inp = [
                    config.path_green_dynamic,
                    obs_plane[i, :3],
                    1 / config.sampling_interval_stf,
                    config.tectonic_stress_type,
                    config.tectonic_stress,
                    config.mu_f,
                    config.B_pore,
                    config.max_slowness,
                    green_info,
                    path_results_each,
                    config.use_spherical,
                    None if static_stress_obs is None else static_stress_obs[i],
                ]
                inp_list.append(inp)
            with open(
                os.path.join(path_results_each, "group_list_%d.pkl" % ind_obs), "wb"
            ) as fw:
                pickle.dump(inp_list, fw)  # type: ignore


def build_parallel_jobs(input_list, path_source_array, optimal_type, check_finished):
    """
    Convert the pickled input lists into positional args of the target function.
    Each input list ends with `static_stress`, the targets take `check_finish`
    as the next positional argument.
    """
    jobs = []
    for args in input_list:
        # [path_green, source_array, ..., static_stress, check_finish]
        args = [args[0], path_source_array] + args[1:] + [check_finished]
        jobs.append(args)
    return jobs


# Thread-count variables of OpenMP, Intel MKL, OpenBLAS and Apple Accelerate
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@contextlib.contextmanager
def limit_worker_threads(num_threads: int = 1):
    """
    Set the thread-count environment variables while worker processes are started.

    BLAS/OpenMP libraries read these variables when numpy is imported, so they only
    take effect in newly spawned child processes, not in the current process.
    The original values are restored on exit, so they do not leak into later code
    or subprocesses.
    """
    old_values = {key: os.environ.get(key) for key in THREAD_ENV_VARS}
    for key in THREAD_ENV_VARS:
        os.environ[key] = str(num_threads)
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_parallel_jobs(target, jobs, processes_num: int):
    """
    Run target(*job) for every job in a pool of spawned worker processes.

    - The "spawn" start method is used through a local context, the global start
      method of the program is not changed. "spawn" is the only method on Windows
      and the default on macOS. On Linux it avoids forking a parent whose BLAS/OpenMP
      thread pools are already initialized: a forked child keeps all these threads
      (the thread-count variables are ignored) and some OpenMP runtimes may hang.
    - Each worker uses one BLAS/OpenMP thread to avoid oversubscription.
    - Workers are reused for all chunks, so numpy/scipy/pygrnwang are imported only
      once per worker.

    Scripts calling this function must be protected by `if __name__ == "__main__":`.
    """
    if len(jobs) == 0:
        return
    processes_num = max(1, min(int(processes_num), len(jobs)))
    chunksize = max(1, math.ceil(len(jobs) / (processes_num * 4)))
    ctx = get_context("spawn")
    with limit_worker_threads(1):
        with ctx.Pool(processes=processes_num) as pool:
            pool.starmap(target, jobs, chunksize=chunksize)


def compute_dynamic_cfs_parallel(config: CfsConfig):
    s = datetime.datetime.now()

    prepare_compute_cfs(config)
    path_results_each = str(os.path.join(config.path_output, "grn_d", "results_each"))
    path_source_array = str(os.path.join(path_results_each, "source_array.npy"))
    for ind_obs in config.obs_inds:
        print("Parallel computing cfs on Plane No.%d" % ind_obs)
        print("This process may take a long time.")
        with open(
            os.path.join(path_results_each, f"group_list_{ind_obs}.pkl"), "rb"
        ) as fr:
            input_list = pickle.load(fr)

        jobs = build_parallel_jobs(
            input_list, path_source_array, config.optimal_type, config.check_finished
        )

        if config.optimal_type == 0:
            target = cal_cfs_dynamic_single_point_fm
        elif config.optimal_type == 1:
            target = cal_cfs_dynamic_single_point_opt_rake
        elif config.optimal_type == 2:
            target = cal_cfs_dynamic_single_point_oop
        else:
            raise ValueError("optimal_type must be 0/1/2")

        run_parallel_jobs(target, jobs, config.processes_num)

        obs_plane = pd.read_csv(
            os.path.join(config.path_input, f"obs_plane{ind_obs}.csv"),
            index_col=False,
            header=None,
        ).to_numpy()
        read_stress_results(
            config=config,
            obs_plane=obs_plane,
            ind_obs=ind_obs,
        )

    e = datetime.datetime.now()
    print("run time:", e - s)


def compute_dynamic_cfs_sequential(config: CfsConfig):
    s = datetime.datetime.now()
    source_array = read_source_array(
        source_inds=config.source_inds,
        path_input=config.path_input,
    )
    if config.slip_thresh > 0:
        source_array = ignore_slip_source_array(source_array, config.slip_thresh)
    if config.cut_stf > 0:
        source_array = cut_stf_modify_source_array(source_array, config.cut_stf)
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
        static_stress_obs = load_static_stress_enz(
            config, "stress_tensor_plane%d.npy" % ind_obs, len(obs_plane)
        )

        if config.optimal_type == 0:
            for i in tqdm(
                range(len(obs_plane)),
                desc="Computing dynamic Coulomb Failure Stress change at No.%d plane"
                "(fixed focal mechanism)" % ind_obs,
            ):
                static_stress = None if static_stress_obs is None else static_stress_obs[i]
                cal_cfs_dynamic_single_point_fm(
                    path_green=config.path_green_dynamic,
                    source_array=source_array,
                    obs_array_single_point=obs_plane[i, :6],
                    srate_stf=1 / config.sampling_interval_stf,
                    mu_f=config.mu_f,
                    B_pore=config.B_pore,
                    static_stress=static_stress,
                    max_slowness=config.max_slowness,
                    green_info=green_info,
                    path_results_each=path_results_each,
                    use_spherical=config.use_spherical,
                    check_finish=config.check_finished,
                )
        elif config.optimal_type == 1:
            for i in tqdm(
                range(len(obs_plane)),
                desc="Computing dynamic Coulomb Failure Stress change at No.%d plane"
                "(optimal rake)" % ind_obs,
            ):
                cal_cfs_dynamic_single_point_opt_rake(
                    path_green=config.path_green_dynamic,
                    source_array=source_array,
                    obs_array_single_point=obs_plane[i, :5],
                    srate_stf=1 / config.sampling_interval_stf,
                    tectonic_stress=config.tectonic_stress,
                    mu_f=config.mu_f,
                    B_pore=config.B_pore,
                    static_stress=None if static_stress_obs is None else static_stress_obs[i],
                    max_slowness=config.max_slowness,
                    green_info=green_info,
                    path_results_each=path_results_each,
                    use_spherical=config.use_spherical,
                    check_finish=config.check_finished,
                )
        elif config.optimal_type == 2:
            for i in tqdm(
                range(len(obs_plane)),
                desc="Computing dynamic Coulomb Failure Stress change at No.%d plane"
                "(optimally oriented planes)" % ind_obs,
            ):
                cal_cfs_dynamic_single_point_oop(
                    path_green=config.path_green_dynamic,
                    source_array=source_array,
                    obs_array_single_point=obs_plane[i, :3],
                    srate_stf=1 / config.sampling_interval_stf,
                    tectonic_stress_type=config.tectonic_stress_type,
                    tectonic_stress=config.tectonic_stress,
                    mu_f=config.mu_f,
                    B_pore=config.B_pore,
                    static_stress=None if static_stress_obs is None else static_stress_obs[i],
                    max_slowness=config.max_slowness,
                    green_info=green_info,
                    path_results_each=path_results_each,
                    use_spherical=config.use_spherical,
                    check_finish=config.check_finished,
                )
        read_stress_results(
            config=config,
            obs_plane=obs_plane,
            ind_obs=ind_obs,
        )
    e = datetime.datetime.now()
    print("run time:", e - s)


result_name_list = [
    "normal_vector",
    "rupture_vector",
    "normal_stress",
    "shear_stress",
]


def read_stress_results(config: CfsConfig, obs_plane, ind_obs):
    print("Outputting results for No.%d obs_plane to csv files" % ind_obs)
    path_results_each = os.path.join(config.path_output, "grn_d", "results_each")
    path_output_results = config.path_output_results_dynamic
    os.makedirs(path_output_results, exist_ok=True)

    if config.optimal_type == 0:
        file_info = [[item, "_%s.npy" % item, 1] for item in result_name_list]
        file_info[0][2] = 3
        file_info[1][2] = 3
        file_info = file_info + [["cfs", "_cfs.npy", 1]]
    elif config.optimal_type == 1:
        file_info = [[item, "_os_%s.npy" % item, 1] for item in result_name_list]
        file_info[0][2] = 3
        file_info[1][2] = 3
        file_info = file_info + [["rake", "_os_rake.npy", 1], ["cfs", "_os_cfs.npy", 1]]
    elif config.optimal_type == 2:
        file_info1 = [
            [item + "1", "_oop_%s1.npy" % item, 1] for item in result_name_list
        ]
        file_info2 = [
            [item + "2", "_oop_%s2.npy" % item, 1] for item in result_name_list
        ]
        file_info1[0][2] = 3
        file_info1[1][2] = 3
        file_info2[0][2] = 3
        file_info2[1][2] = 3
        file_info = file_info1 + file_info2 + [["cfs", "_oop_cfs.npy", 1]]
    else:
        raise ValueError("optimal_type must be 0/1/2")
    file_info = [["stress_ned", "_stress_ned.npy", 6]] + file_info
    data = {
        key: np.zeros((len(obs_plane) * col_num, config.sampling_num))
        for key, _, col_num in file_info
    }
    for i in range(len(obs_plane)):
        lat, lon, dep = obs_plane[i, 0:3]
        fname = "%.4f_%.4f_%.4f" % (lat, lon, dep)
        for key, suffix, col_num in file_info:
            arr = np.load(os.path.join(path_results_each, fname + suffix))
            data[key][i * col_num : (i + 1) * col_num] = arr.T

    for key, arr in data.items():
        if config.optimal_type == 0:
            out_name = "%s_dynamic_plane%d.csv" % (key, ind_obs)
        elif config.optimal_type == 1:
            out_name = "%s_dynamic_plane%d.csv" % (key + "_os", ind_obs)
        elif config.optimal_type == 2:
            out_name = "%s_dynamic_plane%d.csv" % (key + "_oop", ind_obs)
        else:
            raise ValueError("optimal_type must be 0/1/2")
        pd.DataFrame(arr).to_csv(
            os.path.join(path_output_results, out_name), header=False, index=False
        )


def prepare_compute_cfs_fix_depth(
    config: CfsConfig, obs_depth: float = None, receiver_mechanism=None
):
    if obs_depth is None:
        obs_depth = config.fixed_obs_depth
    source_array = read_source_array(
        source_inds=config.source_inds,
        path_input=config.path_input,
    )
    if config.slip_thresh > 0:
        source_array = ignore_slip_source_array(source_array, config.slip_thresh)
    if config.cut_stf > 0:
        source_array = cut_stf_modify_source_array(source_array, config.cut_stf)
    if config.optimal_type in (0, 1) and receiver_mechanism is None:
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

    Nx = cal_grid_num(config.obs_lat_range, config.obs_delta_lat)
    Ny = cal_grid_num(config.obs_lon_range, config.obs_delta_lon)

    obs_plane = np.zeros((Nx * Ny, 6))
    obs_plane[:, 2] = obs_plane[:, 2] + obs_depth
    lat_array = np.linspace(config.obs_lat_range[0], config.obs_lat_range[1], Nx)
    lon_array = np.linspace(config.obs_lon_range[0], config.obs_lon_range[1], Ny)
    for i in range(Nx):
        for j in range(Ny):
            ind = j + i * Ny
            obs_plane[ind, :2] = np.array([lat_array[i], lon_array[j]])
    if config.optimal_type == 0:
        obs_plane[:, 3] = obs_plane[:, 3] + receiver_mechanism[0]
        obs_plane[:, 4] = obs_plane[:, 4] + receiver_mechanism[1]
        obs_plane[:, 5] = obs_plane[:, 5] + receiver_mechanism[2]
    elif config.optimal_type == 1:
        obs_plane[:, 3] = obs_plane[:, 3] + receiver_mechanism[0]
        obs_plane[:, 4] = obs_plane[:, 4] + receiver_mechanism[1]
        obs_plane = obs_plane[:, :5]
    elif config.optimal_type == 2:
        obs_plane = obs_plane[:, :3]

    with open(
        os.path.join(config.path_green_dynamic, "green_lib_info.json"), "r"
    ) as fr:
        green_info = json.load(fr)
    path_results_each = os.path.join(config.path_output, "grn_d", "results_each")
    os.makedirs(path_results_each, exist_ok=True)
    np.save(os.path.join(path_results_each, "source_array.npy"), source_array)
    np.save(
        os.path.join(path_results_each, "obs_plane_%.2f.npy" % obs_depth), obs_plane
    )
    static_stress_obs = load_static_stress_enz(
        config, "stress_tensor_dep_%.2f.npy" % obs_depth, len(obs_plane)
    )

    if config.optimal_type == 0:
        inp_list = []
        for i in range(len(obs_plane)):
            inp = [
                config.path_green_dynamic,
                obs_plane[i, :6],
                1 / config.sampling_interval_stf,
                config.mu_f,
                config.B_pore,
                config.max_slowness,
                green_info,
                path_results_each,
                config.use_spherical,
                None if static_stress_obs is None else static_stress_obs[i],
            ]
            inp_list.append(inp)
        with open(
            os.path.join(path_results_each, "group_list_%.2f.pkl" % obs_depth), "wb"
        ) as fw:
            pickle.dump(inp_list, fw)  # type: ignore
    elif config.optimal_type == 1:
        inp_list = []
        for i in range(len(obs_plane)):
            inp = [
                config.path_green_dynamic,
                obs_plane[i, :5],
                1 / config.sampling_interval_stf,
                config.tectonic_stress,
                config.mu_f,
                config.B_pore,
                config.max_slowness,
                green_info,
                path_results_each,
                config.use_spherical,
                None if static_stress_obs is None else static_stress_obs[i],
            ]
            inp_list.append(inp)
        with open(
            os.path.join(path_results_each, "group_list_%.2f.pkl" % obs_depth), "wb"
        ) as fw:
            pickle.dump(inp_list, fw)  # type: ignore
    elif config.optimal_type == 2:
        inp_list = []
        for i in range(len(obs_plane)):
            inp = [
                config.path_green_dynamic,
                obs_plane[i, :3],
                1 / config.sampling_interval_stf,
                config.tectonic_stress_type,
                config.tectonic_stress,
                config.mu_f,
                config.B_pore,
                config.max_slowness,
                green_info,
                path_results_each,
                config.use_spherical,
                None if static_stress_obs is None else static_stress_obs[i],
            ]
            inp_list.append(inp)
        with open(
            os.path.join(path_results_each, "group_list_%.2f.pkl" % obs_depth), "wb"
        ) as fw:
            pickle.dump(inp_list, fw)  # type: ignore


def compute_dynamic_cfs_fix_depth_parallel(
    config: CfsConfig, obs_depth: float = None, receiver_mechanism=None
):
    s = datetime.datetime.now()
    if obs_depth is None:
        obs_depth = config.fixed_obs_depth

    prepare_compute_cfs_fix_depth(config, obs_depth, receiver_mechanism)
    path_results_each = str(os.path.join(config.path_output, "grn_d", "results_each"))
    path_source_array = str(os.path.join(path_results_each, "source_array.npy"))
    print("Parallel computing cfs on  %.2f km depth" % obs_depth)
    print("This process may take a long time.")
    with open(
        os.path.join(path_results_each, f"group_list_%.2f.pkl" % obs_depth), "rb"
    ) as fr:
        input_list = pickle.load(fr)

    jobs = build_parallel_jobs(
        input_list, path_source_array, config.optimal_type, config.check_finished
    )

    if config.optimal_type == 0:
        target = cal_cfs_dynamic_single_point_fm
    elif config.optimal_type == 1:
        target = cal_cfs_dynamic_single_point_opt_rake
    elif config.optimal_type == 2:
        target = cal_cfs_dynamic_single_point_oop
    else:
        raise ValueError("optimal_type must be 0/1/2")

    run_parallel_jobs(target, jobs, config.processes_num)

    obs_plane = np.load(
        os.path.join(path_results_each, "obs_plane_%.2f.npy" % obs_depth)
    )
    read_stress_results_fix_depth(
        config=config,
        obs_plane=obs_plane,
        obs_depth=obs_depth,
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
        source_array = ignore_slip_source_array(source_array, config.slip_thresh)
    if config.cut_stf > 0:
        source_array = cut_stf_modify_source_array(source_array, config.cut_stf)
    if config.optimal_type in (0, 1) and receiver_mechanism is None:
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

    Nx = cal_grid_num(obs_lat_range, obs_delta_lat)
    Ny = cal_grid_num(obs_lon_range, obs_delta_lon)

    obs_plane = np.zeros((Nx * Ny, 6))
    obs_plane[:, 2] = obs_plane[:, 2] + obs_depth
    lat_array = np.linspace(obs_lat_range[0], obs_lat_range[1], Nx)
    lon_array = np.linspace(obs_lon_range[0], obs_lon_range[1], Ny)
    for i in range(Nx):
        for j in range(Ny):
            ind = j + i * Ny
            obs_plane[ind, :2] = np.array([lat_array[i], lon_array[j]])
    if config.optimal_type == 0:
        obs_plane[:, 3] = obs_plane[:, 3] + receiver_mechanism[0]
        obs_plane[:, 4] = obs_plane[:, 4] + receiver_mechanism[1]
        obs_plane[:, 5] = obs_plane[:, 5] + receiver_mechanism[2]
    elif config.optimal_type == 1:
        obs_plane[:, 3] = obs_plane[:, 3] + receiver_mechanism[0]
        obs_plane[:, 4] = obs_plane[:, 4] + receiver_mechanism[1]

    static_stress_obs = load_static_stress_enz(
        config, "stress_tensor_dep_%.2f.npy" % obs_depth, len(obs_plane)
    )

    for i in tqdm(
        range(len(obs_plane)),
        desc="Computing dynamic Coulomb Failure Stress change at depth %f" % obs_depth,
    ):
        static_stress = None if static_stress_obs is None else static_stress_obs[i]
        if config.optimal_type == 0:
            cal_cfs_dynamic_single_point_fm(
                path_green=config.path_green_dynamic,
                source_array=source_array,
                obs_array_single_point=obs_plane[i, :6],
                srate_stf=1 / config.sampling_interval_stf,
                mu_f=config.mu_f,
                B_pore=config.B_pore,
                static_stress=static_stress,
                max_slowness=config.max_slowness,
                green_info=green_info,
                path_results_each=path_results_each,
                use_spherical=config.use_spherical,
                check_finish=config.check_finished,
            )
        elif config.optimal_type == 1:
            cal_cfs_dynamic_single_point_opt_rake(
                path_green=config.path_green_dynamic,
                source_array=source_array,
                obs_array_single_point=obs_plane[i, :5],
                tectonic_stress=config.tectonic_stress,
                srate_stf=1 / config.sampling_interval_stf,
                mu_f=config.mu_f,
                B_pore=config.B_pore,
                static_stress=static_stress,
                max_slowness=config.max_slowness,
                green_info=green_info,
                path_results_each=path_results_each,
                use_spherical=config.use_spherical,
                check_finish=config.check_finished,
            )
        elif config.optimal_type == 2:
            cal_cfs_dynamic_single_point_oop(
                path_green=config.path_green_dynamic,
                source_array=source_array,
                obs_array_single_point=obs_plane[i, :3],
                tectonic_stress_type=config.tectonic_stress_type,
                tectonic_stress=config.tectonic_stress,
                srate_stf=1 / config.sampling_interval_stf,
                mu_f=config.mu_f,
                B_pore=config.B_pore,
                static_stress=static_stress,
                max_slowness=config.max_slowness,
                green_info=green_info,
                path_results_each=path_results_each,
                use_spherical=config.use_spherical,
                check_finish=config.check_finished,
            )
    read_stress_results_fix_depth(
        config=config,
        obs_plane=obs_plane,
        obs_depth=obs_depth,
    )
    e = datetime.datetime.now()
    print("run time:", e - s)


def read_stress_results_fix_depth(config: CfsConfig, obs_plane, obs_depth):
    print("Outputting results for depth %f km to csv files" % obs_depth)
    path_results_each = os.path.join(config.path_output, "grn_d", "results_each")
    path_output_results = config.path_output_results_dynamic
    os.makedirs(path_output_results, exist_ok=True)

    if config.optimal_type == 0:
        file_info = [[item, "_%s.npy" % item, 1] for item in result_name_list]
        file_info[0][2] = 3
        file_info[1][2] = 3
        file_info = file_info + [["cfs", "_cfs.npy", 1]]
    elif config.optimal_type == 1:
        file_info = [[item, "_os_%s.npy" % item, 1] for item in result_name_list]
        file_info[0][2] = 3
        file_info[1][2] = 3
        file_info = file_info + [["cfs", "_os_cfs.npy", 1], ["rake", "_os_rake.npy", 1]]
    elif config.optimal_type == 2:
        file_info1 = [
            [item + "1", "_oop_%s1.npy" % item, 1] for item in result_name_list
        ]
        file_info2 = [
            [item + "2", "_oop_%s2.npy" % item, 1] for item in result_name_list
        ]
        file_info1[0][2] = 3
        file_info1[1][2] = 3
        file_info2[0][2] = 3
        file_info2[1][2] = 3
        file_info = file_info1 + file_info2 + [["cfs", "_oop_cfs.npy", 1]]
    else:
        raise ValueError("optimal_type must be 0/1/2")
    file_info = [["stress_ned", "_stress_ned.npy", 6]] + file_info
    data = {
        key: np.zeros((len(obs_plane) * col_num, config.sampling_num))
        for key, _, col_num in file_info
    }
    for i in range(len(obs_plane)):
        lat, lon, dep = obs_plane[i, 0:3]
        fname = "%.4f_%.4f_%.4f" % (lat, lon, dep)
        for key, suffix, col_num in file_info:
            arr = np.load(os.path.join(path_results_each, fname + suffix))
            data[key][i * col_num : (i + 1) * col_num] = arr.T

    for key, arr in data.items():
        if config.optimal_type == 0:
            out_name = "%s_dynamic_dep_%.2f.csv" % (key, obs_depth)
        elif config.optimal_type == 1:
            out_name = "%s_dynamic_dep_%.2f.csv" % (key + "_os", obs_depth)
        elif config.optimal_type == 2:
            out_name = "%s_dynamic_dep_%.2f.csv" % (key + "_oop", obs_depth)
        else:
            raise ValueError("optimal_type must be 0/1/2")
        pd.DataFrame(arr).to_csv(
            os.path.join(path_output_results, out_name), header=False, index=False
        )


def run_all_dynamic(config: CfsConfig):
    create_dynamic_lib(config)
    if config.processes_num == 1:
        compute_dynamic_cfs_sequential(config)
    else:
        compute_dynamic_cfs_parallel(config)
    compute_dynamic_cfs_fix_depth_parallel(config)


if __name__ == "__main__":
    pass
