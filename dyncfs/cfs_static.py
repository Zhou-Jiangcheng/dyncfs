import datetime
import os
from typing import Union
import json
import multiprocessing as mp
from multiprocessing import get_context

import numpy as np
import pandas as pd
from tqdm import tqdm

from .configuration import CfsConfig
from .focal_mechanism import (
    plane2nd,
    tensor2full_tensor_matrix,
    check_convert_fm,
    mt2plane,
)
from .create_edgrn_bulk import (
    pre_process_edgrn2,
    create_grnlib_edgrn2_parallel,
)
from .create_edcmp_bulk import (
    pre_process_edcmp2,
    compute_static_stress_edcmp2_sequential,
    compute_static_stress_edcmp2_parallel,
)
from .read_edcmp import seek_edcmp2
from .utils import (
    read_source_array,
    ignore_slip_source_array,
    cut_stf_modify_source_array,
)


def create_static_lib(config: CfsConfig):
    s = datetime.datetime.now()
    pre_process_edgrn2(
        processes_num=config.processes_num,
        path_green=config.path_green_staic,
        path_bin=config.path_bin_edgrn,
        grn_source_depth_range=config.static_source_depth_range,
        grn_source_delta_depth=config.static_source_delta_depth,
        grn_dist_range=config.static_dist_range,
        grn_delta_dist=config.static_delta_dist,
        grn_obs_depth_list=config.static_obs_depth_list,
        wavenumber_sampling_rate=config.wavenumber_sampling_rate,
        path_nd=config.path_nd,
        earth_model_layer_num=config.earth_model_layer_num,
    )
    create_grnlib_edgrn2_parallel(
        path_green=config.path_green_staic, check_finished=config.check_finished
    )
    e = datetime.datetime.now()
    print("run time:", e - s)


def cal_coulomb_failure_stress(
    norm_stress,
    shear_stress,
    mu_f=0.4,
):
    """
    :param norm_stress: Normal stress (Pa).
    :param shear_stress: Shear stress (Pa).
    :param mu_f: Effective coefficient of friction.
    :return: coulomb_stress(Pa)
    """
    coulomb_stress = shear_stress + mu_f * norm_stress
    return coulomb_stress


def cal_coulomb_failure_stress_poroelasticity(
    norm_stress,
    shear_stress,
    mean_stress,
    mu_f=0.6,
    B_pore=0,
):
    """

    :param norm_stress: Normal stress (Pa).
    :param shear_stress: Shear stress (Pa).
    :param mean_stress: (Pa)
    :param mu_f: Coefficient of friction.
    :param B_pore: Skempton's coefficient.
    :return: coulomb_stress(Pa)
    """
    coulomb_stress = shear_stress + mu_f * (norm_stress + B_pore * mean_stress)
    return coulomb_stress


def cal_cfs_static_single_point_fix_fm(
    obs_fm: Union[list[float], np.ndarray],
    stress: Union[list[float], np.ndarray],
    mu_f: float = 0.4,
    B_pore: float = 0.0,
):
    """
    :param obs_fm: [strike, dip, rake](deg)
    :param stress: Earthquake induce stress perturbation.
        [sigma_nn, sigma_ne, sigma_nd, sigma_ee, sigma_ed, sigma_dd](Pa)
    :param mu_f: Coefficient or effective coefficient of friction.
    :param B_pore: Skempton's coefficient. If zero, the mu_f means effective coefficient of friction.


    :return n.flatten(), d.flatten(), sigma, tau, cfs
    """
    sigma_tensor = tensor2full_tensor_matrix(mt=stress, flag="ned")
    n, d = plane2nd(*obs_fm)
    n = np.array([n.flatten()]).T
    d = np.array([d.flatten()]).T
    sigma_vector = np.dot(sigma_tensor, n)
    sigma = np.dot(sigma_vector.T, n)[0][0]
    tau = np.dot(sigma_vector.T, d)[0][0]
    if B_pore != 0:
        cfs = cal_coulomb_failure_stress(norm_stress=sigma, shear_stress=tau, mu_f=mu_f)
    else:
        cfs = cal_coulomb_failure_stress_poroelasticity(
            norm_stress=sigma,
            shear_stress=tau,
            mean_stress=np.mean(np.diag(sigma_tensor)),
            mu_f=mu_f,
            B_pore=B_pore,
        )
    return n.flatten(), d.flatten(), sigma, tau, cfs


def cal_cfs_static_single_point_opt_rake(
    obs_strike,
    obs_dip,
    stress: Union[list[float], np.ndarray],
    tectonic_stress: Union[list[float], np.ndarray],
    mu_f: float = 0.4,
    B_pore: float = 0.0,
):
    """
    :param obs_strike: strike of the obs plane (deg).
    :param obs_dip: dip of the obs plane (deg).
    :param stress: Earthquake induce stress perturbation.
        [sigma_nn, sigma_ne, sigma_nd, sigma_ee, sigma_ed, sigma_dd](Pa)
    :param tectonic_stress: Tectonic stress in NED axis.
        [sigma_nn, sigma_ne, sigma_nd, sigma_ee, sigma_ed, sigma_dd](Pa)
    :param mu_f: Coefficient or effective coefficient of friction.
    :param B_pore: Skempton's coefficient. If zero, the mu_f means effective coefficient of friction.

    :return: n.flatten(), d.flatten(), sigma, tau, cfs, rake
    """
    strike, dip = np.deg2rad(obs_strike), np.deg2rad(obs_dip)
    sin_strike, cos_strike = np.sin(strike), np.cos(strike)
    sin_dip, cos_dip = np.sin(dip), np.cos(dip)
    n = np.array([-sin_dip * sin_strike, -sin_dip * cos_strike, -cos_dip])

    sigma_tensor = tensor2full_tensor_matrix(mt=np.array(stress), flag="ned")
    tectonic_stress_tensor = tensor2full_tensor_matrix(
        mt=np.array(tectonic_stress), flag="ned"
    )
    sigma_tensor_sum = sigma_tensor + tectonic_stress_tensor

    sigma_vector_sum = np.dot(sigma_tensor_sum, n)
    sigma_sum = np.dot(sigma_vector_sum.T, n)[0][0]
    d = sigma_vector_sum - sigma_sum * n
    d = d / np.linalg.norm(d)
    rake = np.rad2deg(np.arcsin(d[2] / sin_dip))
    if rake > 180:
        rake = rake - 360

    sigma_vector = np.dot(sigma_tensor, n)
    sigma = np.dot(sigma_vector.T, n)[0][0]
    tau = np.dot(sigma_vector.T, d)[0][0]

    if B_pore != 0:
        cfs = cal_coulomb_failure_stress(norm_stress=sigma, shear_stress=tau, mu_f=mu_f)
    else:
        cfs = cal_coulomb_failure_stress_poroelasticity(
            norm_stress=sigma,
            shear_stress=tau,
            mean_stress=np.mean(np.diag(sigma_tensor)),
            mu_f=mu_f,
            B_pore=B_pore,
        )

    return (n.flatten(), d.flatten(), sigma, tau, cfs, rake)


def cal_cfs_static_single_point_opt_plane(
    stress: Union[list[float], np.ndarray],
    tectonic_stress_type: int,
    tectonic_stress: Union[list[float], np.ndarray],
    mu_f: float = 0.4,
    B_pore: float = 0.0,
):
    """
    :param stress: Earthquake induce stress perturbation.
        [sigma_nn, sigma_ne, sigma_nd, sigma_ee, sigma_ed, sigma_dd](Pa)
    :param tectonic_stress_type:
       If tectonic_stress_type = 1, a full stress tensor must be supplied in the
       tectonic_stress: [s_nn, s_ne, s_nd, s_ee, s_ed, s_dd] (Pa).
       If tectonic_stress_type = 2, only the principal stress orientations,
       (principal stresses ordered from smallest to largest, tensile is positive) be supplied in the
       tectonic_stress: [azimuth1, plunge1, azimuth2, plunge2, azimuth3, plunge3] (deg).
    :param tectonic_stress: Tectonic stress [s_nn, s_ne, s_nd, s_ee, s_ed, s_dd] (MPa)
            or principal stress orientations [azimuth1, plunge1, azimuth2, plunge2, azimuth3, plunge3] (deg)
    :param mu_f: Coefficient or effective coefficient of friction.
    :param B_pore: Skempton's coefficient. If zero, the mu_f means effective coefficient of friction.

    :return: ([n1.flatten(), d1.flatten(), sigma1, tau1],
            [n2.flatten(), d2.flatten(), sigma2, tau2],
            cfs)
    """
    if mu_f == 0:
        theta = np.pi / 4
    else:
        theta = 1 / 2 * np.arctan(1 / mu_f)
    sigma_tensor = tensor2full_tensor_matrix(mt=stress.copy(), flag="ned")
    tectonic_stress = np.array(tectonic_stress)
    if tectonic_stress_type == 1:
        tectonic_stress_tensor = tensor2full_tensor_matrix(
            mt=tectonic_stress.copy(), flag="ned"
        )
        S = sigma_tensor + tectonic_stress_tensor
        [eigenvalues, eigenvectors] = np.linalg.eig(S)
        index = eigenvalues.argsort()[::-1]
        R = eigenvectors[:, index]
    elif tectonic_stress_type == 2:
        R = np.zeros((3, 3))
        for i in range(3):
            phi = np.deg2rad(tectonic_stress[i])
            delta = np.deg2rad(tectonic_stress[i + 1])
            R[:, i] = np.array(
                [
                    np.cos(phi) * np.cos(delta),
                    np.sin(phi) * np.cos(delta),
                    np.sin(delta),
                ]
            )
        R = R[:, ::-1]
    else:
        raise ValueError("tectonic_stress_type must be 1 or 2")
    n_prin_axis1 = np.array([[np.cos(theta), 0, np.sin(theta)]]).T
    n1 = R @ n_prin_axis1
    d_prin_axis1 = np.array([[np.sin(theta), 0, -np.cos(theta)]]).T
    d1 = R @ d_prin_axis1
    n_prin_axis2 = np.array([[np.cos(theta), 0, -np.sin(theta)]]).T
    n2 = R @ n_prin_axis2
    d_prin_axis2 = np.array([[np.sin(theta), 0, np.cos(theta)]]).T
    d2 = R @ d_prin_axis2
    if n1[2] > 0:
        n1 = -n1
        d1 = -d1
    if n2[2] > 0:
        n2 = -n2
        d2 = -d2
    sigma_vector1 = np.dot(sigma_tensor, n1)
    sigma1 = np.dot(sigma_vector1.T, n1)[0][0]
    # tau1 = np.linalg.norm(sigma_vector1-sigma1*n1)
    tau1 = np.dot(sigma_vector1.T, d1)[0][0]

    sigma_vector2 = np.dot(sigma_tensor, n2)
    sigma2 = np.dot(sigma_vector2.T, n2)[0][0]
    # tau2 = np.linalg.norm(sigma_vector2-sigma2*n2)
    tau2 = np.dot(sigma_vector2.T, d2)[0][0]
    if B_pore != 0:
        cfs2 = cal_coulomb_failure_stress(
            norm_stress=sigma2, shear_stress=tau2, mu_f=mu_f
        )
    else:
        cfs2 = cal_coulomb_failure_stress_poroelasticity(
            norm_stress=sigma2,
            shear_stress=tau2,
            mean_stress=np.mean(np.diag(sigma_tensor)),
            mu_f=mu_f,
            B_pore=B_pore,
        )

    return (
        [n1.flatten(), d1.flatten(), sigma1, tau1],
        [n2.flatten(), d2.flatten(), sigma2, tau2],
        cfs2,
    )


def save_csv_file(path_output_results, data, name, ind_obs):
    df = pd.DataFrame(data)
    df.to_csv(
        str(os.path.join(path_output_results, "%s_plane%d.csv" % (name, ind_obs))),
        header=False,
        index=False,
    )


def compute_static_cfs(config: CfsConfig):
    s = datetime.datetime.now()

    source_array = read_source_array(
        source_inds=config.source_inds,
        path_input=config.path_input,
        shift2corner=True,
    )
    if config.slip_thresh > 0:
        source_array = ignore_slip_source_array(source_array, config.slip_thresh)
    if config.cut_stf > 0:
        source_array = cut_stf_modify_source_array(source_array, config.cut_stf)
    source_array_new = np.zeros((len(source_array), 9))
    source_array_new[:, 0] = source_array[:, 8]
    source_array_new[:, 1:9] = source_array[:, 0:8]
    pre_process_edcmp2(
        processes_num=config.processes_num,
        path_green=config.path_green_staic,
        path_bin=config.path_bin_edcmp,
        obs_depth_list=config.static_obs_depth_list,
        obs_x_range=config.obs_lat_range,
        obs_y_range=config.obs_lon_range,
        obs_delta_x=config.obs_delta_lat,
        obs_delta_y=config.obs_delta_lon,
        source_array=source_array_new,
        source_ref=config.source_ref,
        obs_ref=config.obs_ref,
        layered=config.layered,
        lam=config.lam,
        mu=config.mu,
    )
    if config.processes_num == 1:
        if config.multiprocessing_flag is not None:
            os.environ["OMP_NUM_THREADS"] = ""
            os.environ["MKL_NUM_THREADS"] = ""
            os.environ["OPENBLAS_NUM_THREADS"] = ""
            config.multiprocessing_flag = None
        compute_static_stress_edcmp2_sequential(
            path_green=config.path_green_staic, check_finished=config.check_finished
        )
    elif config.processes_num > 1:
        if config.multiprocessing_flag is None:
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
        mp.set_start_method("spawn", force=True)
        compute_static_stress_edcmp2_parallel(
            path_green=config.path_green_staic, check_finished=config.check_finished
        )

    for ind_obs in config.obs_inds:
        obs_plane = pd.read_csv(
            str(os.path.join(config.path_input, "obs_plane%d.csv" % ind_obs)),
            index_col=False,
            header=None,
        ).to_numpy()
        N = len(obs_plane)
        stress_tensor_array_enz = seek_edcmp2(
            str(os.path.join(config.path_output, "grn_s")),
            "stress",
            obs_plane[:, :3],
            geo_coordinate=True,
        )

        stress_tensor_array = np.zeros_like(stress_tensor_array_enz)
        # ee en ez nn nz zz
        # nn ne nd ee ed dd
        stress_tensor_array[:, 0] = stress_tensor_array_enz[:, 3]
        stress_tensor_array[:, 1] = stress_tensor_array_enz[:, 1]
        stress_tensor_array[:, 2] = -stress_tensor_array_enz[:, 4]
        stress_tensor_array[:, 3] = stress_tensor_array_enz[:, 0]
        stress_tensor_array[:, 4] = -stress_tensor_array_enz[:, 2]
        stress_tensor_array[:, 5] = stress_tensor_array_enz[:, 5]

        path_stress_tensor = str(
            os.path.join(
                config.path_output_results_static,
                "stress_tensor_plane%d.npy" % ind_obs,
            )
        )
        np.save(path_stress_tensor, stress_tensor_array)

        if config.optimal_type == 0:
            n_array = np.zeros((N, 3))
            d_array = np.zeros((N, 3))
            norm_stress_array = np.zeros(N)
            shear_stress_array = np.zeros(N)
            cfs_array = np.zeros(N)
            for i in tqdm(
                range(N),
                desc="Computing static Coulomb Failure Stress change at No.%d plane"
                % ind_obs,
            ):
                (n, d, sigma, tau, cfs) = cal_cfs_static_single_point_fix_fm(
                    obs_fm=obs_plane[i, 3:],
                    stress=stress_tensor_array[i, :],
                    mu_f=config.mu_f,
                    B_pore=config.B_pore,
                )
                n_array[i] = n
                d_array[i] = d
                norm_stress_array[i] = sigma
                shear_stress_array[i] = tau
                cfs_array[i] = cfs
            tasks = [
                (n_array, "normal_vector_static"),
                (d_array, "rupture_vector_static"),
                (norm_stress_array, "normal_stress_static"),
                (shear_stress_array, "shear_stress_static"),
                (cfs_array, "cfs_static"),
            ]
        elif config.optimal_type == 1:
            n_array = np.zeros((N, 3))
            d_array = np.zeros((N, 3))
            norm_stress_array = np.zeros(N)
            shear_stress_array = np.zeros(N)
            cfs_array = np.zeros(N)
            rake_array = np.zeros(N)
            for i in tqdm(
                range(N),
                desc="Computing static Coulomb Failure Stress change at No.%d plane (optimal rake)"
                % ind_obs,
            ):
                (n, d, sigma, tau, cfs, rake) = cal_cfs_static_single_point_opt_rake(
                    obs_strike=obs_plane[i, 4],
                    obs_dip=obs_plane[i, 5],
                    stress=stress_tensor_array[i, :],
                    tectonic_stress=config.tectonic_stress,
                    mu_f=config.mu_f,
                    B_pore=config.B_pore,
                )
                n_array[i] = n
                d_array[i] = d
                norm_stress_array[i] = sigma
                shear_stress_array[i] = tau
                cfs_array[i] = cfs
                rake_array[i] = rake
            tasks = [
                (n_array, "normal_vector_os_static"),
                (d_array, "rupture_vector_os_static"),
                (norm_stress_array, "normal_stress_os_static"),
                (shear_stress_array, "shear_stress_os_static"),
                (cfs_array, "cfs_os_static"),
                (rake_array, "rake_os_static"),
            ]
        elif not config.optimal_type == 2:
            n1_array = np.zeros((N, 3))
            d1_array = np.zeros((N, 3))
            norm_stress1_array = np.zeros(N)
            shear_stress1_array = np.zeros(N)
            n2_array = np.zeros((N, 3))
            d2_array = np.zeros((N, 3))
            norm_stress2_array = np.zeros(N)
            shear_stress2_array = np.zeros(N)
            cfs_array = np.zeros(N)
            for i in tqdm(
                range(N),
                desc="Computing static Coulomb Failure Stress change (OOP) at No.%d plane"
                % ind_obs,
            ):
                ([n1, d1, sigma1, tau1], [n2, d2, sigma2, tau2], cfs) = (
                    cal_cfs_static_single_point_opt_plane(
                        stress=stress_tensor_array[i, :],
                        tectonic_stress_type=config.tectonic_stress_type,
                        tectonic_stress=config.tectonic_stress,
                        mu_f=config.mu_f,
                        B_pore=config.B_pore,
                    )
                )
                n1_array[i] = n1
                d1_array[i] = d1
                norm_stress1_array[i] = sigma1
                shear_stress1_array[i] = tau1

                n2_array[i] = n2
                d2_array[i] = d2
                norm_stress2_array[i] = sigma2
                shear_stress2_array[i] = tau2

                cfs_array[i] = cfs
            tasks = [
                (n1_array, "normal_vector1_oop_static"),
                (d1_array, "rupture_vector1_oop_static"),
                (norm_stress1_array, "normal_stress1_oop_static"),
                (shear_stress1_array, "shear_stress1_oop_static"),
                (n2_array, "normal_vector2_oop_static"),
                (d2_array, "rupture_vector2_oop_static"),
                (norm_stress2_array, "normal_stress2_oop_static"),
                (shear_stress2_array, "shear_stress2_oop_static"),
                (cfs_array, "cfs_oop_static"),
            ]
        else:
            raise ValueError("optimal_type must be 0/1/2")
        for array, name in tasks:
            save_csv_file(config.path_output_results_static, array, name, ind_obs)
    e = datetime.datetime.now()
    print("run time:", e - s)


def compute_static_cfs_fix_depth(
    config: CfsConfig,
    obs_depth: float = None,
    optimal_type: int = True,
    receiver_mechanism: list = None,
    obs_lat_range: list = None,
    obs_lon_range: list = None,
    obs_delta_lat: float = None,
    obs_delta_lon: float = None,
):
    """
    Compute static Coulomb failure stress change at fixed depth.

    :param config:
    :param obs_depth: Observation depth, unit km.
    :param optimal_type: same as config.optimal_type
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
        shift2corner=True,
    )
    if config.slip_thresh > 0:
        source_array = ignore_slip_source_array(source_array, config.slip_thresh)
    if config.cut_stf > 0:
        source_array = cut_stf_modify_source_array(source_array, config.cut_stf)
    if optimal_type == 0 and receiver_mechanism is None:
        mt_mean = np.zeros(6)
        for i in range(len(source_array)):
            mt_i = check_convert_fm(source_array[i, 3:6])
            mt_i = np.array(mt_i) * source_array[i, 9]
            mt_mean = mt_mean + mt_i
        receiver_mechanism = mt2plane(mt=mt_mean)[0]

    source_array_new = np.zeros((len(source_array), 9))
    source_array_new[:, 0] = source_array[:, 8]
    source_array_new[:, 1:9] = source_array[:, 0:8]
    with open(os.path.join(config.path_green_staic, "green_lib_info.json"), "r") as fr:
        green_info = json.load(fr)
    grn_obs_depth_list = green_info["grn_obs_depth_list"]
    grn_obs_depth = grn_obs_depth_list[
        np.argmin(np.abs(obs_depth - np.array(grn_obs_depth_list)))
    ]
    pre_process_edcmp2(
        processes_num=config.processes_num,
        path_green=config.path_green_staic,
        path_bin=config.path_bin_edcmp,
        obs_depth_list=[grn_obs_depth],
        obs_x_range=obs_lat_range,
        obs_y_range=obs_lon_range,
        obs_delta_x=obs_delta_lat,
        obs_delta_y=obs_delta_lon,
        source_array=source_array_new,
        source_ref=config.source_ref,
        obs_ref=config.obs_ref,
        layered=config.layered,
        lam=config.lam,
        mu=config.mu,
    )
    if config.processes_num == 1:
        compute_static_stress_edcmp2_sequential(
            path_green=config.path_green_staic, check_finished=config.check_finished
        )
    elif config.processes_num > 1:
        compute_static_stress_edcmp2_parallel(
            path_green=config.path_green_staic, check_finished=config.check_finished
        )

    Nx = int(np.ceil((obs_lat_range[1] - obs_lat_range[0]) / obs_delta_lat) + 1)
    Ny = int(np.ceil((obs_lon_range[1] - obs_lon_range[0]) / obs_delta_lon) + 1)

    obs_plane = np.zeros((Nx * Ny, 6))
    lat_array = np.linspace(obs_lat_range[0], obs_lat_range[1], Nx)
    lon_array = np.linspace(obs_lon_range[0], obs_lon_range[1], Ny)
    for i in range(Nx):
        for j in range(Ny):
            ind = j + i * Ny
            obs_plane[ind, :2] = np.array([lat_array[i], lon_array[j]])
    obs_plane[:, 2] = obs_plane[:, 2] + obs_depth
    if optimal_type == 0:
        obs_plane[:, 3] = obs_plane[:, 3] + receiver_mechanism[0]
        obs_plane[:, 4] = obs_plane[:, 4] + receiver_mechanism[1]
        obs_plane[:, 5] = obs_plane[:, 5] + receiver_mechanism[2]
        print(
            "receiver_mechanism is (strike, dip, rake)=(%.2f, %.2f, %.2f) deg."
            % (receiver_mechanism[0], receiver_mechanism[1], receiver_mechanism[2])
        )

    N = len(obs_plane)
    stress_tensor_array_enz = seek_edcmp2(
        str(os.path.join(config.path_output, "grn_s")),
        "stress",
        obs_plane[:, :3],
        geo_coordinate=True,
    )

    stress_tensor_array = np.zeros_like(stress_tensor_array_enz)
    # ee en ez nn nz zz
    # nn ne nd ee ed dd
    stress_tensor_array[:, 0] = stress_tensor_array_enz[:, 3]
    stress_tensor_array[:, 1] = stress_tensor_array_enz[:, 1]
    stress_tensor_array[:, 2] = -stress_tensor_array_enz[:, 4]
    stress_tensor_array[:, 3] = stress_tensor_array_enz[:, 0]
    stress_tensor_array[:, 4] = -stress_tensor_array_enz[:, 2]
    stress_tensor_array[:, 5] = stress_tensor_array_enz[:, 5]

    path_stress_tensor = str(
        os.path.join(
            config.path_output_results_static,
            "stress_tensor_dep_%.2f.npy" % obs_depth,
        )
    )
    np.save(path_stress_tensor, stress_tensor_array)

    if config.optimal_type == 0:
        n_array = np.zeros((N, 3))
        d_array = np.zeros((N, 3))
        norm_stress_array = np.zeros(N)
        shear_stress_array = np.zeros(N)
        cfs_array = np.zeros(N)
        for i in tqdm(
            range(N),
            desc="Computing static Coulomb Failure Stress change at %.2f km depth"
            % obs_depth,
        ):
            (n, d, sigma, tau, cfs) = cal_cfs_static_single_point_fix_fm(
                obs_fm=obs_plane[i, 3:],
                stress=stress_tensor_array[i, :],
                mu_f=config.mu_f,
                B_pore=config.B_pore,
            )
            n_array[i] = n
            d_array[i] = d
            norm_stress_array[i] = sigma
            shear_stress_array[i] = tau
            cfs_array[i] = cfs
        tasks = [
            (n_array, "normal_vector_static"),
            (d_array, "rupture_vector_static"),
            (norm_stress_array, "normal_stress_static"),
            (shear_stress_array, "shear_stress_static"),
            (cfs_array, "cfs_static"),
        ]
    elif config.optimal_type == 1:
        n_array = np.zeros((N, 3))
        d_array = np.zeros((N, 3))
        norm_stress_array = np.zeros(N)
        shear_stress_array = np.zeros(N)
        cfs_array = np.zeros(N)
        rake_array = np.zeros(N)
        for i in tqdm(
            range(N),
            desc="Computing static Coulomb Failure Stress change (OOP) at %.2f km depth"
            % obs_depth,
        ):
            (n, d, sigma, tau, cfs, rake) = cal_cfs_static_single_point_opt_rake(
                obs_strike=obs_plane[i, 4],
                obs_dip=obs_plane[i, 5],
                stress=stress_tensor_array[i, :],
                tectonic_stress=config.tectonic_stress,
                mu_f=config.mu_f,
                B_pore=config.B_pore,
            )
            n_array[i] = n
            d_array[i] = d
            norm_stress_array[i] = sigma
            shear_stress_array[i] = tau
            cfs_array[i] = cfs
            rake_array[i] = rake
        tasks = [
            (n_array, "normal_vector_static_os"),
            (d_array, "rupture_vector_static_os"),
            (norm_stress_array, "normal_stress_static_os"),
            (shear_stress_array, "shear_stress_static_os"),
            (cfs_array, "cfs_static"),
            (rake_array, "rake_static_os"),
        ]
    elif optimal_type == 2:
        n1_array = np.zeros((N, 3))
        d1_array = np.zeros((N, 3))
        norm_stress1_array = np.zeros(N)
        shear_stress1_array = np.zeros(N)

        n2_array = np.zeros((N, 3))
        d2_array = np.zeros((N, 3))
        norm_stress2_array = np.zeros(N)
        shear_stress2_array = np.zeros(N)

        cfs_array = np.zeros(N)
        for i in tqdm(
            range(N),
            desc="Computing static Coulomb Failure Stress change (OOP) at %.2f km depth"
            % obs_depth,
        ):
            ([n1, d1, sigma1, tau1], [n2, d2, sigma2, tau2], cfs) = (
                cal_cfs_static_single_point_opt_plane(
                    stress=stress_tensor_array[i, :],
                    tectonic_stress_type=config.tectonic_stress_type,
                    tectonic_stress=config.tectonic_stress,
                    mu_f=config.mu_f,
                    B_pore=config.B_pore,
                )
            )
            n1_array[i] = n1
            d1_array[i] = d1
            norm_stress1_array[i] = sigma1
            shear_stress1_array[i] = tau1

            n2_array[i] = n2
            d2_array[i] = d2
            norm_stress2_array[i] = sigma2
            shear_stress2_array[i] = tau2
            cfs_array[i] = cfs
        tasks = [
            (n1_array, "normal_vector1_oop_static"),
            (d1_array, "rupture_vector1_oop_static"),
            (norm_stress1_array, "normal_stress1_oop_static"),
            (shear_stress1_array, "shear_stress1_oop_static"),
            (n2_array, "normal_vector2_oop_static"),
            (d2_array, "rupture_vector2_oop_static"),
            (norm_stress2_array, "normal_stress2_oop_static"),
            (shear_stress2_array, "shear_stress2_oop_static"),
            (cfs_array, "cfs_oop_static"),
        ]
    else:
        raise ValueError("optimal_type must be 0/1/2")
    for array, name in tasks:
        df = pd.DataFrame(array)
        df.to_csv(
            str(
                os.path.join(
                    config.path_output_results_static,
                    "%s_dep_%.2f.csv" % (name, obs_depth),
                )
            ),
            header=False,
            index=False,
        )

    e = datetime.datetime.now()
    print("run time:", e - s)


def run_all_static(config: CfsConfig):
    create_static_lib(config)
    compute_static_cfs(config)
    compute_static_cfs_fix_depth(config)


if __name__ == "__main__":
    pass
