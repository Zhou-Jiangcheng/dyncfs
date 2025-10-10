import os
import re
import datetime
from pathlib import Path
import csv
from typing import List, Tuple

import numpy as np
import pandas as pd

from .configuration import CfsConfig
from .focal_mechanism import tensor2full_tensor_matrix
from .geo import d2km, cartesian_2_spherical, convert_sub_faults_geo2ned
from .utils import read_source_array, reshape_sub_faults


def get_number_in_line(line):
    numbers = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", line)
    # print(numbers)
    numbers = [float(item) for item in numbers if item != ""]
    return numbers


def convert_usgs_basic2source_csvs(
    path_rup_model, path_input_dir, sampling_interval_stf
):
    """
    :param path_rup_model: Path to basic_inversion.param downloaded from usgs website.
    :param path_input_dir: The same as the path_input parameter in the ini file.
    :param sampling_interval_stf: Sampling interval of source time function to compute CFS.

    Output columns in each csv file:
    lat(deg), lon(deg), depth(km), strike(deg), dip(deg), rake(deg),
    length_strike(km), length_dip(km), slip(m), m0(Nm), stf(dimensionless)
    """
    srate_stf = 1 / sampling_interval_stf
    with open(path_rup_model, "r") as fr:
        lines = fr.readlines()
    n_planes = int(get_number_in_line(lines[0])[0])
    N = len(lines)
    flag = "#Fault_segment"
    inds_list = []
    Lx_list = []
    Ly_list = []
    source_shapes = []
    for i in range(1, N):
        if lines[i][:14] == flag:
            n_plane, nx, Lx, ny, Ly = get_number_in_line(lines[i])
            nx, ny = int(nx), int(ny)
            source_shapes.append([nx, ny])
            # if Lx != Ly:
            #     warnings.warn("Dx!=Dy, please check Dx(km) and Dy(km) in model")
            inds_list.append([i + 9, i + 9 + nx * ny])
            Lx_list.append(Lx)
            Ly_list.append(Ly)
    if not inds_list:
        raise ValueError('Can not find "#Fault_segment" in basic_inversion.param file')

    for j in range(n_planes):
        data = []
        for i in range(inds_list[j][0], inds_list[j][1]):
            data.append(get_number_in_line(lines[i]))
        data = np.array(data)
        data[:, -1] = data[:, -1] / 1e7
        N_sub = len(data)
        nt_cut_len_max = round(
            (
                round(np.max(data[:, -4]))
                + round(np.max(data[:, -3]))
                + round(np.max(data[:, -2]))
            )
            * srate_stf
            + 1
        )

        sub_stfs = np.zeros([N_sub, nt_cut_len_max])
        for i in range(N_sub):
            start = round(data[i, -4] * srate_stf)
            peak = round((data[i, -4] + data[i, -3]) * srate_stf)
            end = round((data[i, -4] + data[i, -3] + data[i, -2]) * srate_stf)
            sub_stfs[i, start:peak] = np.linspace(
                0, peak - start - 1, peak - start, endpoint=True
            )
            sub_stfs[i, peak:end] = np.linspace(
                end - peak, 1, end - peak, endpoint=True
            )
            sub_stfs[i, start:end] = (
                sub_stfs[i, start:end]
                / np.sum(sub_stfs[i, start:end] / srate_stf)
                * data[i, -1]
            )
            # tau_s_p = (peak - start) / srate_stf
            # t_s_p = np.linspace(0, tau_s_p, (peak - start))
            # sub_stfs[i, start:peak] = np.sin((2 * np.pi) / (4 * tau_s_p) * t_s_p)
            # tau_p_e = (end - peak) / srate_stf
            # t_p_e = np.linspace(0, tau_p_e, (end - peak))
            # sub_stfs[i, peak:end] = np.cos((2 * np.pi) / (4 * tau_p_e) * t_p_e)
            # sub_stfs[i, start:end] = sub_stfs[i, start:end] / np.sum(
            #     sub_stfs[i, start:end] / srate_stf) * data[i, -1]

        #    [lat(deg), lon(deg), depth(km), strike(deg), dip(deg), rake(deg),
        #    length_strike(km), length_dip(km), slip(m), m0(Nm),
        #    stf(dimensionless)]
        source_plane = np.zeros((N_sub, 10 + nt_cut_len_max))
        # lat(deg), lon(deg), depth(km)
        source_plane[:, :3] = data[:, :3]
        # strike(deg), dip(deg), rake(deg)
        source_plane[:, 3] = data[:, 5]
        source_plane[:, 4] = data[:, 6]
        source_plane[:, 5] = data[:, 4]
        # length_strike(km), length_dip(km)
        source_plane[:, 6] = Lx_list[j] * np.ones(N_sub)
        source_plane[:, 7] = Ly_list[j] * np.ones(N_sub)
        # slip(m)
        source_plane[:, 8] = data[:, 3] / 1e2
        # m0(Nm)
        source_plane[:, 9] = data[:, -1]
        # stf(dimensionless)
        source_plane[:, 10:] = sub_stfs
        order = (
            np.arange(N_sub)
            .reshape(source_shapes[j][1], source_shapes[j][0])
            .T.flatten()
        )
        source_plane = source_plane[order, :]
        df = pd.DataFrame(source_plane)
        df.to_csv(
            str(os.path.join(path_input_dir, "source_plane%d.csv" % (j + 1))),
            header=False,
            index=False,
        )

    print("source_shapes=", source_shapes)
    print("convert usgs basic_inversion.param to input csv successfully")


def convert_fsp2source_csvs(path_fsp, path_input_dir, sampling_interval_stf):
    """
    to be continued
    Read a .fsp file and export each SEGMENT block
    into a separate CSV file (without header).

    :param path_fsp: Path to a .fsp format file.
    :param path_input_dir: The same as the path_input parameter in the ini file.
    :param sampling_interval_stf: Sampling interval of source time function to compute CFS.

    Output columns in each csv file:
    lat(deg), lon(deg), depth(km), strike(deg), dip(deg), rake(deg),
    length_strike(km), length_dip(km), slip(m), m0(Nm), stf(dimensionless)
    """
    srate_stf = 1 / sampling_interval_stf

    with open(path_fsp, "r") as fr:
        lines = fr.readlines()

    lines_paras_info = [[]]
    for i in range(len(lines) - 1):
        if "-----" in lines[i] and len(lines_paras_info[-1]) > 0:
            lines_paras_info.append([])
        if (lines[i][0] == "%") and ("-----" not in lines[i]):
            lines_paras_info[-1].append(lines[i])

    fsp_info = {}
    for i in range(len(lines_paras_info[0])):
        if lines_paras_info[0][i][:5] == "% Loc":
            lat, lon, dep = get_number_in_line(lines_paras_info[0][i])
            fsp_info["lat"] = lat
            fsp_info["lon"] = lon
            fsp_info["dep"] = dep
        if lines_paras_info[0][i][:6] == "% Size":
            _, _, Mw, M0 = get_number_in_line(lines_paras_info[0][i])
            fsp_info["mw"] = Mw
            fsp_info["m0"] = M0
    for i in range(len(lines_paras_info[1])):
        if lines_paras_info[1][i][:11] == "% Invs : Dx":
            Dx, Dy = get_number_in_line(lines_paras_info[1][i])
            fsp_info["dx"] = Dx
            fsp_info["dy"] = Dy
    # print(fsp_info)


def convert_source_csvs2coulomb3(
    config: CfsConfig, obs_depth, possion_ratio=0.25, youngs_modulus=8e5
):
    """
    Convert dyncfs input files to Coulomb3 input file.
    :param config: Config object of dyncfs.
    :param obs_depth: Observation depth, unit km.
    :param possion_ratio: Possion's Ratio.
    :param youngs_modulus: Young's Modulus, unit bar.
    """
    lines_srcs = [
        "\n  #   X-start    Y-start     X-fin      Y-fin   Kode  rake     netslip   dip angle     top      bot\n",
        "xxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx xxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx\n",
    ]
    N_srcs = 0
    for ind_src in range(len(config.source_inds)):
        source_plane = read_source_array(
            source_inds=[config.source_inds[ind_src]],
            path_input=config.path_input,
            shift2corner=False,
        )
        N_srcs = N_srcs + len(source_plane)
        sub_faults = convert_sub_faults_geo2ned(
            sub_faults=source_plane[:, :3],
            source_point=np.concatenate([config.source_ref, np.zeros(1)]),
            approximate=True,
        )
        sub_fms = source_plane[:, 3:6]
        sub_lengths = source_plane[:, 6:8] * 1e3
        num_strike = config.source_shapes[ind_src][0]
        num_dip = config.source_shapes[ind_src][1]
        X, Y, Z = reshape_sub_faults(
            sub_faults, sub_fms, sub_lengths, num_strike, num_dip
        )
        X = X / 1e3
        Y = Y / 1e3
        Z = Z / 1e3
        Z[Z < 0] = 0
        for i in range(num_strike):
            for j in range(num_dip):
                ind = j + i * num_dip
                # exchange x,y direction to correspond x-east, y-north
                line_ij = (
                    "  1 %10.4f %10.4f %10.4f %10.4f 100 "
                    % (
                        Y[i, j],
                        X[i, j],
                        Y[i + 1, j],
                        X[i + 1, j],
                    )
                    + "%10.4f %10.4f %10.4f "
                    % (
                        float(source_plane[ind, 5]),
                        float(source_plane[ind, 8]),
                        float(source_plane[ind, 4]),
                    )
                    + "%10.4f %10.4f\n"
                    % (
                        float(Z[i, j]),
                        float(Z[i, j + 1]),
                    )
                )
                lines_srcs.append(line_ij)

    lines_head = [
        "Coulomb.inp automatically created by dyncfs. x-east. y-north.\n"
        "Generated at %s.\n" % str(datetime.datetime.now().date()),
        "#reg1=  0  #reg2=  0  #fixed= %d  sym=  1\n" % N_srcs,
        "PR1=%15.3f PR2=%15.3f DEPTH=%15.3f\n"
        % (possion_ratio, possion_ratio, obs_depth),
        "E1= %15.3e E2= %15.3e\n" % (youngs_modulus, youngs_modulus),
        "XSYM=%15.3f YSYM=%15.3f\n" % (0.0, 0.0),
        "FRIC=%15.3f\n" % config.mu_f,
    ]

    if config.optimal_type == 2:
        st = tensor2full_tensor_matrix(config.tectonic_stress, "ned")
        [eigenvalues, eigenvectors] = np.linalg.eig(st)
        index = eigenvalues.argsort()
        eigenvalues = -eigenvalues[:, index]
        eigenvectors = eigenvectors[:, index]
        lines_regional_stress = []
        for i in range(3):
            n = eigenvectors[:, i].flatten()
            _, phi, theta = cartesian_2_spherical(*n)
            S_ind = i + 1
            lines_regional_stress.append(
                "S%dDR=%15.3f S%dDP=%15.3f S%dIN=%15.3f S1GD=%15.3f\n"
                % (
                    S_ind,
                    eigenvalues[i] / 1e5,
                    S_ind,
                    np.rad2deg(phi),
                    S_ind,
                    np.rad2deg(theta),
                    0.0,
                )
            )
    else:
        lines_regional_stress = [
            "S1DR=         19.000 S1DP=         -0.010 S1IN=        100.000 S1GD=          0.000\n",
            "S2DR=         89.990 S2DP=         89.990 S2IN=         30.000 S2GD=          0.000\n",
            "S3DR=        109.000 S3DP=         -0.010 S3IN=          0.000 S3GD=          0.000\n",
        ]

    # exchange x,y direction to correspond x-east, y-north
    x_min = (config.obs_lat_range[0] - config.obs_ref[0]) * d2km
    x_max = (config.obs_lat_range[1] - config.obs_ref[0]) * d2km
    y_min = (config.obs_lon_range[0] - config.obs_ref[1]) * d2km
    y_max = (config.obs_lon_range[1] - config.obs_ref[1]) * d2km
    delta_x = config.obs_delta_lat * d2km
    delta_y = config.obs_delta_lon * d2km

    def format_coulomb_value(value, width=15, precision=7):
        """Helper function to format numbers for Coulomb input."""
        if value >= 0:
            return f"{value:{width}.{precision}f}"
        else:
            return f"{value:{width}.{precision - 1}f}"

    lines_grid = [
        "\n Grid Parameters\n",
        f"1 ----------------------------  Start-x ={format_coulomb_value(y_min)}\n",
        f"2 ----------------------------  Start-y ={format_coulomb_value(x_min)}\n",
        f"3 --------------------------   Finish-x ={format_coulomb_value(y_max)}\n",
        f"4 --------------------------   Finish-y ={format_coulomb_value(x_max)}\n",
        f"5 -----------------------   x-increment ={format_coulomb_value(delta_y)}\n",
        f"6 -----------------------   y-increment ={format_coulomb_value(delta_x)}\n",
    ]

    lines = lines_head + lines_regional_stress + lines_srcs + lines_grid

    with open(os.path.join(config.path_input, "coulomb3.inp"), "w") as fw:
        fw.writelines(lines)
    print("convert input csv to coulomb3.inp successfully")
