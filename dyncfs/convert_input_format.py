import os
import re
import datetime
from pathlib import Path
import csv
from typing import List, Tuple

import numpy as np
import pandas as pd

from pygrnwang.focal_mechanism import tensor2full_tensor_matrix
from pygrnwang.geo import d2km, convert_sub_faults_geo2ned

from .configuration import CfsConfig
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
        # Lat. Lon. depth slip(cm) rake strike dip t_rup t_ris t_fal mo(dyne*cm)
        write_source_plane_csv(
            path_csv=os.path.join(path_input_dir, "source_plane%d.csv" % (j + 1)),
            lat_lon_dep=data[:, :3],
            strike=data[:, 5],
            dip=data[:, 6],
            rake=data[:, 4],
            length_strike=Lx_list[j],
            length_dip=Ly_list[j],
            slip_m=data[:, 3] / 1e2,
            m0=data[:, -1] / 1e7,
            t_rup=data[:, -4],
            t_ris=data[:, -3],
            t_fal=data[:, -2],
            srate_stf=srate_stf,
            nx=source_shapes[j][0],
            nz=source_shapes[j][1],
        )

    print("source_shapes=", source_shapes)
    print("convert usgs basic_inversion.param to input csv successfully")
    return source_shapes


def create_triangle_stfs(t_rup, t_ris, t_fal, m0, srate_stf):
    """
    Moment rate functions of sub faults: linear increase during t_ris after the
    rupture time t_rup, then linear decrease during t_fal. The integral of each
    stf equals its m0.

    :return: sub_stfs, shape (N_sub, nt)
    """
    t_rup, t_ris, t_fal, m0 = (np.asarray(v, dtype=float) for v in (t_rup, t_ris, t_fal, m0))
    N_sub = len(m0)
    # the longest t_rup + t_ris + t_fal of all sub faults
    nt = int(np.ceil(np.max(t_rup + t_ris + t_fal) * srate_stf)) + 1
    sub_stfs = np.zeros([N_sub, nt])
    for i in range(N_sub):
        start = round(t_rup[i] * srate_stf)
        peak = round((t_rup[i] + t_ris[i]) * srate_stf)
        end = round((t_rup[i] + t_ris[i] + t_fal[i]) * srate_stf)
        sub_stfs[i, start:peak] = np.linspace(
            0, peak - start - 1, peak - start, endpoint=True
        )
        sub_stfs[i, peak:end] = np.linspace(end - peak, 1, end - peak, endpoint=True)
        area = np.sum(sub_stfs[i, start:end] / srate_stf)
        if area > 0:
            sub_stfs[i, start:end] = sub_stfs[i, start:end] / area * m0[i]
        else:
            # duration shorter than one sample: impulse carrying the moment
            sub_stfs[i, start] = m0[i] * srate_stf
    return sub_stfs


def write_source_plane_csv(
    path_csv,
    lat_lon_dep,
    strike,
    dip,
    rake,
    length_strike,
    length_dip,
    slip_m,
    m0,
    t_rup,
    t_ris,
    t_fal,
    srate_stf,
    nx,
    nz,
):
    """
    Write one source_plane csv file.
    Input rows are ordered row by row along dip (nz rows of nx sub faults), the
    output rows are ordered along strike (nx columns of nz sub faults) as required
    by source_shapes = [nx, nz].

    Output columns:
    lat(deg), lon(deg), depth(km), strike(deg), dip(deg), rake(deg),
    length_strike(km), length_dip(km), slip(m), m0(Nm), stf(dimensionless)
    """
    N_sub = len(slip_m)
    if nx * nz != N_sub:
        raise ValueError(
            "nx (%d) * nz (%d) != number of sub faults (%d)" % (nx, nz, N_sub)
        )
    sub_stfs = create_triangle_stfs(t_rup, t_ris, t_fal, m0, srate_stf)
    source_plane = np.zeros((N_sub, 10 + sub_stfs.shape[1]))
    source_plane[:, :3] = lat_lon_dep
    source_plane[:, 3] = strike
    source_plane[:, 4] = dip
    source_plane[:, 5] = rake
    source_plane[:, 6] = length_strike
    source_plane[:, 7] = length_dip
    source_plane[:, 8] = slip_m
    source_plane[:, 9] = m0
    source_plane[:, 10:] = sub_stfs
    order = np.arange(N_sub).reshape(nz, nx).T.flatten()
    source_plane = source_plane[order, :]
    pd.DataFrame(source_plane).to_csv(str(path_csv), header=False, index=False)


def _fsp_value(line, key):
    """Value after 'key =' in a fsp header line, e.g. _fsp_value(line, 'STRK')."""
    match = re.search(
        r"(?<![A-Za-z_])%s\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        % re.escape(key),
        line,
    )
    return float(match.group(1)) if match else None


def convert_fsp2source_csvs(
    path_fsp, path_input_dir, sampling_interval_stf, rise_ratio=0.5
):
    """
    Convert a finite-fault model in FSP format (e.g. complete_inversion.fsp from
    the USGS website, single or multiple segments) to source_plane[m].csv files.

    Each SEGMENT is written to source_plane[k].csv (k starts from 1). The data
    columns are recognized by the header line
    "% LAT LON X==EW Y==NS Z SLIP RAKE TRUP RISE SF_MOMENT"; LAT, LON, Z, SLIP,
    TRUP, RISE and SF_MOMENT are required, RAKE defaults to the RAKE in "% Mech".

    :param path_fsp: Path to a .fsp format file.
    :param path_input_dir: The same as the path_input parameter in the ini file.
    :param sampling_interval_stf: Sampling interval of source time function to compute CFS.
    :param rise_ratio: The fsp format only provides the total duration RISE of the
        slip-rate function of each sub fault. The stf increases linearly during
        rise_ratio * RISE and decreases linearly during (1 - rise_ratio) * RISE,
        the same shape as convert_usgs_basic2source_csvs with
        t_ris = rise_ratio * RISE and t_fal = (1 - rise_ratio) * RISE.
    :return: source_shapes, [[nx, nz], ...] of each segment (along strike, along dip)

    Output columns in each csv file:
    lat(deg), lon(deg), depth(km), strike(deg), dip(deg), rake(deg),
    length_strike(km), length_dip(km), slip(m), m0(Nm), stf(dimensionless)
    """
    if not 0 <= rise_ratio <= 1:
        raise ValueError("rise_ratio must be in [0, 1]")
    srate_stf = 1 / sampling_interval_stf

    with open(path_fsp, "r") as fr:
        lines = fr.readlines()

    header = {}
    segments = []
    columns = None
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text.startswith("%"):
            upper = text.upper()
            if upper.startswith("% MECH"):
                header["strike"] = _fsp_value(upper, "STRK")
                header["dip"] = _fsp_value(upper, "DIP")
                header["rake"] = _fsp_value(upper, "RAKE")
            elif upper.startswith("% SIZE"):
                header["len"] = _fsp_value(upper, "LEN")
                header["wid"] = _fsp_value(upper, "WID")
            elif upper.startswith("% INVS"):
                for key in ("NX", "NZ", "DX", "DZ", "NSG"):
                    value = _fsp_value(upper, key)
                    if value is not None:
                        header[key.lower()] = value
            elif "SEGMENT #" in upper and "STRIKE" in upper:
                segments.append(
                    {
                        "strike": _fsp_value(upper, "STRIKE"),
                        "dip": _fsp_value(upper, "DIP"),
                        "rows": [],
                    }
                )
            elif segments and upper.startswith("% LEN") and "WID" in upper:
                segments[-1]["len"] = _fsp_value(upper, "LEN")
                segments[-1]["wid"] = _fsp_value(upper, "WID")
            elif " LAT " in upper + " " and " LON " in upper and " SLIP" in upper:
                columns = upper.lstrip("%").split()
            continue
        if columns is None:
            continue
        values = get_number_in_line(text)
        if len(values) != len(columns):
            continue
        if not segments:
            # single segment model without SEGMENT blocks
            segments.append(
                {
                    "strike": header.get("strike"),
                    "dip": header.get("dip"),
                    "len": header.get("len"),
                    "wid": header.get("wid"),
                    "rows": [],
                }
            )
        segments[-1]["rows"].append(values)

    if columns is None or not segments:
        raise ValueError("Can not find sub fault data in %s" % path_fsp)
    col = {name: i for i, name in enumerate(columns)}
    for name in ("LAT", "LON", "Z", "SLIP", "TRUP", "RISE", "SF_MOMENT"):
        if name not in col:
            raise ValueError("Column %s is required in the fsp file" % name)

    source_shapes = []
    k = 0
    for seg in segments:
        if not seg["rows"]:
            continue
        k += 1
        data = np.array(seg["rows"])
        N_sub = len(data)
        if seg["strike"] is None or seg["dip"] is None:
            raise ValueError("Strike/dip of segment %d not found" % k)
        # rows are ordered along dip: sub faults in the first row share the depth
        depth = data[:, col["Z"]]
        nx = int(np.argmax(np.abs(depth - depth[0]) > 1e-3)) or N_sub
        if N_sub % nx != 0:
            raise ValueError(
                "Can not determine the number of sub faults along strike of "
                "segment %d (%d sub faults, %d in the first row)" % (k, N_sub, nx)
            )
        nz = N_sub // nx
        length_strike = seg["len"] / nx if seg.get("len") else header.get("dx")
        length_dip = seg["wid"] / nz if seg.get("wid") else header.get("dz")
        rake = (
            data[:, col["RAKE"]]
            if "RAKE" in col
            else np.full(N_sub, header.get("rake"))
        )
        rise = data[:, col["RISE"]]
        write_source_plane_csv(
            path_csv=os.path.join(path_input_dir, "source_plane%d.csv" % k),
            lat_lon_dep=data[:, [col["LAT"], col["LON"], col["Z"]]],
            strike=seg["strike"],
            dip=seg["dip"],
            rake=rake,
            length_strike=length_strike,
            length_dip=length_dip,
            slip_m=data[:, col["SLIP"]],
            m0=data[:, col["SF_MOMENT"]],
            t_rup=data[:, col["TRUP"]],
            t_ris=rise_ratio * rise,
            t_fal=(1 - rise_ratio) * rise,
            srate_stf=srate_stf,
            nx=nx,
            nz=nz,
        )
        source_shapes.append([nx, nz])

    print("source_shapes=", source_shapes)
    print("convert fsp to input csv successfully")
    return source_shapes


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

    tectonic_stress_type = getattr(config, "tectonic_stress_type", None)
    if tectonic_stress_type in (1, 2):
        if tectonic_stress_type == 1:
            # tectonic_stress in Pa (NED, tension positive)
            st = tensor2full_tensor_matrix(config.tectonic_stress, "ned")
            eigenvalues, eigenvectors = np.linalg.eigh(st)
            index = eigenvalues.argsort()  # most compressive first
            # Coulomb3: compression positive, unit bar
            intensities = -eigenvalues[index] / 1e5
            axes = eigenvectors[:, index]
        else:
            # principal axes ordered from the smallest to the largest principal
            # stress (tension positive), magnitudes are not provided
            ts = np.asarray(config.tectonic_stress, dtype=float)
            axes = np.zeros((3, 3))
            for i in range(3):
                phi = np.deg2rad(ts[2 * i])
                delta = np.deg2rad(ts[2 * i + 1])
                axes[:, i] = [
                    np.cos(phi) * np.cos(delta),
                    np.sin(phi) * np.cos(delta),
                    np.sin(delta),
                ]
            intensities = np.array([100.0, 30.0, 0.0])
            print(
                "tectonic_stress_type=2 gives no stress magnitudes, "
                "S1IN/S2IN/S3IN are set to 100/30/0 bar."
            )
        lines_regional_stress = []
        for i in range(3):
            n = axes[:, i] / np.linalg.norm(axes[:, i])  # NED
            if n[2] < 0:
                n = -n
            azimuth = np.rad2deg(np.arctan2(n[1], n[0])) % 360
            plunge = np.rad2deg(np.arcsin(np.clip(n[2], -1, 1)))
            S_ind = i + 1
            lines_regional_stress.append(
                "S%dDR=%15.3f S%dDP=%15.3f S%dIN=%15.3f S%dGD=%15.3f\n"
                % (
                    S_ind,
                    azimuth,
                    S_ind,
                    plunge,
                    S_ind,
                    intensities[i],
                    S_ind,
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
