import os
import platform
import subprocess

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from .signal_process import linear_interp
from .geo import d2km


def read_source_array(source_inds, path_input, shift2corner=False):
    source_array = None
    for ind_src in range(len(source_inds)):
        source_plane = pd.read_csv(
            str(os.path.join(path_input, "source_plane%d.csv" % source_inds[ind_src])),
            index_col=False,
            header=None,
        ).to_numpy()
        if shift2corner:
            # source_plane columns:
            # 3: strike(deg), 4: dip(deg)
            # 6: length_strike(km), 7: length_dip(km)
            strike_rad = np.deg2rad(source_plane[:, 3])
            dip_rad = np.deg2rad(source_plane[:, 4])
            length_strike_km = source_plane[:, 6]
            length_dip_km = source_plane[:, 7]

            # NED-xyz
            # Strike vector (s_vec)
            s_vec_x = np.cos(strike_rad)  # North
            s_vec_y = np.sin(strike_rad)  # East

            # Dip vector (d_vec)
            d_vec_x = -np.sin(strike_rad) * np.cos(dip_rad)  # North
            d_vec_y = np.cos(strike_rad) * np.cos(dip_rad)  # East
            d_vec_z = np.sin(dip_rad)  # Down

            # Calculate the shift vector from the center to the top-left corner.
            shift_x = -0.5 * (s_vec_x * length_strike_km + d_vec_x * length_dip_km)
            shift_y = -0.5 * (s_vec_y * length_strike_km + d_vec_y * length_dip_km)
            shift_z = -0.5 * (d_vec_z * length_dip_km)

            source_plane[:, 0] = source_plane[:, 0] + shift_x / d2km
            source_plane[:, 1] = source_plane[:, 1] + shift_y / d2km
            source_plane[:, 2] = source_plane[:, 2] + shift_z

        if ind_src == 0:
            source_array = source_plane.copy()
        else:
            source_array = np.concatenate([source_array, source_plane.copy()], axis=0)
    return source_array


def reshape_sub_faults(sub_faults, sub_fms, sub_lengths, num_strike, num_dip):
    """
    Calculate the four corner coordinates for each sub-fault and return them as a mesh.
    This function is suitable for both planar and curved faults.

    Parameters:
        sub_faults (np.ndarray): Array of sub-fault center coordinates (N, 3) in NED, units in meters.
        sub_fms (np.ndarray): Array with strike (deg) and dip (deg) for each sub-fault (N, 2).
        sub_lengths (np.ndarray): Array with length_strike (m) and length_dip (m) for each sub-fault (N, 2).
        num_strike (int): Number of sub-faults along strike.
        num_dip (int): Number of sub-faults along dip.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: X, Y, Z coordinate meshes of shape (num_strike + 1, num_dip + 1).
    """

    strike_rad = np.deg2rad(sub_fms[:, 0])
    dip_rad = np.deg2rad(sub_fms[:, 1])
    length_strike_m = sub_lengths[:, 0]
    length_dip_m = sub_lengths[:, 1]

    # NED-xyz
    # Strike vector (s_vec)
    s_vec_x = np.cos(strike_rad) * length_strike_m  # North
    s_vec_y = np.sin(strike_rad) * length_strike_m  # East

    # Dip vector (d_vec)
    d_vec_x = -np.sin(strike_rad) * np.cos(dip_rad) * length_dip_m  # North
    d_vec_y = np.cos(strike_rad) * np.cos(dip_rad) * length_dip_m  # East
    d_vec_z = np.sin(dip_rad) * length_dip_m  # Down

    sub_faults[:, 0] = sub_faults[:, 0] - 0.5 * (s_vec_x + d_vec_x)
    sub_faults[:, 1] = sub_faults[:, 1] - 0.5 * (s_vec_y + d_vec_y)
    sub_faults[:, 2] = sub_faults[:, 2] - 0.5 * d_vec_z

    X = np.zeros((num_strike + 1, num_dip + 1))
    Y = np.zeros((num_strike + 1, num_dip + 1))
    Z = np.zeros((num_strike + 1, num_dip + 1))

    X[:-1, :-1] = sub_faults[:, 0].reshape(num_strike, num_dip)
    Y[:-1, :-1] = sub_faults[:, 1].reshape(num_strike, num_dip)
    Z[:-1, :-1] = sub_faults[:, 2].reshape(num_strike, num_dip)

    X[-1, :-1] = X[-2, :-1] + s_vec_x.reshape(num_strike, num_dip)[-1, :]
    Y[-1, :-1] = Y[-2, :-1] + s_vec_y.reshape(num_strike, num_dip)[-1, :]
    Z[-1, :-1] = Z[-2, :-1]

    X[:-1, -1] = X[:-1, -2] + d_vec_x.reshape(num_strike, num_dip)[:, -1]
    Y[:-1, -1] = Y[:-1, -2] + d_vec_y.reshape(num_strike, num_dip)[:, -1]
    Z[:-1, -1] = Z[:-1, -2] + d_vec_z.reshape(num_strike, num_dip)[:, -1]

    X[-1, -1] = X[-2, -1] + s_vec_x.reshape(num_strike, num_dip)[-1, -1]
    Y[-1, -1] = Y[-2, -1] + s_vec_y.reshape(num_strike, num_dip)[-1, -1]
    Z[-1, -1] = Z[-2, -1]

    return X, Y, Z


def reshape_sub_faults_flat(sub_faults, num_strike, num_dip, zoom_x, zoom_y):
    mu_strike = sub_faults[num_dip] - sub_faults[0]
    mu_dip = sub_faults[1] - sub_faults[0]
    sub_faults = sub_faults - mu_strike / 2 - mu_dip / 2
    X: np.ndarray = sub_faults[:, 0]
    Y: np.ndarray = sub_faults[:, 1]
    Z: np.ndarray = sub_faults[:, 2]

    X = X.reshape(num_strike, num_dip)
    Y = Y.reshape(num_strike, num_dip)
    Z = Z.reshape(num_strike, num_dip)

    X = zoom(X, [zoom_x, zoom_y])
    Y = zoom(Y, [zoom_x, zoom_y])
    Z = zoom(Z, [zoom_x, zoom_y])

    X = np.concatenate([X, np.array([X[:, -1] + mu_dip[0]]).T], axis=1)
    Y = np.concatenate([Y, np.array([Y[:, -1] + mu_dip[1]]).T], axis=1)
    Z = np.concatenate([Z, np.array([Z[:, -1] + mu_dip[2]]).T], axis=1)

    X = np.concatenate([X, np.array([X[-1, :] + mu_strike[0]])], axis=0)
    Y = np.concatenate([Y, np.array([Y[-1, :] + mu_strike[1]])], axis=0)
    Z = np.concatenate([Z, np.array([Z[-1, :] + mu_strike[2]])], axis=0)

    return X, Y, Z


def group(inp_list, num_in_each_group):
    group_list = []
    for i in range(len(inp_list) // num_in_each_group):
        group_list.append(inp_list[i * num_in_each_group: (i + 1) * num_in_each_group])
    rest = len(inp_list) % num_in_each_group
    if rest != 0:
        group_list.append(inp_list[-rest:])
    return group_list


def shift_green2real_tpts(
        seismograms,
        tpts_table,
        green_before_p,
        srate,
        event_depth_km,
        dist_in_km,
        receiver_depth_km=0,
        model_name="ak135",
):
    from .pytaup import cal_first_p_s
    first_p, first_s = cal_first_p_s(
        event_depth_km=event_depth_km,
        dist_km=dist_in_km,
        receiver_depth_km=receiver_depth_km,
        model_name=model_name,
    )
    p_count = round(green_before_p * srate)
    s_count = round(
        (tpts_table["s_onset"] - tpts_table["p_onset"] + green_before_p) * srate
    )
    p_count_new = round((first_p - tpts_table["p_onset"] + green_before_p) * srate)
    s_count_new = min(
        len(seismograms[0]),
        round((first_s - tpts_table["p_onset"] + green_before_p) * srate),
    )
    for i in range(seismograms.shape[0]):
        green_before_p = seismograms[i][:p_count]
        p_s = linear_interp(seismograms[i][p_count:s_count], s_count_new - p_count_new)
        after_s = seismograms[i][s_count:]
        if len(after_s) > 0:
            after_s = linear_interp(
                after_s, len(seismograms[i]) - len(green_before_p) - len(p_s)
            )
            seismograms[i] = np.concatenate([green_before_p, p_s, after_s])
        else:
            seismograms[i] = np.concatenate([green_before_p, p_s])[
                : len(seismograms[i])
            ]

    return seismograms, first_p, first_s


def cal_max_dist_from_2d_points(A: np.ndarray, B: np.ndarray):
    """
    :param A: (m,2)
    :param B: (n,2)
    :return: max_distance
    """
    differences = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    squared_distances = np.sum(differences ** 2, axis=2)
    distances = np.sqrt(squared_distances)
    max_distance = np.max(distances)
    return max_distance


def cal_min_max_dist_from_points(A: np.ndarray, B: np.ndarray):
    """
    :param A: (m,3) or (m,2)
    :param B: (n,3) or (n,2)
    :return: dmin, dmax, (imin, jmin), (imax, jmax)
    """
    diff = A[:, None, :] - B[None, :, :]
    dist2 = np.einsum('mni,mni->mn', diff, diff)  # 等同于 (diff**2).sum(axis=-1)，更高效
    jmin = dist2.argmin()
    jmax = dist2.argmax()
    m_, n_ = dist2.shape
    imin, jmin = divmod(jmin, n_)
    imax, jmax = divmod(jmax, n_)
    return (np.sqrt(dist2[imin, jmin]), np.sqrt(dist2[imax, jmax]),
            (imin, jmin), (imax, jmax))


def create_rotate_z_mat(gamma):
    """
    Generates a rotation matrix about the Z-axis.

    Parameters:
        gamma : float
            Rotation angle in radians.

    Returns:
        R : numpy.ndarray
            A 3x3 rotation matrix.
    """
    R = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )
    return R


def rotate_symmetric_tensor_series(tensor, gamma):
    """
    Rotates a series of symmetric tensors without using an explicit loop.

    Parameters:
        tensor: numpy array of shape (n, 6)
            Each row is [xx, xy, xz, yy, yz, zz] representing a symmetric tensor.
        gamma: float
            Rotation angle (in radians) used to create the rotation matrix.

    Returns:
        rotated_tensor: numpy array of shape (n, 6)
            Rotated tensor components in the same order as the input.
    """
    # Create the 3x3 rotation matrix (assumed to be defined elsewhere).
    R = create_rotate_z_mat(gamma)
    n = tensor.shape[0]

    # Construct full symmetric matrices from the condensed tensor representation.
    A = np.empty((n, 3, 3), dtype=tensor.dtype)
    A[:, 0, 0] = tensor[:, 0]
    A[:, 0, 1] = tensor[:, 1]
    A[:, 0, 2] = tensor[:, 2]
    A[:, 1, 0] = tensor[:, 1]
    A[:, 1, 1] = tensor[:, 3]
    A[:, 1, 2] = tensor[:, 4]
    A[:, 2, 0] = tensor[:, 2]
    A[:, 2, 1] = tensor[:, 4]
    A[:, 2, 2] = tensor[:, 5]

    # Rotate each tensor using batch matrix multiplication:
    # Compute rotated_A = R.T @ A @ R for each tensor.
    rotated_A = np.einsum("ij,njk,kl->nil", R.T, A, R)

    # Extract the independent components from the rotated tensors.
    rotated_tensor = np.empty((n, 6), dtype=tensor.dtype)
    rotated_tensor[:, 0] = rotated_A[:, 0, 0]
    rotated_tensor[:, 1] = rotated_A[:, 0, 1]
    rotated_tensor[:, 2] = rotated_A[:, 0, 2]
    rotated_tensor[:, 3] = rotated_A[:, 1, 1]
    rotated_tensor[:, 4] = rotated_A[:, 1, 2]
    rotated_tensor[:, 5] = rotated_A[:, 2, 2]

    return rotated_tensor


def convert_earth_model_nd2inp(path_nd, path_output):
    with open(path_nd, "r") as fr:
        lines = fr.readlines()
    lines_new = []
    for i in range(len(lines)):
        temp = lines[i].split()
        if len(temp) > 1:
            lines_new.append(temp)
    for i in range(len(lines_new)):
        # print(lines_new[i])
        lines_new[i] = "  ".join([str(int(i + 1))] + lines_new[i]) + "\n"  # type:ignore
    # with open(path_output, "w") as fw:
    #     fw.writelines(lines_new)
    return lines_new


def convert_earth_model_nd2nd_without_Q(path_nd, path_output, epsilon=0):
    with open(path_nd, "r") as fr:
        lines = fr.readlines()
    data = []
    for i in range(len(lines)):
        temp = lines[i].split()
        if len(temp) > 1:
            data.append([float(_) for _ in temp[:-2]])
    data = np.array(data)
    for i in range(len(data) - 2):
        if (
                (data[i, 0] == data[i + 1, 0])
                and (data[i, 2] != 0)
                and (data[i + 1, 2] != 0)
        ):
            data[i + 1, 0] = data[i + 1, 0] + epsilon

    ind_data = 0
    lines_new = []
    for i in range(len(lines)):
        temp = lines[i].split()
        if len(temp) > 1:
            lines_new.append(["%10.5f" % float(_) for _ in data[ind_data]])
            lines_new[i] = "  ".join(lines_new[i]) + "\n"
            ind_data += 1
        else:
            lines_new.append(lines[i].strip() + "\n")
    with open(path_output, "w") as fw:
        fw.writelines(lines_new)
    return lines_new


def read_nd(path_nd, with_Q=False):
    with open(path_nd, "r") as fr:
        lines = fr.readlines()
    lines_new = []
    for i in range(len(lines)):
        temp = lines[i].split()
        if len(temp) > 1:
            for j in range(len(temp)):
                lines_new.append(float(temp[j]))
    if with_Q:
        nd_model = np.array(lines_new).reshape(-1, 6)
    else:
        nd_model = np.array(lines_new).reshape(-1, 4)
    return nd_model


def read_layerd_material(path_layerd_dat, depth_in_km):
    # thickness, rho, vp, vs, qp, qs
    depth_in_m = depth_in_km * 1e3
    dat = np.loadtxt(path_layerd_dat)
    ind = np.argwhere((np.cumsum(dat[:, 0]) - depth_in_m) >= 0)[0][0]
    return dat[ind]


def create_stf(tau, srate):
    t = np.linspace(0, tau, round(tau * srate) + 1, endpoint=True)
    stf = (2 / tau) * (np.sin(np.pi * t / tau)) ** 2
    return stf


def group_planes(strike_array):
    """
    It is necessary to ensure that the sub faults on
    the same fault plane have the same strikes!!!
    :param strike_array: numpy array

    Returns:
    np.array: An array containing the lengths of each group.
    """
    # Find the indices where the value changes
    # a[1:] != a[:-1] produces a boolean array that's True
    # at positions where a value differs from its predecessor.
    change_indices = np.where(strike_array[1:] != strike_array[:-1])[0] + 1

    # Include the start and end indices to get boundaries for each group.
    boundaries = np.concatenate(([0], change_indices, [len(strike_array)]))

    # The difference between consecutive boundaries gives the group lengths.
    lengths = np.diff(boundaries)
    return lengths


def bool2int(input_bool):
    if input_bool:
        return 1
    else:
        return 0


def call_exe(path_green, path_inp, path_finished, name):
    name_exe = "%s.exe" % name if platform.system() == "Windows" else "%s.bin" % name
    path_exe = os.path.join(path_green, name_exe)
    proc = subprocess.Popen(
        [path_exe],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = proc.communicate(str.encode(path_inp))
    stdout_text = stdout_bytes.decode(errors="ignore")
    stderr_text = stderr_bytes.decode(errors="ignore")
    output = stdout_text + stderr_text
    with open(path_finished, "w") as fw:
        fw.writelines(output)
        return None


def ignore_slip_source_array(source_array, slip_thresh):
    source_array = source_array.copy()
    inds_ignore_slip = np.argwhere(source_array[:, 8] < slip_thresh)
    source_array[inds_ignore_slip, 8] = 0
    source_array[inds_ignore_slip, 9] = 0
    return source_array


def cut_stf_modify_source_array(source_array, cut_stf):
    source_array = source_array.copy()
    sub_stfs = source_array[:, 10:].copy()
    source_array[:, 10 + cut_stf:] = 0
    m0_stf_origin = np.sum(sub_stfs, axis=1)
    m0_stf_cut = np.sum(source_array[:, 10:], axis=1)
    cut_ratio = np.zeros_like(m0_stf_cut)
    inds_not_0 = np.argwhere(m0_stf_origin != 0).flatten()
    cut_ratio[inds_not_0] = m0_stf_cut[inds_not_0] / m0_stf_origin[inds_not_0]
    source_array[:, 8] = cut_ratio * source_array[:, 8]
    source_array[:, 9] = cut_ratio * source_array[:, 9]
    return source_array


if __name__ == "__main__":
    pass
