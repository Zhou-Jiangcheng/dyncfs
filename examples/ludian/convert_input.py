import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dyncfs.focal_mechanism import plane2mt, mt2plane
from dyncfs.utils import read_nd
from dyncfs.signal_process import resample

if __name__ == "__main__":
    df = pd.read_csv("ludian_joint.txt", sep="\\s+", header=None, index_col=False)
    data = df.to_numpy()
    nd_model = read_nd("/home/zjc/python_works/dyncfs/test/ak135fc.nd", with_Q=True)
    path_input = "/home/zjc/dyncfs_data/ludian/input"

    N = len(data) // 2
    dep_list = data[:N, 2]
    slip_strike_list = data[:N, 8]
    slip_dip_list = data[N:, 8]
    slip_rate_strike_list = data[:N, 10:]
    slip_rate_dip_list = data[N:, 10:]
    mu_list = np.zeros(N)
    rake_list = np.zeros(N)
    slip_list = np.zeros(N)
    m0_list = np.zeros(N)
    stf_list = np.zeros((N, slip_rate_dip_list.shape[1] * 2))

    sub_len = 2000  # m
    sub_area = sub_len**2
    dep_nd_list = nd_model[:, 0]
    mt = np.zeros(6)
    for i in range(N):
        dep_nd, vp, vs, rho, _, _ = list(
            nd_model[np.argmin(np.abs(dep_nd_list - dep_list[i]))]
        )
        # print(i, dep_list[i], dep_nd, vp, vs, rho)
        mu_list[i] = rho * 1e3 * (vs * 1e3) ** 2
        slip_dip = slip_dip_list[i]
        slip_strike = slip_strike_list[i]
        slip_list[i] = np.sqrt(np.sum(slip_dip**2 + slip_strike**2))
        if slip_strike != 0:
            rake_list[i] = np.rad2deg(np.arctan(slip_dip / slip_strike))
        elif slip_dip >= 0:
            rake_list[i] = 90
        else:
            rake_list[i] = -90
        m0_list[i] = mu_list[i] * sub_area * slip_list[i]
        temp = np.sqrt(slip_rate_dip_list[i] ** 2 + slip_rate_strike_list[i] ** 2)
        stf_list[i] = resample(temp, srate_old=4, srate_new=8)

        if np.sum(stf_list[i]) > 0:
            stf_list[i] = stf_list[i] / np.sum(stf_list[i] / 2) * m0_list[i]
        print(data[i, 5], data[i, 6], rake_list[i], slip_list[i])
        mt_i = plane2mt(
            M0=m0_list[i], strike=data[i, 5], dip=data[i, 6], rake=rake_list[i]
        )
        mt = mt + mt_i

    data_new = np.zeros((N, 10))
    data_new[:, :3] = data[:N, :3].copy()
    data_new[:, 3] = data[:N, 5]
    data_new[:, 4] = data[:N, 6]
    data_new[:, 5] = rake_list.copy()
    data_new[:, 6] = np.ones(N) * sub_len / 1e3
    data_new[:, 7] = np.ones(N) * sub_len / 1e3
    data_new[:, 8] = slip_list
    data_new[:, 9] = m0_list
    data_new = np.concatenate([data_new, stf_list], axis=1)

    strike_list = data_new[:N, 3]
    segments = np.where(np.insert(np.diff(strike_list) != 0, 0, True))[0]

    # 按照连续相同元素切割
    split_data = [
        (
            data_new[segments[i] : segments[i + 1]]
            if i + 1 < len(segments)
            else data_new[segments[i] :]
        )
        for i in range(len(segments))
    ]
    for i, segment in enumerate(split_data):
        print(i, len(segment))
        df_i = pd.DataFrame(segment, index=None)
        path_i = str(os.path.join(path_input, "source_plane%d.csv" % (i + 1)))
        df_i.to_csv(path_i, index=False, header=False)
    stf = np.sum(stf_list, axis=0)
    fm, _ = mt2plane(mt)[:2]
    print(2 / 3 * (np.log10(np.sum(m0_list)) - 9.1))
    print(2 / 3 * (np.log10(np.sum(stf) / 2) - 9.1))
    print(fm, _)
    print(mt2plane(mt))

    from obspy.imaging.beachball import beachball
    from dyncfs.focal_mechanism import convert_mt_axis

    bb = beachball(
        convert_mt_axis(plane2mt(1, fm[0], fm[1], fm[2]), convert_flag="ned2rtp")
    )
    bb = beachball(convert_mt_axis(mt, convert_flag="ned2rtp"))
    plt.figure()
    plt.plot(stf)
    plt.show()
