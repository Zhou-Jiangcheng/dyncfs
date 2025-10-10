import os
import warnings

import numpy as np
from scipy.ndimage import zoom
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize, LinearSegmentedColormap

from .utils import reshape_sub_faults
from .geo import convert_sub_faults_geo2ned

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "Arial",
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def plot_slip_3d(
    elev,
    azim,
    nt_cut,
    sampling_interval_stf,
    path_input,
    plane_inds,
    plane_shapes,
    slip_thresh=0,
    color_saturation=None,
    # zoom_x=1,
    # zoom_y=1,
    save=True,
    show=True,
):
    """
    Slip distribution plot, cumulative slip before nt_cut.
    """
    if not show:
        matplotlib.use("Agg")
    max_slip = -np.inf
    slip_list = []
    ref = None
    for ind_obs in range(len(plane_inds)):
        source_array = pd.read_csv(
            str(os.path.join(path_input, "source_plane%d.csv" % plane_inds[ind_obs])),
            index_col=False,
            header=None,
        ).to_numpy()
        sub_faults = source_array[:, :3]
        sub_fms = source_array[:, 3:6]
        sub_lengths = source_array[:, 6:8]
        if ref is None:
            ref = sub_faults[0].tolist()
        sub_faults = convert_sub_faults_geo2ned(sub_faults=sub_faults, source_point=ref)
        # sub_faults[:, 2] = sub_faults[:, 2] + source1[2]

        X, Y, Z = reshape_sub_faults(
            sub_faults=sub_faults,
            sub_fms=sub_fms,
            sub_lengths=sub_lengths * 1e3,
            num_strike=plane_shapes[ind_obs][0],
            num_dip=plane_shapes[ind_obs][1],
            # zoom_x=zoom_x,
            # zoom_y=zoom_y,
        )
        X = X / 1e3
        Y = Y / 1e3
        Z = Z / 1e3
        if slip_thresh > 0:
            inds_ignore_slip = np.where(source_array[:, 8] < slip_thresh)
            source_array[inds_ignore_slip, 8] = 0
            source_array[inds_ignore_slip, 9] = 0
        if 0 < nt_cut < len(source_array[0, 10:]):
            sub_stfs = source_array[:, 10:].copy()
            source_array[:, 10 + nt_cut :] = 0
            m0_stf_origin = np.sum(sub_stfs, axis=1)
            m0_stf_cut = np.sum(source_array[:, 10:], axis=1)
            cut_ratio = np.zeros_like(m0_stf_cut)
            inds_not_0 = np.argwhere(m0_stf_origin != 0).flatten()
            cut_ratio[inds_not_0] = m0_stf_cut[inds_not_0] / m0_stf_origin[inds_not_0]
            sub_slip = cut_ratio * source_array[:, 8]
        else:
            # warnings.warn('nt_cut is ignored, plot the final slip distribution')
            nt_cut = len(source_array[0, 10:])
            sub_slip = source_array[:, 8]

        sub_slip: np.ndarray = sub_slip.reshape(
            plane_shapes[ind_obs][0], plane_shapes[ind_obs][1]
        )
        # sub_slip = zoom(sub_slip, [zoom_x, zoom_y])
        # print(X.shape, sub_slip.shape)
        slip_list.append([X, Y, Z, sub_slip])
        if np.max(sub_slip) > max_slip:
            max_slip = np.max(sub_slip)
    if color_saturation is None:
        color_saturation = max_slip
    colors = ['white', 'cyan', 'orange', 'red']

    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    norm = Normalize(0, color_saturation)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        # figsize=(20 / 2.54, 20 / 2.54),
        subplot_kw={"projection": "3d"},
    )
    ax.view_init(elev=elev, azim=azim)  # type:ignore
    for ind_obs in range(len(plane_inds)):
        ax.plot_surface(  # type:ignore
            slip_list[ind_obs][1],
            slip_list[ind_obs][0],
            -slip_list[ind_obs][2],
            rstride=1,
            cstride=1,
            facecolors=cmap(norm(slip_list[ind_obs][-1])),
            antialiased=False,
            shade=False,
            edgecolor="gray",
            linewidth=0.1,
        )

        # 提取坐标
        X = slip_list[ind_obs][1]
        Y = slip_list[ind_obs][0]
        Z = -slip_list[ind_obs][2]

        # 绘制四个边界
        ax.plot(X[0, :], Y[0, :], Z[0, :], color="k", linewidth=1, zorder=10)
        ax.plot(X[-1, :], Y[-1, :], Z[-1, :], color="k", linewidth=1, zorder=10)
        ax.plot(X[:, 0], Y[:, 0], Z[:, 0], color="k", linewidth=1, zorder=10)
        ax.plot(X[:, -1], Y[:, -1], Z[:, -1], color="k", linewidth=1, zorder=10)

    cax = inset_axes(
        ax,
        width="5%",
        height="75%",
        loc="upper right",
        bbox_to_anchor=(0.2, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
    )
    m = cm.ScalarMappable(cmap=cmap)
    m.set_clim(0, color_saturation)
    cbar = fig.colorbar(m, cax=cax)
    cbar.set_label("Slip (m)")
    ax.set_box_aspect([1.0, 1.0, 0.2])  # type:ignore
    ax.set_xlabel("W-E (km)")
    ax.set_ylabel("S-N (km)")
    ax.set_zlabel("D-U (km)")  # type:ignore
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    title = "Slip Distribution before Time: %.2f s" % float(
        nt_cut * sampling_interval_stf
    )
    fig.suptitle(title)
    if save:
        plt.savefig(
            os.path.join(
                path_input,
                "slip_3d_nt_cut_%d.png" % nt_cut,
            ),
            dpi=600,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    pass
