import os

import numpy as np
from scipy.ndimage import zoom
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize

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


def plot_staic_coulomb_stress_3d(
        elev,
        azim,
        path_input,
        path_output,
        obs_inds,
        obs_shapes,
        sub_length_strike,
        sub_length_dip,
        color_saturation=None,
        # zoom_x=1,
        # zoom_y=1,
        show=True,
        save=True,
):
    if not show:
        matplotlib.use("Agg")
    max_stress_abs = -np.inf
    stress_list = []
    ref = None
    for ind_obs in range(len(obs_inds)):
        obs_plane = pd.read_csv(
            str(os.path.join(path_input, "obs_plane%d.csv" % obs_inds[ind_obs])),
            index_col=False,
            header=None,
        ).to_numpy()
        sub_faults = obs_plane[:, :3]
        sub_fms = obs_plane[:, 3:6]
        sub_lengths = np.concatenate(
            [
                np.ones((len(obs_plane), 1)) * sub_length_strike,
                np.ones((len(obs_plane), 1)) * sub_length_dip,
            ], axis=1
        )
        if ref is None:
            ref = sub_faults[0].tolist()
        sub_faults = convert_sub_faults_geo2ned(sub_faults=sub_faults, source_point=ref)
        # sub_faults[:, 2] = sub_faults[:, 2] + source1[2]

        X, Y, Z = reshape_sub_faults(
            sub_faults=sub_faults,
            sub_fms=sub_fms,
            sub_lengths=sub_lengths * 1e3,
            num_strike=obs_shapes[ind_obs][0],
            num_dip=obs_shapes[ind_obs][1],
            # zoom_x=zoom_x,
            # zoom_y=zoom_y,
        )
        X = X / 1e3
        Y = Y / 1e3
        Z = Z / 1e3

        sub_stress = pd.read_csv(
            str(
                os.path.join(
                    path_output,
                    "results",
                    "static",
                    "cfs_static_plane%d.csv" % obs_inds[ind_obs],
                )
            ),
            index_col=False,
            header=None,
        ).to_numpy()
        sub_stress: np.ndarray = sub_stress.reshape(
            obs_shapes[ind_obs][0], obs_shapes[ind_obs][1]
        )
        # sub_stress = zoom(sub_stress, [zoom_x, zoom_y])
        stress_list.append([X, Y, Z, sub_stress])
        if np.max(np.abs(sub_stress)) > max_stress_abs:
            max_stress_abs = np.max(np.abs(sub_stress))
    if color_saturation is None:
        color_saturation = max_stress_abs
    tick_range = [-color_saturation / 1e6, color_saturation / 1e6]
    cmap = matplotlib.colormaps["seismic"]
    norm = Normalize(vmin=tick_range[0], vmax=tick_range[1])

    plt.ioff()
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(20 / 2.54, 20 / 2.54),
        subplot_kw={"projection": "3d"},
    )
    ax.view_init(elev=elev, azim=azim)  # type:ignore
    for ind_obs in range(len(obs_inds)):
        ax.plot_surface(  # type:ignore
            stress_list[ind_obs][1],
            stress_list[ind_obs][0],
            -stress_list[ind_obs][2],
            rstride=1,
            cstride=1,
            facecolors=cmap(norm(stress_list[ind_obs][-1] / 1e6)),
            antialiased=False,
            shade=False,
            edgecolor="gray",
            linewidth=0.1,
        )

        # 提取坐标
        X = stress_list[ind_obs][1]
        Y = stress_list[ind_obs][0]
        Z = -stress_list[ind_obs][2]

        # 绘制四个边界
        ax.plot(X[0, :], Y[0, :], Z[0, :], color="k", linewidth=1, zorder=10)
        ax.plot(X[-1, :], Y[-1, :], Z[-1, :], color="k", linewidth=1, zorder=10)
        ax.plot(X[:, 0], Y[:, 0], Z[:, 0], color="k", linewidth=1, zorder=10)
        ax.plot(X[:, -1], Y[:, -1], Z[:, -1], color="k", linewidth=1, zorder=10)

    cax = inset_axes(
        ax,
        width="3%",
        height="75%",
        loc="upper right",
        bbox_to_anchor=(0.15, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
    )
    m = cm.ScalarMappable(cmap=cmap)
    m.set_clim(tick_range[0], tick_range[1])
    cbar = fig.colorbar(m, cax=cax)
    cbar.set_label("Coulomb Failure Stress Change (MPa)")
    ax.set_box_aspect([1.0, 1.0, 0.2])  # type:ignore
    ax.set_xlabel("W-E (km)")
    ax.set_ylabel("S-N (km)")
    ax.set_zlabel("D-U (km)")  # type:ignore
    fig.subplots_adjust(left=0, right=0.75, bottom=0, top=1)
    title = "Static Coulomb Failure Stress Change"
    fig.suptitle(title)
    if save:
        plt.savefig(
            os.path.join(
                path_output,
                "results",
                "static",
                "cfs_3d_static.png",
            ),
            dpi=600,
        )
    if show:
        plt.ion()
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    pass
