import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe

from dyncfs.utils import reshape_sub_faults
from dyncfs.geo import convert_sub_faults_geo2ned

plt.rcParams.update(
    {
        "font.size": 10,
    }
)

if __name__ == "__main__":
    path_input = "./input"
    elev = 20
    azim = 170
    nt_cut = -1
    sampling_interval_stf = 0.5
    path_input = path_input
    plane_inds = [1, 2, 3, 4, 5]
    plane_shapes = [[22, 9], [6, 9], [8, 9], [62, 9], [17, 6]]

    color_saturation = 9
    save = True
    show = True
    slip_thresh = 0

    ref = None
    max_slip = -np.inf
    slip_list = []

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
    tick_range = [0, 9]
    tick_interval = 1
    colors = ["blue", "cyan", "orange", "red"]
    from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    boundaries = np.linspace(
        tick_range[0],
        tick_range[1],
        round((tick_range[1] - tick_range[0]) / tick_interval) + 1,
    )
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(20 / 2.54, 20 / 2.54),
        subplot_kw={"projection": "3d"},
    )
    ax.view_init(elev=elev, azim=azim)  # type:ignore
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False  # 不填充背景
        # axis.pane.set_edgecolor('none')  # 去掉边框线
    for ind_obs in range(len(plane_inds)):
        if plane_inds[ind_obs] == 4:

            ax.text(
                slip_list[ind_obs][1][25, 3],
                slip_list[ind_obs][0][25, 3],
                -slip_list[ind_obs][2][25, 3] + 1e-3,
                "$\\star$",
                fontsize=24,
                ha="center",
                va="center",
                color="red",
                zorder=100,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )
        if (
            plane_inds[ind_obs] != 5
            and plane_inds[ind_obs] != 100
            and plane_inds[ind_obs] != 200
        ):
            ax.plot_surface(  # type:ignore
                slip_list[ind_obs][1],
                slip_list[ind_obs][0],
                -slip_list[ind_obs][2],
                rstride=1,
                cstride=1,
                facecolors=cmap(norm(slip_list[ind_obs][-1])),
                antialiased=True,
                shade=False,
                edgecolor="gray",
                linewidth=0.1,
            )
            X = slip_list[ind_obs][1]
            Y = slip_list[ind_obs][0]
            Z = -slip_list[ind_obs][2]

            ax.plot(X[0, :], Y[0, :], Z[0, :], color="k", linewidth=0.5, zorder=10)
            ax.plot(X[-1, :], Y[-1, :], Z[-1, :], color="k", linewidth=0.5, zorder=10)
            ax.plot(X[:, 0], Y[:, 0], Z[:, 0], color="k", linewidth=0.5, zorder=10)
            ax.plot(X[:, -1], Y[:, -1], Z[:, -1], color="k", linewidth=0.5, zorder=10)

            ax.text(
                slip_list[ind_obs][1][3, -3],
                slip_list[ind_obs][0][3, -3],
                -slip_list[ind_obs][2][3, -3],
                "S%d" % plane_inds[ind_obs],
                fontsize=8,
                ha="center",
                va="center",
                color="black",
                zorder=100,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )
        if plane_inds[ind_obs] == 5:
            ax.plot_surface(  # type:ignore
                slip_list[ind_obs][1],
                slip_list[ind_obs][0],
                -slip_list[ind_obs][2],
                rstride=1,
                cstride=1,
                antialiased=True,
                shade=False,
                edgecolor="gray",
                linewidth=0.1,
                alpha=0.3,
            )
            Z_new = slip_list[ind_obs][2] - 40
            X = slip_list[ind_obs][1]
            Y = slip_list[ind_obs][0]
            Z = -Z_new

            ax.plot(X[0, :], Y[0, :], Z[0, :], color="k", linewidth=0.5, zorder=10)
            ax.plot(X[-1, :], Y[-1, :], Z[-1, :], color="k", linewidth=0.5, zorder=10)
            ax.plot(X[:, 0], Y[:, 0], Z[:, 0], color="k", linewidth=0.5, zorder=10)
            ax.plot(X[:, -1], Y[:, -1], Z[:, -1], color="k", linewidth=0.5, zorder=10)

            ax.plot_surface(  # type:ignore
                slip_list[ind_obs][1],
                slip_list[ind_obs][0],
                -Z_new,
                rstride=1,
                cstride=1,
                facecolors=cmap(norm(slip_list[ind_obs][-1])),
                antialiased=True,
                shade=False,
                edgecolor="gray",
                linewidth=0.1,
            )
            ax.plot(
                np.array([slip_list[ind_obs][1][0, 0], slip_list[ind_obs][1][0, 0]]),
                np.array([slip_list[ind_obs][0][0, 0], slip_list[ind_obs][0][0, 0]]),
                np.array(
                    [-slip_list[ind_obs][2][0, 0], -slip_list[ind_obs][2][0, 0] + 40]
                ),
                color="black",
                linestyle="--",
                linewidth=0.5,
                alpha=0.4,
            )
            ax.plot(
                np.array([slip_list[ind_obs][1][0, -1], slip_list[ind_obs][1][0, -1]]),
                np.array([slip_list[ind_obs][0][0, -1], slip_list[ind_obs][0][0, -1]]),
                np.array(
                    [-slip_list[ind_obs][2][0, -1], -slip_list[ind_obs][2][0, -1] + 40]
                ),
                color="black",
                linestyle="--",
                linewidth=0.5,
                alpha=0.4,
            )
            ax.plot(
                np.array([slip_list[ind_obs][1][-1, 0], slip_list[ind_obs][1][-1, 0]]),
                np.array([slip_list[ind_obs][0][-1, 0], slip_list[ind_obs][0][-1, 0]]),
                np.array(
                    [-slip_list[ind_obs][2][-1, 0], -slip_list[ind_obs][2][-1, 0] + 40]
                ),
                color="black",
                linestyle="--",
                linewidth=0.5,
                alpha=0.4,
            )
            ax.plot(
                np.array(
                    [slip_list[ind_obs][1][-1, -1], slip_list[ind_obs][1][-1, -1]]
                ),
                np.array(
                    [slip_list[ind_obs][0][-1, -1], slip_list[ind_obs][0][-1, -1]]
                ),
                np.array(
                    [
                        -slip_list[ind_obs][2][-1, -1],
                        -slip_list[ind_obs][2][-1, -1] + 40,
                    ]
                ),
                color="black",
                linestyle="--",
                linewidth=0.5,
                alpha=0.4,
            )
            ax.text(
                slip_list[ind_obs][1][2, -2],
                slip_list[ind_obs][0][2, -2],
                -Z_new[2, -2],
                "S%d" % plane_inds[ind_obs],
                fontsize=8,
                ha="center",
                va="center",
                color="black",
                zorder=100,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )
    cax = fig.add_axes(
        (0.2, 0.36, 0.18, 0.018)
    )  # (left, bottom, width, height) in figure coords
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_clim(tick_range[0], tick_range[1])
    cbar = fig.colorbar(m, cax=cax, orientation="horizontal")
    cbar.set_label("Slip (m)")
    cbar.ax.xaxis.set_label_coords(0.5, 2)
    cbar.ax.xaxis.set_tick_params(pad=3)
    import matplotlib.ticker as mticker

    cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.set_box_aspect([1.0, 1.0, 0.1])  # type:ignore
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_zlabel("Depth (km)")  # type:ignore

    lat0 = ref[0]
    lon0 = ref[1]
    from dyncfs.geo import d2km

    def x2lon(x, pos):
        # x 为东向位移（km），换算为经度
        return f"{(lon0 + x / d2km):.0f}"

    def y2lat(y, pos):
        # y 为北向位移（km），换算为纬度
        return f"{(lat0 + y / d2km):.0f}"

    def z2dep(z, pos):
        if z == 0:
            return 0
        else:
            return f"{-z:.0f}"

    ax.xaxis.set_major_formatter(FuncFormatter(x2lon))
    ax.yaxis.set_major_formatter(FuncFormatter(y2lat))
    ax.zaxis.set_major_formatter(FuncFormatter(z2dep))
    ax.tick_params(direction="in", width=2, colors="k", grid_color="k")
    # 将刻度对齐到整数度位置：根据当前轴范围自动取整
    xlim_km = np.array(ax.get_xlim())
    ylim_km = np.array(ax.get_ylim())
    lon_range_deg = lon0 + xlim_km / d2km
    lat_range_deg = lat0 + ylim_km / d2km
    lon_ticks_deg = np.arange(
        np.floor(min(lon_range_deg)), np.ceil(max(lon_range_deg)) + 1, 1
    )
    lat_ticks_deg = np.arange(
        np.floor(min(lat_range_deg)), np.ceil(max(lat_range_deg)) + 1, 1
    )
    ax.set_xticks((lon_ticks_deg - lon0) * d2km)
    ax.set_yticks((lat_ticks_deg - lat0) * d2km)

    zticks = [t for t in ax.get_zticks() if t <= 0]
    if len(zticks) == 0:
        zmin, zmax = ax.get_zlim()
        # 基于当前 zlim 生成负向刻度，避免无刻度情况
        z_end = min(0.0, zmax) - 1e-9
        zticks = np.linspace(zmin, z_end, num=4)
    ax.set_xlim(xlim_km)
    ax.set_ylim(ylim_km)
    ax.set_zticks(zticks)
    ax.set_zlim([-30, 0])

    towns = {
        "Wenchuan": (103.595, 31.476),
        # 'Yingxiu': (103.423, 31.009),
        "Qingchuan": (105.2353, 32.5883),
        "Beichuan": (104.454, 31.826),
        "Dujiangyan": (103.6242, 30.9933),
    }
    for name, (lon, lat) in towns.items():
        x_km = (lon - lon0) * d2km
        y_km = (lat - lat0) * d2km
        # ax.scatter3D(x_km, y_km, 0.0,
        #            s=40, c='k', marker='o', depthshade=False, zorder=302)
        ax.text(
            x_km,
            y_km,
            0.0,
            "●",
            color="k",
            fontsize=10,
            ha="left",
            va="bottom",
            zorder=301,
            path_effects=[pe.withStroke(linewidth=1, foreground="black")],
        )
        ax.text(
            x_km + 25,
            y_km,
            0.0,
            name,
            color="k",
            fontsize=9,
            ha="left",
            va="bottom",
            zorder=300,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    stf = np.load("stf.npy")
    t = np.linspace(0, len(stf) / 2, len(stf), endpoint=False)
    ax_stf = fig.add_axes((0.7, 0.55, 0.2, 0.15))
    ax_stf.plot(t, stf, color="black")
    ax_stf.set_xlabel("Time (s)")
    ax_stf.set_ylabel("Moment Rate (Nm/s)")

    fig.subplots_adjust(left=0, right=0.95, bottom=0, top=0.95)
    plt.savefig("slip_3d.pdf", bbox_inches="tight")
    plt.show()
