import os

import numpy as np
from scipy.ndimage import zoom
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from obspy.imaging.beachball import beach

from dyncfs.configuration import CfsConfig
from dyncfs.geo import d2km

plt.rcParams.update(
    {
        "font.size": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def plot_ax(ax: plt.Axes, sub_stress: np.ndarray, focal_mechanism=None):
    X, Y = np.meshgrid(
        np.arange(sub_stress.shape[0]),
        np.arange(sub_stress.shape[1]),
    )
    C = sub_stress / 1e6
    # print(sub_stress.shape)
    # exchange x,y from lat,lon to lon,lat
    ax.pcolormesh(
        Y.T,
        X.T[::-1],
        C[::-1],
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    ax.set_aspect(1)

    delta_tick = 2  # deg

    if (obs_lon_range[1] - obs_lon_range[0]) <= 0 or (
        obs_lat_range[1] - obs_lat_range[0]
    ) <= 0:
        print("Warning: Invalid coordinate ranges")
        return ax

    lon_start = np.ceil(obs_lon_range[0] / delta_tick) * delta_tick
    lon_end = obs_lon_range[1]
    lon_ticks = np.arange(lon_start, lon_end + delta_tick / 2, delta_tick)

    lat_start = np.ceil(obs_lat_range[0] / delta_tick) * delta_tick
    lat_end = obs_lat_range[1]
    lat_ticks = np.arange(lat_start, lat_end + delta_tick / 2, delta_tick)

    xtick_pos = (lon_ticks - obs_lon_range[0]) / obs_delta_lon
    ytick_pos = (lat_ticks - obs_lat_range[0]) / obs_delta_lat

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([f"{lt:.1f}" for lt in lon_ticks])
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([f"{la:.1f}" for la in lat_ticks])

    for ind_src in range(len(config.source_inds[:-1])):
        source_plane = pd.read_csv(
            str(
                os.path.join(
                    config.path_input, f"source_plane{config.source_inds[ind_src]}.csv"
                )
            ),
            index_col=False,
            header=None,
        ).to_numpy()

        strike_rad = np.deg2rad(source_plane[:, 3])
        dip_rad = np.deg2rad(source_plane[:, 4])
        Ls = source_plane[:, 6]  # length along strike (km)
        Ld = source_plane[:, 7]  # length along dip (km)

        s_x = np.cos(strike_rad)  # North
        s_y = np.sin(strike_rad)  # East
        d_x = -np.sin(strike_rad) * np.cos(dip_rad)  # North
        d_y = np.cos(strike_rad) * np.cos(dip_rad)  # East
        d_z = np.sin(dip_rad)  # Down (>0)

        shift_x = -0.5 * (s_x * Ls + d_x * Ld)
        shift_y = -0.5 * (s_y * Ls + d_y * Ld)
        shift_z = -0.5 * (d_z * Ld)

        p0_lat = source_plane[:, 0] + shift_x / d2km
        p0_lon = source_plane[:, 1] + shift_y / d2km
        p0_dep = source_plane[:, 2] + shift_z

        with np.errstate(divide="ignore", invalid="ignore"):
            vstar = (obs_depth - p0_dep) / d_z
        mask = np.isfinite(vstar) & (vstar >= 0.0) & (vstar <= Ld) & (d_z > 0)
        if not np.any(mask):
            continue

        lat1 = p0_lat[mask] + (vstar[mask] * d_x[mask]) / d2km
        lon1 = p0_lon[mask] + (vstar[mask] * d_y[mask]) / d2km
        lat2 = p0_lat[mask] + (Ls[mask] * s_x[mask] + vstar[mask] * d_x[mask]) / d2km
        lon2 = p0_lon[mask] + (Ls[mask] * s_y[mask] + vstar[mask] * d_y[mask]) / d2km

        i1 = (lat1 - obs_lat_range[0]) / obs_delta_lat
        j1 = (lon1 - obs_lon_range[0]) / obs_delta_lon
        i2 = (lat2 - obs_lat_range[0]) / obs_delta_lat
        j2 = (lon2 - obs_lon_range[0]) / obs_delta_lon

        x1, y1 = j1, i1
        x2, y2 = j2, i2

        dep_step = np.median(d_z[mask] * Ld[mask])  # 每一行的深度增量（km）
        p0_min = np.min(p0_dep[mask])
        row_id = np.round((p0_dep[mask] - p0_min) / max(dep_step, 1e-6)).astype(int)

        s_plot_x = (s_y[mask] / d2km) / obs_delta_lon
        s_plot_y = (s_x[mask] / d2km) / obs_delta_lat
        s_norm = np.hypot(s_plot_x, s_plot_y)
        s_plot_x = s_plot_x / np.where(s_norm == 0, 1, s_norm)
        s_plot_y = s_plot_y / np.where(s_norm == 0, 1, s_norm)

        groups = {}
        for k, xx1, yy1, xx2, yy2, spx, spy in zip(
            row_id, x1, y1, x2, y2, s_plot_x, s_plot_y
        ):
            groups.setdefault(k, []).append((xx1, yy1, xx2, yy2, spx, spy))

        for k, segs in groups.items():
            oriented = []
            for xx1, yy1, xx2, yy2, spx, spy in segs:
                if (xx2 - xx1) * spx + (yy2 - yy1) * spy < 0:
                    xx1, yy1, xx2, yy2 = xx2, yy2, xx1, yy1
                mx, my = 0.5 * (xx1 + xx2), 0.5 * (yy1 + yy2)
                t = mx * spx + my * spy
                oriented.append((t, xx1, yy1, xx2, yy2))

            oriented.sort(key=lambda z: z[0])
            xs, ys = [], []
            for _, xx1, yy1, xx2, yy2 in oriented:
                if not xs:
                    xs.extend([xx1, xx2])
                    ys.extend([yy1, yy2])
                else:
                    xs.append(xx2)
                    ys.append(yy2)

            ax.plot(xs, ys, "-", linewidth=4, color="white", alpha=1)
            ax.plot(xs, ys, "-", linewidth=2.2, color="black", alpha=1)

    if focal_mechanism is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.2  # 左上角位置
        y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.2
        width = (xlim[1] - xlim[0]) * 0.15  # 相对于轴大小的宽度

        bb = beach(
            focal_mechanism,
            xy=(x_pos, y_pos),
            width=width,
            linewidth=1,
            facecolor="black",
            edgecolor="black",
        )
        ax.add_collection(bb)

    return ax


if __name__ == "__main__":
    config = CfsConfig()
    config.read_config("wenchuan.ini")

    obs_depth = 15
    sub_stress_half = pd.read_csv(
        str(
            os.path.join(
                config.path_output,
                "results",
                "static_half",
                "cfs_static_dep_%.2f.csv" % obs_depth,
            )
        ),
        index_col=False,
        header=None,
    ).to_numpy()
    sub_stress_layer = pd.read_csv(
        str(
            os.path.join(
                config.path_output,
                "results",
                "static_layer",
                "cfs_static_dep_%.2f.csv" % obs_depth,
            )
        ),
        index_col=False,
        header=None,
    ).to_numpy()

    path_cou_file = os.path.join(config.path_output, "dcff.cou")
    sub_stress_c3 = (
        pd.read_csv(path_cou_file, sep="\\s+", skiprows=3, header=None).to_numpy()[:, 3]
        * 1e5
    )

    color_saturation = 0.25e6
    tick_interval = 0.05  # MPa
    zoom_lat = 1
    zoom_lon = 1
    focal_mechanism = [223, 47, 131]  # strike, dip, rake

    obs_lat_range = config.obs_lat_range
    obs_lon_range = config.obs_lon_range
    obs_delta_lat = config.obs_delta_lat
    obs_delta_lon = config.obs_delta_lon

    Nx = int(np.ceil((obs_lat_range[1] - obs_lat_range[0]) / obs_delta_lat) + 1)
    Ny = int(np.ceil((obs_lon_range[1] - obs_lon_range[0]) / obs_delta_lon) + 1)
    if color_saturation is None:
        color_saturation = np.max(np.abs(sub_stress_half))
    tick_range = [-color_saturation / 1e6, color_saturation / 1e6]

    sub_stress_half: np.ndarray = sub_stress_half.reshape(Nx, Ny)
    sub_stress_layer: np.ndarray = sub_stress_layer.reshape(Nx, Ny)
    sub_stress_c3 = sub_stress_c3.reshape(Nx, Ny).T
    sub_stress_half = zoom(
        sub_stress_half, [zoom_lat, zoom_lon], order=1, mode="nearest", prefilter=False
    )
    sub_stress_layer = zoom(
        sub_stress_layer, [zoom_lat, zoom_lon], order=1, mode="nearest", prefilter=False
    )
    sub_stress_c3 = zoom(
        sub_stress_c3, [zoom_lat, zoom_lon], order=1, mode="nearest", prefilter=False
    )
    obs_delta_lat = obs_delta_lat / zoom_lat
    obs_delta_lon = obs_delta_lon / zoom_lon

    colors = ["blue", "cyan", "white", "yellow", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # boundaries = np.linspace(tick_range[0], tick_range[1],
    #                          round((tick_range[1] - tick_range[0]) / tick_interval) + 1)
    # norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    norm = Normalize(tick_range[0], tick_range[1])

    length = 20 / 2.54
    height = 22 / 2.54

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(length, height))
    ax0 = plot_ax(ax=axs[0, 0], sub_stress=sub_stress_half)
    ax1 = plot_ax(ax=axs[0, 1], sub_stress=sub_stress_c3)
    ax2 = plot_ax(ax=axs[1, 0], sub_stress=sub_stress_layer)
    ax3 = plot_ax(
        ax=axs[1, 1],
        sub_stress=sub_stress_layer - sub_stress_half,
        focal_mechanism=focal_mechanism,
    )
    ax0.set_xticks([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax3.set_yticks([])
    ax0.set_ylabel("Latitude (deg)")
    ax2.set_xlabel("Longitude (deg)")
    ax2.set_ylabel("Latitude (deg)")
    ax3.set_xlabel("Longitude (deg)")

    ax0.text(
        0.05,
        0.95,
        "(a)",
        transform=ax0.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax0.text(
        0.8,
        0.95,
        "Half-Space",
        transform=ax0.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        ha="center",
    )
    ax1.text(
        0.05,
        0.95,
        "(b)",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax1.text(
        0.8,
        0.95,
        "Coulomb3",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        ha="center",
    )
    ax2.text(
        0.05,
        0.95,
        "(c)",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax2.text(
        0.8,
        0.95,
        "Layered",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        ha="center",
    )
    ax3.text(
        0.05,
        0.95,
        "(d)",
        transform=ax3.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax3.text(
        0.8,
        0.95,
        "Difference",
        transform=ax3.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        ha="center",
    )
    cax = fig.add_axes([0.15, 0.07, 0.8, 0.025])  # [left, bottom, width, height]
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_clim(tick_range[0], tick_range[1])
    cbar = fig.colorbar(m, cax=cax, shrink=0.8, orientation="horizontal", extend="both")
    cbar.set_label("Static Coulomb Failure Stress Change (MPa)")

    ticks = np.linspace(
        tick_range[0],
        tick_range[1],
        round((tick_range[1] - tick_range[0]) / tick_interval) + 1,
    )
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    # ax.text(xlim[0] + 1, ylim[1] + 1, "Static", ha="left", va="top", weight="bold")
    # title = "Static Coulomb Failure Stress Change at Depth %.2f km" % obs_depth
    # fig.suptitle(title, x=0.54)
    fig.subplots_adjust(
        left=0.1, right=0.95, bottom=0.15, top=0.95, wspace=0.05, hspace=0
    )

    plt.savefig("compare_static_cfs_wenchuan.pdf", bbox_inches="tight")
    plt.show()
