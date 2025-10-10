import os

import numpy as np
from scipy.ndimage import zoom
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

from dyncfs.configuration import CfsConfig
from dyncfs.geo import d2km

plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "Arial",
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def draw_fault_intersections(
    ax: plt.Axes,
    plane_csv_files,
    obs_depth: float,
    obs_lon_range,
    obs_lat_range,
    obs_delta_lon: float,
    obs_delta_lat: float,
):
    for csv_path in plane_csv_files:
        if not os.path.exists(csv_path):
            continue
        source_plane = pd.read_csv(csv_path, index_col=False, header=None).to_numpy()

        strike_rad = np.deg2rad(source_plane[:, 3])
        dip_rad = np.deg2rad(source_plane[:, 4])
        Ls = source_plane[:, 6]  # along strike (km)
        Ld = source_plane[:, 7]  # along dip (km)

        s_x = np.cos(strike_rad)
        s_y = np.sin(strike_rad)
        d_x = -np.sin(strike_rad) * np.cos(dip_rad)
        d_y = np.cos(strike_rad) * np.cos(dip_rad)
        d_z = np.sin(dip_rad)

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

        dep_step = np.median(d_z[mask] * Ld[mask])
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

        for segs in groups.values():
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

            ax.plot(xs, ys, "-", linewidth=4, color="white", alpha=1, zorder=95)
            ax.plot(xs, ys, "-", linewidth=2.2, color="black", alpha=1, zorder=96)


def plot_ax(
    ax: plt.Axes,
    sub_stress: np.ndarray,
    obs_lon_range,
    obs_lat_range,
    obs_delta_lon,
    obs_delta_lat,
    cmap,
    norm,
):
    X, Y = np.meshgrid(
        np.arange(sub_stress.shape[0]),
        np.arange(sub_stress.shape[1]),
    )
    C = sub_stress / 1e6
    print(obs_lat_range, obs_lon_range)
    print(sub_stress.shape)
    # exchange x,y from lat,lon to lon,lat
    im = ax.pcolormesh(
        Y.T,
        X.T[::-1],
        C[::-1],
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    ax.scatter(
        (lon_af_1d[0] - obs_lon_range[0]) / obs_delta_lon,
        (lat_af_1d[0] - obs_lat_range[0]) / obs_delta_lat,
        color="black",
        edgecolors="white",
        marker="*",
        linewidths=1,
        s=200,
        zorder=101,
    )
    ax.scatter(
        (lon_af_1d - obs_lon_range[0]) / obs_delta_lon,
        (lat_af_1d - obs_lat_range[0]) / obs_delta_lat,
        color="white",
        edgecolors="black",
        alpha=1,
        linewidths=0.2,
        marker="o",
        s=5,
        zorder=100,
    )
    ax.scatter(
        (lon_af_rest - obs_lon_range[0]) / obs_delta_lon,
        (lat_af_rest - obs_lat_range[0]) / obs_delta_lat,
        color="white",
        edgecolors="black",
        alpha=1,
        linewidths=0.2,
        marker="o",
        s=5,
        zorder=99,
    )

    ax.set_aspect(1)
    ax.set_xlabel("Longitude (deg)")

    delta_tick = 0.1  # deg

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

    ax.set_xticks(xtick_pos[1:])
    ax.set_xticklabels([f"{lt:.1f}" for lt in lon_ticks[1:]])
    ax.set_yticks(ytick_pos[1:])
    ax.set_yticklabels([f"{la:.1f}" for la in lat_ticks[1:]])
    ax.set_xlim([0, (obs_lon_range[1] - obs_lon_range[0]) / obs_delta_lon - 1])
    ax.set_ylim([0, (obs_lat_range[1] - obs_lat_range[0]) / obs_delta_lat - 1])
    return ax, im


if __name__ == "__main__":
    config = CfsConfig()
    config.read_config("ludian.ini")

    plot_lat_range = (26.9, 27.3)
    plot_lon_range = (103.2, 103.6)

    data_lat_range = config.obs_lat_range
    data_lon_range = config.obs_lon_range

    sub_stress = pd.read_csv(
        str(
            os.path.join(
                config.path_output,
                "results",
                "static",
                "cfs_oop_static_dep_%.2f.csv" % config.fixed_obs_depth,
            )
        ),
        header=None,
        index_col=False,
    ).to_numpy()
    sub_stress_d = pd.read_csv(
        str(
            os.path.join(
                config.path_output,
                "results",
                "dynamic",
                "cfs_oop_dynamic_dep_%.2f.csv" % config.fixed_obs_depth,
            )
        ),
        header=None,
        index_col=False,
    ).to_numpy()
    sub_stress_d_min = np.minimum(sub_stress_d, np.zeros_like(sub_stress_d))
    sub_stress_d_min = np.min(sub_stress_d_min, axis=1)
    sub_stress_d_max = np.maximum(sub_stress_d, np.zeros_like(sub_stress_d))
    sub_stress_d_max = np.max(sub_stress_d_max, axis=1)
    sub_stress_d = sub_stress_d_max - sub_stress_d_min
    # sub_stress_d = sub_stress_d_max

    af_shocks = pd.read_csv(
        "./after_ludian.txt", sep="\\s+", header=None, index_col=False
    ).to_numpy()
    lat_af_1d = af_shocks[:133, 1]
    lon_af_1d = af_shocks[:133, 0]
    mag_1d = af_shocks[:133, 3]
    lat_af_rest = af_shocks[133:, 1]
    lon_af_rest = af_shocks[133:, 0]
    mag_rest = af_shocks[133:, 3]

    tick_interval_left = 0.05  # MPa
    tick_range_left = [-0.25, 0.25]

    tick_interval_right = 0.1
    tick_range_right = [0, 1]

    zoom_lat_s = 5
    zoom_lon_s = 5
    zoom_lat_d = 5
    zoom_lon_d = 5

    obs_lat_range = data_lat_range
    obs_lon_range = data_lon_range
    obs_delta_lat_s = 0.01
    obs_delta_lon_s = 0.01
    obs_delta_lat_d = 0.01
    obs_delta_lon_d = 0.01

    Nx_s = int(np.ceil((data_lat_range[1] - data_lat_range[0]) / obs_delta_lat_s) + 1)
    Ny_s = int(np.ceil((data_lon_range[1] - data_lon_range[0]) / obs_delta_lon_s) + 1)
    Nx_d = int(np.ceil((data_lat_range[1] - data_lat_range[0]) / obs_delta_lat_d) + 1)
    Ny_d = int(np.ceil((data_lon_range[1] - data_lon_range[0]) / obs_delta_lon_d) + 1)

    sub_stress: np.ndarray = sub_stress.reshape(Nx_s, Ny_s)
    sub_stress = zoom(
        sub_stress, [zoom_lat_s, zoom_lon_s], order=1, mode="nearest", prefilter=False
    )
    obs_delta_lat_s_out = obs_delta_lat_s / zoom_lat_s
    obs_delta_lon_s_out = obs_delta_lon_s / zoom_lon_s

    sub_stress_d = sub_stress_d.reshape(Nx_d, Ny_d)
    sub_stress_d = zoom(
        sub_stress_d, [zoom_lat_d, zoom_lon_d], order=1, mode="nearest", prefilter=False
    )
    obs_delta_lat_d_out = obs_delta_lat_d / zoom_lat_d
    obs_delta_lon_d_out = obs_delta_lon_d / zoom_lon_d

    def _slice_idx(plot_range, full_range, delta, nmax):
        i0 = int(round((plot_range[0] - full_range[0]) / delta))
        i1 = int(round((plot_range[1] - full_range[0]) / delta))
        i0 = max(i0, 0)
        i1 = min(i1, nmax)
        return slice(i0, i1)

    # 静态
    s_lat_idx = _slice_idx(
        plot_lat_range, data_lat_range, obs_delta_lat_s_out, sub_stress.shape[0]
    )
    s_lon_idx = _slice_idx(
        plot_lon_range, data_lon_range, obs_delta_lon_s_out, sub_stress.shape[1]
    )
    sub_stress = sub_stress[s_lat_idx, s_lon_idx]

    d_lat_idx = _slice_idx(
        plot_lat_range, data_lat_range, obs_delta_lat_d_out, sub_stress_d.shape[0]
    )
    d_lon_idx = _slice_idx(
        plot_lon_range, data_lon_range, obs_delta_lon_d_out, sub_stress_d.shape[1]
    )
    sub_stress_d = sub_stress_d[d_lat_idx, d_lon_idx]

    obs_lat_range = plot_lat_range
    obs_lon_range = plot_lon_range

    # colors_left = ['blue', 'cyan', 'white', 'yellow', 'red']
    # cmap_left = LinearSegmentedColormap.from_list('custom_cmap_left', colors_left)
    # norm_left = Normalize(vmin=tick_range_left[0], vmax=tick_range_left[1])

    colors_left = ["blue", "cyan", "white", "yellow", "red"]
    cmap_left = LinearSegmentedColormap.from_list("custom_cmap", colors_left)
    boundaries = np.linspace(
        tick_range_left[0],
        tick_range_left[1],
        round((tick_range_left[1] - tick_range_left[0]) / tick_interval_left) + 1,
    )
    norm_left = BoundaryNorm(boundaries, cmap_left.N, clip=True)

    colors_right = ["yellow", "orange", "red"]
    # cmap_right = LinearSegmentedColormap.from_list(
    #     'custom_cmap_right', colors_right)
    # norm_right = Normalize(vmin=0.0, vmax=tick_range_right[1])  # MPa

    cmap_right = LinearSegmentedColormap.from_list("custom_cmap", colors_right)
    boundaries_right = np.linspace(
        tick_range_right[0],
        tick_range_right[1],
        round((tick_range_right[1] - tick_range_right[0]) / tick_interval_right) + 1,
    )
    norm_right = BoundaryNorm(boundaries_right, cmap_right.N, clip=True)

    length = 2 * 15 / 2.54
    height = 17 / 2.54

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(length, height))
    ax0, im0 = plot_ax(
        ax=axs[0],
        sub_stress=sub_stress,
        obs_lon_range=obs_lon_range,
        obs_lat_range=obs_lat_range,
        obs_delta_lon=obs_delta_lon_s_out,
        obs_delta_lat=obs_delta_lat_s_out,
        cmap=cmap_left,
        norm=norm_left,
    )
    ax1, im1 = plot_ax(
        ax=axs[1],
        sub_stress=sub_stress_d,
        obs_lon_range=obs_lon_range,
        obs_lat_range=obs_lat_range,
        obs_delta_lon=obs_delta_lon_d_out,
        obs_delta_lat=obs_delta_lat_d_out,
        cmap=cmap_right,
        norm=norm_right,
    )

    Zr = sub_stress_d / 1e6  # MPa
    Xr, Yr = np.meshgrid(np.arange(Zr.shape[0]), np.arange(Zr.shape[1]))
    levels_r = np.arange(
        tick_range_right[0], tick_range_right[1] + 1e-9, tick_interval_right
    )
    ax1.contour(
        Yr.T,
        Xr.T[::-1],
        Zr[::-1],
        levels=levels_r,
        colors="k",
        linewidths=0.5,
        alpha=1,
        zorder=90,
    )

    ax0.set_ylabel("Latitude (deg)")

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

    plane_csvs = [
        os.path.join(config.path_input, "source_plane1.csv"),
        os.path.join(config.path_input, "source_plane2.csv"),
    ]
    draw_fault_intersections(
        ax0,
        plane_csvs,
        config.fixed_obs_depth,
        obs_lon_range,
        obs_lat_range,
        obs_delta_lon_s_out,
        obs_delta_lat_s_out,
    )
    draw_fault_intersections(
        ax1,
        plane_csvs,
        config.fixed_obs_depth,
        obs_lon_range,
        obs_lat_range,
        obs_delta_lon_d_out,
        obs_delta_lat_d_out,
    )

    cbar0 = fig.colorbar(
        im0, ax=ax0, orientation="horizontal", fraction=0.046, pad=0.12, extend="both"
    )
    cbar0.set_label("Static Coulomb Failure Stress Change (MPa)")
    ticks_left = np.linspace(-0.2, 0.2, 5)
    cbar0.set_ticks(ticks_left)
    cbar0.set_ticklabels([f"{t:.1f}" for t in ticks_left])

    cbar1 = fig.colorbar(
        im1, ax=ax1, orientation="horizontal", fraction=0.046, pad=0.12, extend="max"
    )
    cbar1.set_label("Peak Dynamic Coulomb Failure Stress Change (MPa)")
    ticks_right = np.linspace(
        0.0, tick_range_right[1], round(tick_range_right[1] / tick_interval_right + 1)
    )
    cbar1.set_ticks(ticks_right)
    cbar1.set_ticklabels([f"{t:.2f}" for t in ticks_right])

    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.15, top=0.98)

    plt.savefig(
        "compare_cfs_ludian_%.2f.pdf" % config.fixed_obs_depth, bbox_inches="tight"
    )
    plt.show()
