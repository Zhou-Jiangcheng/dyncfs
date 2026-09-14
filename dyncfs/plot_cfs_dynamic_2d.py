import os

import numpy as np
from scipy.ndimage import zoom
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from .utils import cal_grid_num, cal_geo_ticks

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "Arial",
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def plot_cfs_dynamic_2d_nt(
    path_output: str,
    nt: int,
    sampling_interval_cfs: float,
    ind_obs: int,
    obs_shape: list,
    sub_length_strike_km: float,
    sub_length_dip_km: float,
    color_saturation: float = None,
    tick_interval: int = 5,
    zoom_strike: int = 1,
    zoom_dip: int = 1,
    show: bool = True,
    save: bool = True,
):
    if not show:
        matplotlib.use("Agg")
    sub_stress = pd.read_csv(
        str(
            os.path.join(
                path_output, "results", "dynamic", "cfs_dynamic_plane%d.csv" % ind_obs
            )
        ),
        index_col=False,
        header=None,
    ).to_numpy()
    sub_stress = sub_stress[:, nt].flatten()

    sub_stress: np.ndarray = sub_stress.reshape(obs_shape[0], obs_shape[1])
    sub_stress = zoom(sub_stress, [zoom_strike, zoom_dip])
    sub_length_strike_km = sub_length_strike_km / zoom_strike
    sub_length_dip_km = sub_length_dip_km / zoom_dip

    if color_saturation is None:
        color_saturation = np.max(np.abs(sub_stress))
        # print(color_saturation/1e6)
    tick_range = [-color_saturation / 1e6, color_saturation / 1e6]
    cmap = matplotlib.colormaps["seismic"]
    norm = Normalize(vmin=tick_range[0], vmax=tick_range[1])

    ratio = obs_shape[1] / obs_shape[0]
    length = obs_shape[0] / 2.54
    height = length * ratio

    plt.ioff()
    fig, ax = plt.subplots(figsize=(length, height))
    X, Y = np.meshgrid(
        np.arange(sub_stress.shape[0]),
        np.arange(sub_stress.shape[1]),
    )
    C = sub_stress / 1e6

    ax.pcolormesh(
        X.T,
        Y.T,
        C,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    ax.invert_yaxis()
    ax.set_aspect(1)
    cax = fig.add_axes((0.85, 0.2, 0.025, 0.6))
    m = cm.ScalarMappable(cmap=cmap)
    m.set_clim(tick_range[0], tick_range[1])
    cbar = fig.colorbar(m, cax=cax)
    cbar.set_label("Dynamic Coulomb Failure Stress Change (MPa)")

    ax.set_xlabel("Along Strike (km)")
    ax.set_ylabel("Along Dip (km)")

    xt = np.arange(round(ax.get_xlim()[1]))
    xl = [f"{float(i) * sub_length_strike_km:.1f}" for i in xt]
    ax.set_xticks(xt[::tick_interval] - 0.5)
    ax.set_xticklabels(xl[::tick_interval])

    yt = np.arange(round(ax.get_ylim()[0]))
    yl = [f"{float(i) * sub_length_dip_km:.1f}" for i in yt]
    ax.set_yticks(yt[::tick_interval] - 0.5)
    ax.set_yticklabels(yl[::tick_interval])

    # ax.text(xlim[0] + 1, ylim[1] + 1, "Static", ha="left", va="top", weight="bold")
    title = "Dynamic Coulomb Failure Stress Change at Time: %.2f s on No.%d Plane" % (
        float(nt * sampling_interval_cfs),
        ind_obs,
    )
    fig.suptitle(title)
    fig.subplots_adjust(left=0.1, right=0.8, bottom=0, top=1)
    if save:
        plt.savefig(
            os.path.join(
                path_output,
                "results",
                "dynamic",
                "cfs_dynamic_nt_%d_plane_%d.png" % (nt, ind_obs),
            ),
            dpi=600,
        )
    if show:
        plt.ion()
        plt.show()
    else:
        plt.close(fig)


def plot_cfs_dynamic_2d_series(
    path_output: str,
    nt_list: list,
    sampling_interval_cfs: float,
    ind_obs: int,
    obs_shape: list,
    sub_length_strike_km: float,
    sub_length_dip_km: float,
    color_saturation: float = None,
    tick_interval: int = 5,
    zoom_strike: int = 1,
    zoom_dip: int = 1,
    show: bool = True,
    save: bool = True,
):
    """
    Dynamic CFS distributions on the obs plane at the time points in nt_list.
    color_saturation is in MPa.
    """
    if not show:
        matplotlib.use("Agg")
    sub_stress = (
        pd.read_csv(
            str(
                os.path.join(
                    path_output,
                    "results",
                    "dynamic",
                    "cfs_dynamic_plane%d.csv" % ind_obs,
                )
            ),
            index_col=False,
            header=None,
        ).to_numpy()
        / 1e6
    )
    nt_list = list(nt_list)
    if color_saturation is None:
        vmax = np.max(np.abs(sub_stress[:, nt_list]))
        vmin = -vmax
    else:
        vmin = -color_saturation
        vmax = color_saturation
    cmap = matplotlib.colormaps["seismic"]
    norm = Normalize(vmin=vmin, vmax=vmax)
    zoom_factors = [zoom_strike, zoom_dip]

    time_sec_0 = nt_list[0] * sampling_interval_cfs
    time_sec_1 = nt_list[-1] * sampling_interval_cfs
    title = (
        f"Dynamic CFS Distribution during Time: {time_sec_0:.2f}-{time_sec_1:.2f} s "
        f"on No.{ind_obs} Plane"
    )
    save_path = os.path.join(
        path_output, f"cfs_dynamic_2d_{nt_list[0]}_{nt_list[-1]}_plane_{ind_obs}.png"
    )
    n_panels = len(nt_list)
    nrows = max(1, int(np.floor(np.sqrt(n_panels))))
    ncols = int(np.ceil(n_panels / nrows))
    scale = 15 / ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * scale, nrows * obs_shape[1] / obs_shape[0] * scale),
        squeeze=False,
    )
    sub_length_strike_zoom = sub_length_strike_km / zoom_strike
    sub_length_dip_zoom = sub_length_dip_km / zoom_dip
    for i_row in range(nrows):
        for i_col in range(ncols):
            ax = axes[i_row, i_col]
            ind = i_row * ncols + i_col
            if ind >= n_panels:
                ax.set_axis_off()
                continue
            nt = nt_list[ind]
            sub_stress_nt = sub_stress[:, nt]
            data = sub_stress_nt.reshape(obs_shape)  # unit: MPa
            data: np.ndarray = zoom(data, zoom_factors)

            X, Y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
            ax.pcolormesh(X.T, Y.T, data, cmap=cmap, norm=norm, shading="auto")
            ax.invert_yaxis()
            ax.set_aspect(1)
            if i_col == 0:
                ax.set_ylabel("Along Dip (km)")
                yt = np.arange(round(ax.get_ylim()[0]))
                yl = [f"{float(i) * sub_length_dip_zoom:.1f}" for i in yt]
                ax.set_yticks(yt[::tick_interval] - 0.5)
                ax.set_yticklabels(yl[::tick_interval])
            else:
                ax.set_yticks([])
            # bottom panel of each column
            if ind + ncols >= n_panels:
                ax.set_xlabel("Along Strike (km)")
                xt = np.arange(round(ax.get_xlim()[1]))
                xl = [f"{float(i) * sub_length_strike_zoom:.1f}" for i in xt]
                ax.set_xticks(xt[::tick_interval] - 0.5)
                ax.set_xticklabels(xl[::tick_interval])
            else:
                ax.set_xticks([])
            ax.text(
                ax.get_xlim()[0],
                ax.get_ylim()[1],
                f"{float(nt * sampling_interval_cfs):.2f} s",
                ha="left",
                va="top",
                weight="bold",
            )
    cax = fig.add_axes((0.925, 0.2, 0.025, 0.6))
    m = cm.ScalarMappable(cmap=cmap)
    m.set_clim(vmin, vmax)
    cbar = fig.colorbar(m, cax=cax)
    cbar.set_label("Dynamic Coulomb Failure Stress Change (MPa)")
    fig.suptitle(title)
    fig.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.01 * ncols / nrows,
        hspace=0.01,
    )

    # save/show
    if save:
        fig.savefig(save_path, dpi=600)
    if show:
        plt.ion()
        plt.show()
    else:
        plt.close(fig)


def plot_cfs_dynamic_fix_depth_one_time_point(
    path_output: str,
    nt: int,
    sampling_interval_cfs: float,
    obs_depth: float,
    obs_lat_range: list = None,
    obs_lon_range: list = None,
    obs_delta_lat: float = None,
    obs_delta_lon: float = None,
    color_saturation: float = None,
    zoom_lat: int = 1,
    zoom_lon: int = 1,
    show: bool = True,
    save: bool = True,
    delta_tick: float = None,
):
    """
    :param delta_tick: Interval of longitude/latitude ticks (deg), chosen
                       automatically if None.
    """
    if not show:
        matplotlib.use("Agg")
    Nx = cal_grid_num(obs_lat_range, obs_delta_lat)
    Ny = cal_grid_num(obs_lon_range, obs_delta_lon)

    sub_stress = pd.read_csv(
        str(
            os.path.join(
                path_output,
                "results",
                "dynamic",
                "cfs_dynamic_dep_%.2f.csv" % obs_depth,
            )
        ),
        index_col=False,
        header=None,
    ).to_numpy()[:, nt]
    if color_saturation is None:
        color_saturation = np.max(np.abs(sub_stress))
        # print(color_saturation/1e6)
    tick_range = [-color_saturation / 1e6, color_saturation / 1e6]
    sub_stress: np.ndarray = sub_stress.reshape(Nx, Ny)
    sub_stress = zoom(sub_stress, [zoom_lat, zoom_lon])

    cmap = matplotlib.colormaps["seismic"]
    norm = Normalize(vmin=tick_range[0], vmax=tick_range[1])

    ratio = Nx / Ny
    length = 15 / 2.54
    height = length * ratio

    plt.ioff()
    fig, ax = plt.subplots(figsize=(length, height))
    X, Y = np.meshgrid(
        np.arange(sub_stress.shape[0]),
        np.arange(sub_stress.shape[1]),
    )
    C = sub_stress / 1e6
    # exchange x,y from lat,lon to lon,lat
    ax.pcolormesh(
        Y.T,
        X.T,
        C[::-1],
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    ax.invert_yaxis()
    ax.set_aspect(1)
    cax = fig.add_axes((0.85, 0.2, 0.025, 0.6))
    m = cm.ScalarMappable(cmap=cmap)
    m.set_clim(tick_range[0], tick_range[1])
    cbar = fig.colorbar(m, cax=cax)
    cbar.set_label("Dynamic Coulomb Failure Stress Change (MPa)")

    # ax.set_axis_off()
    # ax.grid(False)
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")

    # grid points span obs_lat_range/obs_lon_range linearly (also after zooming);
    # rows are flipped (C[::-1]) so that row 0 is the northernmost latitude
    xtick_pos, _, xtick_labels = cal_geo_ticks(
        obs_lon_range, sub_stress.shape[1], delta_tick
    )
    ytick_pos, _, ytick_labels = cal_geo_ticks(
        obs_lat_range, sub_stress.shape[0], delta_tick, reverse=True
    )
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels)

    # ax.text(xlim[0] + 1, ylim[1] + 1, "Static", ha="left", va="top", weight="bold")
    title = "Dynamic Coulomb Failure Stress Change at Depth: %.2f km, Time: %.2f s" % (
        obs_depth,
        float(nt * sampling_interval_cfs),
    )
    fig.suptitle(title)
    fig.subplots_adjust(left=0.1, right=0.8, bottom=0, top=1)
    if save:
        plt.savefig(
            os.path.join(
                path_output,
                "results",
                "dynamic",
                "cfs_dynamic_nt_%d_depth_%.2f.png" % (nt, obs_depth),
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
