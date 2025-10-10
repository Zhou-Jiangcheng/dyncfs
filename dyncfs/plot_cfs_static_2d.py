import os

import numpy as np
from scipy.ndimage import zoom
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "Arial",
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def plot_cfs_static_2d(
    path_output,
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
                path_output, "results", "static", "cfs_static_plane%d.csv" % ind_obs
            )
        ),
        index_col=False,
        header=None,
    ).to_numpy()
    if color_saturation is None:
        color_saturation = np.max(np.abs(sub_stress))
    tick_range = [-color_saturation / 1e6, color_saturation / 1e6]

    sub_stress: np.ndarray = sub_stress.reshape(obs_shape[0], obs_shape[1])
    sub_stress = zoom(sub_stress, [zoom_strike, zoom_dip])
    sub_length_strike_km = sub_length_strike_km / zoom_strike
    sub_length_dip_km = sub_length_dip_km / zoom_dip

    cmap = matplotlib.colormaps["seismic"]
    norm = Normalize(vmin=tick_range[0], vmax=tick_range[1])

    ratio = obs_shape[1] / obs_shape[0]
    length = obs_shape[0]
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
    cbar.set_label("Static Coulomb Failure Stress Change (MPa)")

    ax.set_xlabel("Along Strike (km)")
    ax.set_ylabel("Along Dip (km)")

    xt = np.arange(round(ax.get_xlim()[1]))
    xl = [f"{float(i) * sub_length_dip_km:.1f}" for i in xt]
    ax.set_xticks(xt[::tick_interval] - 0.5)
    ax.set_xticklabels(xl[::tick_interval])

    yt = np.arange(round(ax.get_ylim()[0]))
    yl = [f"{float(i) * sub_length_strike_km:.1f}" for i in yt]
    ax.set_yticks(yt[::tick_interval] - 0.5)
    ax.set_yticklabels(yl[::tick_interval])

    # ax.text(xlim[0] + 1, ylim[1] + 1, "Static", ha="left", va="top", weight="bold")
    title = "Static Coulomb Failure Stress Change on No.%d Plane" % ind_obs
    fig.suptitle(title)
    fig.subplots_adjust(left=0.1, right=0.8, bottom=0, top=1)
    if save:
        plt.savefig(
            os.path.join(
                path_output,
                "results",
                "static",
                "cfs_static_plane_%d.png" % ind_obs,
            ),
            dpi=600,
        )
    if show:
        plt.ion()
        plt.show()
    else:
        plt.close(fig)


def plot_cfs_static_fix_depth(
    path_output,
    sub_stress,
    obs_depth,
    obs_lat_range: list = None,
    obs_lon_range: list = None,
    obs_delta_lat: float = None,
    obs_delta_lon: float = None,
    color_saturation=None,
    zoom_lat=1,
    zoom_lon=1,
    show=True,
    save=True,
):
    if not show:
        matplotlib.use("Agg")
    else:
        matplotlib.use("tkagg")
    Nx = int(np.ceil((obs_lat_range[1] - obs_lat_range[0]) / obs_delta_lat) + 1)
    Ny = int(np.ceil((obs_lon_range[1] - obs_lon_range[0]) / obs_delta_lon) + 1)
    if color_saturation is None:
        color_saturation = np.max(np.abs(sub_stress))
    tick_range = [-color_saturation / 1e6, color_saturation / 1e6]

    sub_stress: np.ndarray = sub_stress.reshape(Nx, Ny)
    sub_stress = zoom(sub_stress, [zoom_lat, zoom_lon])
    obs_delta_lat = obs_delta_lat / zoom_lat
    obs_delta_lon = obs_delta_lon / zoom_lon

    cmap = matplotlib.colormaps["seismic"]
    norm = Normalize(vmin=tick_range[0], vmax=tick_range[1])

    ratio = Nx / Ny
    length = 15 / 2.54
    height = length * ratio

    fig, ax = plt.subplots(figsize=(length, height))
    X, Y = np.meshgrid(
        np.arange(sub_stress.shape[0]),
        np.arange(sub_stress.shape[1]),
    )
    C = sub_stress / 1e6
    # print(sub_stress.shape)
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
    cbar.set_label("Static Coulomb Failure Stress Change (MPa)")

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")

    delta_tick = 0.5  # deg
    lon_start = np.ceil(obs_lon_range[0] * 1 / delta_tick) / round(1 / delta_tick)
    lon_end = obs_lon_range[1]
    lont_cuticks = np.arange(lon_start, lon_end + 1e-6, delta_tick)

    lat_start = np.ceil(obs_lat_range[0] * 1 / delta_tick) / round(1 / delta_tick)
    lat_end = obs_lat_range[1]
    lat_ticks = np.arange(lat_start, lat_end + 1e-6, delta_tick)

    xtick_pos = (lont_cuticks - obs_lon_range[0]) / obs_delta_lon
    ytick_pos = (lat_ticks - obs_lat_range[0]) / obs_delta_lat

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([f"{lt:.1f}" for lt in lont_cuticks])
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([f"{la:.1f}" for la in lat_ticks[::-1]])

    # ax.text(xlim[0] + 1, ylim[1] + 1, "Static", ha="left", va="top", weight="bold")
    title = "Static Coulomb Failure Stress Change at Depth %.2f km" % obs_depth
    fig.suptitle(title)
    fig.subplots_adjust(left=0.1, right=0.8, bottom=0, top=1)
    if save:
        plt.savefig(
            os.path.join(
                path_output,
                "results",
                "static",
                "cfs_static_depth_%.2f.png" % obs_depth,
            ),
            dpi=600,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    pass
