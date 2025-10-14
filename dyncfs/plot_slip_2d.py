import os

import numpy as np
from scipy.ndimage import zoom
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "Arial",
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def plot_slip_2d(
    path_input: str,
    nt_cut: int,
    sampling_interval_stf: float,
    ind_source: int,
    source_shape: list,
    sub_length_strike_km: float,
    sub_length_dip_km: float,
    slip_thresh: float = 0,
    color_saturation: float = None,
    tick_interval: int = 5,
    zoom_strike: int = 1,
    zoom_dip: int = 1,
    show: bool = True,
    save: bool = True,
):
    """
    Slip distribution plot, cumulative slip before nt_cut.
    """
    if not show:
        matplotlib.use("Agg")
    source_array = pd.read_csv(
        str(os.path.join(path_input, "source_plane%d.csv" % ind_source)),
        index_col=False,
        header=None,
    ).to_numpy()
    if slip_thresh > 0:
        inds_ignore_slip = np.where(source_array[:, 8] < slip_thresh)
        source_array[inds_ignore_slip, 8] = 0
        source_array[inds_ignore_slip, 9] = 0
    if nt_cut > 0:
        sub_stfs = source_array[:, 10:].copy()
        source_array[:, 10 + nt_cut :] = 0
        m0_stf_origin = np.sum(sub_stfs, axis=1)
        m0_stf_cut = np.sum(source_array[:, 10:], axis=1)
        cut_ratio = np.zeros_like(m0_stf_cut)
        inds_not_0 = np.argwhere(m0_stf_origin != 0).flatten()
        cut_ratio[inds_not_0] = m0_stf_cut[inds_not_0] / m0_stf_origin[inds_not_0]
        sub_slip = cut_ratio * source_array[:, 8]
    else:
        sub_slip = source_array[:, 8]
    if color_saturation is None:
        color_saturation = np.max(sub_slip)
    tick_range = [0, color_saturation]

    sub_slip: np.ndarray = sub_slip.reshape(source_shape[0], source_shape[1])
    sub_slip = zoom(sub_slip, [zoom_strike, zoom_dip])
    sub_length_strike_km = sub_length_strike_km / zoom_strike
    sub_length_dip_km = sub_length_dip_km / zoom_dip

    # cmap = matplotlib.colormaps["seismic"]
    colors_slip = ["blue", "cyan", "orange", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors_slip)
    norm = Normalize(vmin=tick_range[0], vmax=tick_range[1])

    ratio = source_shape[1] / source_shape[0]
    length = source_shape[0]
    height = length * ratio

    plt.ioff()
    fig, ax = plt.subplots(figsize=(length, height))
    X, Y = np.meshgrid(
        np.arange(sub_slip.shape[0]),
        np.arange(sub_slip.shape[1]),
    )
    C = sub_slip

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
    cbar.set_label("Slip (m)")

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

    title = "Slip Distribution before Time: %.2f s on No.%d Plane" % (
        float(nt_cut * sampling_interval_stf),
        ind_source,
    )
    fig.suptitle(title)
    fig.subplots_adjust(left=0.1, right=0.8, bottom=0, top=1)
    if save:
        plt.savefig(
            os.path.join(
                path_input,
                "slip_2d_nt_cut_%d_plane_%d.png" % (nt_cut, ind_source),
            ),
            dpi=600,
        )
    if show:
        plt.ion()
        plt.show()
    else:
        plt.close(fig)


def plot_slip_2d_series(
    path_input: str,
    nt_cut_list: list,
    sampling_interval_stf: float,
    ind_source: int,
    source_shape: list,
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
    Slip distribution plot, cumulative slip before nt_cut.
    """
    fp = os.path.join(path_input, f"source_plane{ind_source}.csv")
    arr = pd.read_csv(fp, header=None, index_col=False).to_numpy()
    vmin = 0
    if color_saturation is None:
        vmax = np.max(arr[:, 8])
    else:
        vmax = color_saturation
    zoom_factors = [zoom_strike, zoom_dip]

    time_sec_0 = nt_cut_list[0] * sampling_interval_stf
    time_sec_1 = nt_cut_list[-1] * sampling_interval_stf
    title = (
        f"Slip Distribution during Time: {time_sec_0:.2f}-{time_sec_1:.2f} s "
        f"on No.{ind_source} Plane"
    )
    save_path = os.path.join(
        path_input, f"slip_2d_{nt_cut_list[0]}_{nt_cut_list[-1]}_plane_{ind_source}.png"
    )
    nrows = int(np.floor(np.sqrt(len(nt_cut_list))))
    ncols = len(nt_cut_list) // nrows
    scale = 15 / ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * scale, nrows * source_shape[1] / source_shape[0] * scale),
    )
    for i_row in range(nrows):
        for i_col in range(ncols):
            ax = axes[i_row, i_col]
            ind = i_row * ncols + i_col
            nt_cut = nt_cut_list[ind]
            # Compute cut_ratio and then multiply by slip
            orig = arr[:, 10:].copy()
            arr[:, 10 + nt_cut :] = 0
            m0_o = np.sum(orig, axis=1)
            m0_c = np.sum(arr[:, 10:], axis=1)
            ratio = np.zeros_like(m0_c)
            nz = m0_o != 0
            ratio[nz] = m0_c[nz] / m0_o[nz]
            slip = ratio * arr[:, 8]
            data = slip.reshape(source_shape)  # unit: m
            data: np.ndarray = zoom(data, zoom_factors)
            # print(nt_cut, np.max(data))

            X, Y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap("seismic")
            ax.pcolormesh(X.T, Y.T, data, cmap=cmap, norm=norm, shading="auto")
            ax.invert_yaxis()
            ax.set_aspect(1)
            if i_col == 0:
                ax.set_ylabel("Along Dip (km)")
                yt = np.arange(round(ax.get_ylim()[0]))
                yl = [f"{float(i) * sub_length_strike_km:.1f}" for i in yt]
                ax.set_yticks(yt[::tick_interval] - 0.5)
                ax.set_yticklabels(yl[::tick_interval])
            else:
                ax.set_yticks([])
            if i_row == nrows - 1:
                ax.set_xlabel("Along Strike (km)")
                xt = np.arange(round(ax.get_xlim()[1]))
                xl = [f"{float(i) * sub_length_dip_km:.1f}" for i in xt]
                ax.set_xticks(xt[::tick_interval] - 0.5)
                ax.set_xticklabels(xl[::tick_interval])
            else:
                ax.set_xticks([])
            ax.text(
                ax.get_xlim()[0],
                ax.get_ylim()[1],
                f"{float(nt_cut * sampling_interval_stf):.2f} s",
                ha="left",
                va="top",
                weight="bold",
            )
    cax = fig.add_axes((0.925, 0.2, 0.025, 0.6))

    colors_slip = ["blue", "cyan", "orange", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors_slip)
    m = cm.ScalarMappable(cmap=cmap)
    m.set_clim(vmin, vmax)

    cbar = fig.colorbar(m, cax=cax)
    cbar.set_label("Slip (m)")
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
