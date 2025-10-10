import os

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap
import matplotlib.patheffects as path_effects

from dyncfs.cfs_dynamic import *


plt.rcParams.update(
    {
        "font.size": 8,
    }
)


def read_cfs_static_along_time(config: CfsConfig):
    cfs_static_time = []
    for nt in nt_list:
        path_csv = os.path.join(
            config.path_output_results_static,
            "time",
            "%d" % nt,
            "cfs_static_plane5.csv",
        )
        cfs_time = (
            pd.read_csv(path_csv, header=None, index_col=False).to_numpy().flatten()
        )
        cfs_static_time.append(cfs_time)
    cfs_static_time = np.array(cfs_static_time).T
    return cfs_static_time


if __name__ == "__main__":
    path_input = "input"
    path_output = "output/results/dynamic"

    num_dip = 6
    zoom_strike, zoom_dip = 1, 1

    nt_list = [_ for _ in range(32, 60, 4)]
    km_per_cell = 5 / zoom_strike
    x_step_cells = 2 * zoom_strike
    y_step_cells = 2 * zoom_strike
    cell_edge_color = "white"
    cell_edge_width = 0.01

    srate = 2
    source_array = pd.read_csv(
        str(os.path.join(path_input, "source_plane5.csv")), header=None, index_col=False
    ).to_numpy()
    sub_stf_m0 = np.array([np.sum(source_array[:, 10:] / srate, axis=1)]).T
    temp = np.divide(
        source_array[:, 10:],
        sub_stf_m0,
        out=np.zeros_like(source_array[:, 10:], dtype=float),
        where=sub_stf_m0 != 0,
    )
    sub_slip_rate_time = temp * source_array[:, 8][:, None]

    config = CfsConfig()
    config.read_config("wenchuan.ini")
    config.source_inds = [2]
    config.source_shapes = [[21, 10]]
    config.obs_inds = [1]
    config.obs_shapes = [[21, 10]]
    cfs_static_time = read_cfs_static_along_time(config=config) / 1e6

    cfs_dynamic = (
        pd.read_csv(
            str(os.path.join(path_output, "cfs_dynamic_plane5.csv")),
            header=None,
            index_col=False,
        ).to_numpy()
        / 1e6
    )

    w = 12 / 2.54
    h = 13 / 2.54
    print(w, h)

    tick_range_cfs = [-1, 1]
    tick_interval_cfs = 0.1
    colors_cfs = ["blue", "cyan", "white", "yellow", "red"]
    cmap_cfs = LinearSegmentedColormap.from_list("custom_cmap", colors_cfs)
    # norm_cfs = Normalize(tick_range_cfs[0], tick_range_cfs[1])
    boundaries_cfs = np.linspace(
        tick_range_cfs[0],
        tick_range_cfs[1],
        round((tick_range_cfs[1] - tick_range_cfs[0]) / tick_interval_cfs) + 1,
    )
    norm_cfs = BoundaryNorm(boundaries_cfs, cmap_cfs.N, clip=True)

    tick_range_cfs_static = [-1, 1]
    tick_interval_cfs_static = 0.1
    # colors_cfs_static = ['blue', 'cyan', 'orange', 'red']
    # cmap_cfs_static = LinearSegmentedColormap.from_list('custom_cmap', colors_cfs_static)
    # boundaries_cfs_static = np.linspace(tick_range_cfs_static[0], tick_range_cfs_static[1],
    #                                     round((tick_range_cfs_static[1] - tick_range_cfs_static[0]) /
    #                                           tick_interval_cfs_static) + 1)
    # norm_cfs_static = BoundaryNorm(boundaries_cfs_static, cmap_cfs_static.N, clip=True)
    cmap_cfs_static = cmap_cfs
    norm_cfs_static = norm_cfs

    tick_range_slip_rate = [0, 0.5]
    tick_interval_slip_rate = 0.1
    colors_slip_rate = ["white", "red"]
    cmap_slip_rate = LinearSegmentedColormap.from_list("custom_cmap", colors_slip_rate)
    # cmap_slip_rate= matplotlib.colormaps['seismic']
    # norm_slip_rate = Normalize(tick_range_slip_rate[0], tick_range_slip_rate[1])
    boundaries_slip_rate = np.linspace(
        tick_range_slip_rate[0],
        tick_range_slip_rate[1],
        round(
            (tick_range_slip_rate[1] - tick_range_slip_rate[0])
            / tick_interval_slip_rate
        )
        + 1,
    )
    norm_slip_rate = BoundaryNorm(boundaries_slip_rate, cmap_slip_rate.N, clip=True)

    fig, axs = plt.subplots(nrows=len(nt_list), ncols=3, figsize=(w, h))
    for i in range(len(nt_list)):
        nt = nt_list[i]
        ax_cfs = axs[i, 1]  # type:plt.axes
        ax_cfs_static = axs[i, 0]  # type:plt.axes
        ax_slip_rate = axs[i, 2]  # type:plt.axes

        sub_stress_dynamic = cfs_dynamic[:, nt].flatten().reshape(-1, num_dip)
        sub_stress_dynamic = zoom(sub_stress_dynamic, [zoom_strike, zoom_dip], order=1)

        sub_stress_static = cfs_static_time[:, i].flatten().reshape(-1, num_dip)
        sub_stress_static = zoom(sub_stress_static, [zoom_strike, zoom_dip], order=1)

        sub_slip_rate = sub_slip_rate_time[:, nt].flatten().reshape(-1, num_dip)
        sub_slip_rate = zoom(sub_slip_rate, [zoom_strike, zoom_dip], order=1)

        X, Y = np.meshgrid(
            np.arange(sub_stress_dynamic.shape[0] + 1),
            np.arange(sub_stress_dynamic.shape[1] + 1),
        )

        ny = sub_stress_dynamic.shape[1]
        yloc = np.arange(0, ny + 1, y_step_cells)
        ax_cfs_static.set_yticks(yloc)
        ax_cfs_static.set_yticklabels((yloc * km_per_cell).astype(int))
        print(
            "%.0f s" % (nt_list[i] / srate),
            np.max(sub_stress_dynamic),
            np.max(sub_slip_rate),
            np.max(sub_stress_static),
        )
        text = ax_cfs_static.text(
            0.025,
            0.9,
            s="%.0f s" % (nt_list[i] / srate),
            ha="left",
            va="top",
            weight="bold",
            color="black",
            transform=ax_cfs_static.transAxes,
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=1, foreground="white"),
                path_effects.Normal(),
            ]
        )

        m_plane_cfs = ax_cfs.pcolormesh(
            X.T,
            Y.T,
            sub_stress_dynamic,
            cmap=cmap_cfs,
            norm=norm_cfs,
            shading="auto",
            edgecolors=cell_edge_color,
            linewidth=cell_edge_width,
        )
        ax_cfs.invert_yaxis()
        ax_cfs.set_yticks([])

        m_plane_slip_rate = ax_slip_rate.pcolormesh(
            X.T,
            Y.T,
            sub_slip_rate,
            cmap=cmap_slip_rate,
            norm=norm_slip_rate,
            shading="auto",
            edgecolors=cell_edge_color,
            linewidth=cell_edge_width,
        )
        ax_slip_rate.set_yticks([])
        ax_slip_rate.invert_yaxis()

        m_plane_cfs_static = ax_cfs_static.pcolormesh(
            X.T,
            Y.T,
            sub_stress_static,
            cmap=cmap_cfs_static,
            norm=norm_cfs_static,
            shading="auto",
            edgecolors=cell_edge_color,
            linewidth=cell_edge_width,
        )
        # ax_cfs_static.set_yticks([])
        ax_cfs_static.invert_yaxis()

        if i % 3 == 1:
            ax_cfs_static.set_ylabel("Along Dip (km)")
            ax_cfs_static.yaxis.set_label_coords(-0.25, 0.25)

        if i != len(nt_list) - 1:
            ax_cfs.set_xticks([])
            ax_slip_rate.set_xticks([])
            ax_cfs_static.set_xticks([])
        else:
            ax_cfs.set_xlabel("Along Strike (km)")
            ax_cfs.xaxis.set_label_coords(0.5, -0.5)
            ax_slip_rate.set_xlabel("Along Strike (km)")
            ax_slip_rate.xaxis.set_label_coords(0.5, -0.5)
            ax_cfs_static.set_xlabel("Along Strike (km)")
            ax_cfs_static.xaxis.set_label_coords(0.5, -0.5)
            # 最后一行显示横向刻度（每2格=10 km）
            nx = sub_stress_dynamic.shape[0]
            xloc = np.arange(0, nx + 1, x_step_cells)
            xlabels = (xloc * km_per_cell).astype(int)
            for ax in (ax_cfs, ax_slip_rate, ax_cfs_static):
                ax.set_xticks(xloc)
                ax.set_xticklabels(xlabels)

        ax_cfs.set_aspect(1)
        ax_slip_rate.set_aspect(1)
        ax_cfs_static.set_aspect(1)

    sm_cfs = matplotlib.cm.ScalarMappable(norm=norm_cfs, cmap=cmap_cfs)
    sm_slip_rate = matplotlib.cm.ScalarMappable(
        norm=norm_slip_rate, cmap=cmap_slip_rate
    )
    sm_slip = matplotlib.cm.ScalarMappable(norm=norm_cfs_static, cmap=cmap_cfs_static)

    pos0 = axs[0, 0].get_position()
    pos1 = axs[0, 1].get_position()
    pos2 = axs[0, 2].get_position()

    y_cbar = 0.1
    h_cbar = 0.015
    cax0 = fig.add_axes([pos0.x0 + 0.0 * pos0.width, y_cbar, pos0.width, h_cbar])
    cax1 = fig.add_axes([pos1.x0 + 0.05 * pos1.width, y_cbar, pos1.width, h_cbar])
    cax2 = fig.add_axes([pos2.x0 + 0.1 * pos2.width, y_cbar, pos2.width, h_cbar])

    cbar_cfs = fig.colorbar(sm_cfs, cax=cax1, orientation="horizontal")
    cbar_cfs_static = fig.colorbar(sm_slip, cax=cax0, orientation="horizontal")
    cbar_slip_rate = fig.colorbar(sm_slip_rate, cax=cax2, orientation="horizontal")

    cbar_cfs.set_ticks(np.arange(tick_range_cfs[0], tick_range_cfs[1] + 1e-9, 0.5))
    cbar_cfs_static.set_ticks(
        np.arange(tick_range_cfs_static[0], tick_range_cfs_static[1] + 1e-9, 0.5)
    )
    cbar_slip_rate.set_ticks(
        np.arange(
            tick_range_slip_rate[0],
            tick_range_slip_rate[1] + 1e-9,
            2 * tick_interval_slip_rate,
        )
    )

    cbar_cfs_static.set_label(r"Static $\Delta$CFS (MPa)", labelpad=1)
    cbar_cfs.set_label(r"Dynamic $\Delta$CFS (MPa)", labelpad=1)
    cbar_slip_rate.set_label("Slip Rate (m/s)", labelpad=1)

    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.96, bottom=0.2, wspace=0.02, hspace=0.15
    )
    plt.savefig("plane5_compare_2d.pdf", bbox_inches="tight")
    plt.show()
