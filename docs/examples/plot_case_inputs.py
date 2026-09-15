"""Plot the Wenchuan and Ludian finite-fault inputs: slip on each plane and moment rate.

Reads only the files in examples/<case>/input/, so it needs no computed results:
    python docs/examples/plot_case_inputs.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/_static/cases"
SLIP_CMAP = LinearSegmentedColormap.from_list(
    "slip", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED, GRID = "#3d3d3a", "#8a8a85", "#e4e3dc"
SERIES = ["#2a78d6", "#eb6834"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def read_plane(case, index, shape):
    rows = np.loadtxt(ROOT / f"examples/{case}/input/source_plane{index}.csv", delimiter=",", ndmin=2)
    assert len(rows) == shape[0] * shape[1]
    return rows


def slip_grid(rows, shape):
    """Slip as (n_dip, n_strike); rows are ordered i_strike * n_dip + i_dip."""
    return rows[:, 8].reshape(shape).T


def draw_slip(ax, rows, shape, x0, norm):
    grid = slip_grid(rows, shape)
    length, width = rows[0, 6], rows[0, 7]
    extent = [x0, x0 + shape[0] * length, shape[1] * width, 0]
    return ax.imshow(grid, extent=extent, cmap=SLIP_CMAP, norm=norm, interpolation="nearest")


def style_fault_axis(ax, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Down dip (km)")
    ax.tick_params(direction="out")


def moment_rate(rows, dt):
    stf = rows[:, 10:]
    area = stf.sum(axis=1) * dt
    rate = np.divide(stf, area[:, None], out=np.zeros_like(stf), where=area[:, None] > 0)
    return (rate * rows[:, 9, None]).sum(axis=0)


def style_rate_axis(ax, duration, scale):
    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Moment rate (10$^{%d}$ N m/s)" % scale)
    ax.grid(True, color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def wenchuan():
    shapes = [(22, 9), (6, 9), (8, 9), (62, 9), (17, 6)]
    planes = [read_plane("wenchuan", i + 1, s) for i, s in enumerate(shapes)]
    norm = Normalize(0, 9)
    fig = plt.figure(figsize=(12, 5.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[.62, 1], width_ratios=[1, 1.25], hspace=.42, wspace=.25)
    strip = fig.add_subplot(grid[0, :])
    x0 = 0.0
    for i in range(4):
        image = draw_slip(strip, planes[i], shapes[i], x0, norm)
        end = x0 + shapes[i][0] * planes[i][0, 6]
        strip.text((x0 + end) / 2, -3, "Plane %d" % (i + 1), ha="center", va="bottom", fontsize=9, color=INK)
        if i:
            strip.axvline(x0, color=INK, lw=.8)
        x0 = end
    strip.set_xlim(0, x0)
    strip.set_ylim(45, 0)
    strip.set_yticks([0, 20, 40])
    style_fault_axis(strip, "Along strike from the northeast end of plane 1 (km)")
    fig.colorbar(image, ax=strip, fraction=.02, pad=.01).set_label("Slip (m)")

    receiver = fig.add_subplot(grid[1, 0])
    draw_slip(receiver, planes[4], shapes[4], 0, norm)
    style_fault_axis(receiver, "Along strike (km)")
    receiver.set_title("Plane 5", fontsize=10)

    rate = fig.add_subplot(grid[1, 1])
    dt = 0.5
    t = np.arange(planes[0].shape[1] - 10) * dt
    total = sum(moment_rate(p, dt) for p in planes) / 1e19
    rate.axvspan(16, 28, color=GRID, lw=0)
    rate.text(22, total.max() * 1.03, "16–28 s", ha="center", va="bottom", fontsize=9, color=MUTED)
    rate.plot(t, total, color=INK, lw=2, label="All five planes")
    rate.plot(t, moment_rate(planes[3], dt) / 1e19, color=SERIES[0], lw=2, label="Plane 4")
    rate.set_ylim(0, total.max() * 1.15)
    rate.set_title("Moment-rate function", fontsize=10)
    style_rate_axis(rate, 100, 19)
    fig.savefig(ASSETS / "wenchuan-source.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def ludian():
    shapes = [(21, 10), (21, 10)]
    planes = [read_plane("ludian", i + 1, s) for i, s in enumerate(shapes)]
    norm = Normalize(0, .45)
    fig = plt.figure(figsize=(12, 4.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.1], hspace=.65, wspace=.28)
    for i in range(2):
        ax = fig.add_subplot(grid[i, 0])
        image = draw_slip(ax, planes[i], shapes[i], 0, norm)
        rows = planes[i]
        ax.set_title("Plane %d: strike %.0f°, dip %.0f°" % (i + 1, rows[0, 3], rows[0, 4]), fontsize=10)
        style_fault_axis(ax, "Along strike (km)" if i else "")
    fig.colorbar(image, ax=fig.axes, fraction=.025, pad=.02).set_label("Slip (m)")
    rate = fig.add_subplot(grid[:, 1])
    dt = 0.125
    t = np.arange(planes[0].shape[1] - 10) * dt
    rates = [moment_rate(p, dt) / 1e17 for p in planes]
    rate.plot(t, rates[0] + rates[1], color=INK, lw=2, label="Both planes")
    for values, color, label in zip(rates, SERIES, ("Plane 1", "Plane 2")):
        rate.plot(t, values, color=color, lw=1.6, label=label)
    rate.set_ylim(bottom=0)
    rate.set_title("Moment-rate function", fontsize=10)
    style_rate_axis(rate, 20, 17)
    fig.savefig(ASSETS / "ludian-source.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    wenchuan()
    ludian()
