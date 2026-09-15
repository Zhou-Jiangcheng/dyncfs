"""Plot completed Wenchuan and Ludian runs with the original scales and sampling.

If extract_case_baselines.py has been run, displayed colors are also compared
with the repository PDFs; that comparison does not recover old CSV/NPY values.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, Normalize
from PIL import Image
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dyncfs.configuration import CfsConfig
from dyncfs.utils import cal_grid_num

CASES = ROOT/"docs/_build/cases"
BASE = ROOT/"docs/_build/case-reruns/baseline"
ASSETS = ROOT/"docs/_static/cases"
COLORS = ["blue","cyan","white","yellow","red"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})


def load_plot_module(case, file):
    spec = importlib.util.spec_from_file_location(case+"_plot", ROOT/"examples"/case/file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    return module


def config_for(case):
    c = CfsConfig()
    c.read_config(str(CASES/case/"case.ini"))
    return c


def palette(bounds, colors=COLORS):
    cmap = LinearSegmentedColormap.from_list("case",colors)
    return cmap, BoundaryNorm(bounds,cmap.N,clip=True)


def save(fig, name):
    ASSETS.mkdir(parents=True,exist_ok=True)
    fig.savefig(ASSETS/(name+".png"),dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def compare_colors(new, old):
    new, old = np.asarray(new)[...,:3], np.asarray(old)[...,:3]
    if new.shape != old.shape:
        raise ValueError(f"Color shape mismatch: {new.shape} != {old.shape}")
    error = np.max(np.abs(new-old),axis=-1)
    return {"display_cells":int(error.size),
            "same_color_cells":int(np.sum(error < 1/255+1e-7)),
            "same_color_percent":float(100*np.mean(error < 1/255+1e-7)),
            "mean_absolute_rgb_error_0_to_255":float(np.abs(new-old).mean()*255),
            "meaning":"Display colors within one 8-bit RGB level; not a numeric CFS tolerance."}


def wenchuan_plane(baseline):
    c = config_for("wenchuan")
    out = CASES/"wenchuan/results"
    dynamic = np.loadtxt(out/"dynamic/cfs_dynamic_plane5.csv",delimiter=",")/1e6
    source = np.loadtxt(CASES/"wenchuan/input/source_plane5.csv",delimiter=",")
    integral = source[:,10:].sum(axis=1)*0.5
    rate = np.divide(source[:,10:],integral[:,None],out=np.zeros_like(source[:,10:]),
                     where=integral[:,None]!=0)*source[:,8,None]
    indices = list(range(32,60,4))
    cmap, norm = palette(np.linspace(-1,1,21))
    slip_cmap, slip_norm = palette(np.linspace(0,0.5,6),["white","red"])
    fig, axes = plt.subplots(7,3,figsize=(10,10))
    fig.subplots_adjust(left=.09,right=.97,top=.95,bottom=.15,wspace=.08,hspace=.22)
    report = []
    for row, nt in enumerate(indices):
        static = np.loadtxt(out/f"static/time/{nt}/cfs_static_plane5.csv",delimiter=",")/1e6
        arrays = [static.reshape(17,6).T,dynamic[:,nt].reshape(17,6).T,rate[:,nt].reshape(17,6).T]
        entry = {"time_s":nt*.5}
        for col, (ax, data) in enumerate(zip(axes[row],arrays)):
            cm, nm = (slip_cmap,slip_norm) if col==2 else (cmap,norm)
            im = ax.imshow(data,origin="upper",extent=[0,85,30,0],cmap=cm,norm=nm,interpolation="nearest")
            ax.set_yticks([0,10,20,30] if col==0 else [])
            ax.set_xticks([0,20,40,60,80] if row==6 else [])
            if row==0:
                ax.set_title(["Quasi-static CFS","Dynamic CFS","Receiver-plane slip rate"][col])
            if col==0:
                ax.text(.02,.9,f"{nt*.5:g} s",transform=ax.transAxes,
                        va="top",bbox={"facecolor":"white","edgecolor":"none","alpha":.75})
                ax.set_ylabel("Dip (km)")
            if row==6:
                ax.set_xlabel("Strike (km)")
            if baseline:
                old = baseline["wenchuan_dynamic"]["meshes"][row*3+col]["rgb_top_down"]
                entry[["quasi_static","dynamic","slip_rate"][col]] = compare_colors(cm(nm(data)),old)
        report.append(entry)
    for col in range(3):
        box = axes[-1,col].get_position()
        bar = fig.add_axes([box.x0+box.width*.05,.055,box.width*.90,.017])
        cm, nm = (slip_cmap,slip_norm) if col==2 else (cmap,norm)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=nm,cmap=cm),cax=bar,orientation="horizontal",
                     ticks=[0,.2,.4] if col==2 else [-1,-.5,0,.5,1]).set_label("m/s" if col==2 else "MPa")
    save(fig,"wenchuan-plane-rerun")
    return {"snapshots":report,"dynamic_shape":list(dynamic.shape)}


def ludian(baseline):
    c = config_for("ludian")
    out = CASES/"ludian/results"
    static = np.loadtxt(out/"static/cfs_oop_static_dep_5.00.csv",delimiter=",")
    dynamic = np.loadtxt(out/"dynamic/cfs_oop_dynamic_dep_5.00.csv",delimiter=",")
    peak = np.maximum(dynamic.max(axis=1),0)-np.minimum(dynamic.min(axis=1),0)
    shape = (cal_grid_num(c.obs_lat_range,.01),cal_grid_num(c.obs_lon_range,.01))
    # Same zoom and crop as original plot, with robust endpoint counting.
    data = [zoom(values.reshape(shape),5,order=1,mode="nearest",prefilter=False)[:200,:200]
            for values in (static,peak)]
    module = load_plot_module("ludian","plot_compare_cfs_oop_fix_dep.py")
    events = np.loadtxt(ROOT/"examples/ludian/after_ludian.txt")
    module.lon_af_1d,module.lat_af_1d = events[:133,0],events[:133,1]
    module.lon_af_rest,module.lat_af_rest = events[133:,0],events[133:,1]
    cmaps = [palette(np.linspace(-.25,.25,11)),palette(np.linspace(0,1,11),["yellow","orange","red"])]
    fig, axes = plt.subplots(1,2,figsize=(12,6))
    fig.subplots_adjust(left=.075,right=.98,bottom=.2,top=.9,wspace=.18)
    report = {}
    for col, (ax, arr, (cm,nm)) in enumerate(zip(axes,data,cmaps)):
        _, im = module.plot_ax(ax,arr,c.obs_lon_range,c.obs_lat_range,.002,.002,cm,nm)
        module.draw_fault_intersections(ax,[str(CASES/f"ludian/input/source_plane{i}.csv") for i in (1,2)],
                                      5,c.obs_lon_range,c.obs_lat_range,.002,.002)
        if col:
            y,x = np.indices(arr.shape)
            ax.contour(x,y,arr/1e6,levels=np.arange(0,1.01,.1),colors="k",linewidths=.5,zorder=90)
        ax.set_title(["Static CFS, 5 km","Dynamic CFS, peak-to-peak"][col])
        fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.05,pad=.16,
                     extend="max" if col else "both").set_label("MPa")
        if baseline:
            old = baseline["ludian"]["meshes"][col]["rgb_top_down"]
            report[["static","dynamic_peak_to_peak"][col]] = compare_colors(cm(nm(arr[::-1]/1e6)),old)
    axes[0].set_ylabel("Latitude (deg)")
    save(fig,"ludian-rerun")
    report.update(grid_shape=list(shape),dynamic_shape=list(dynamic.shape),
                  peak_to_peak_min_Pa=float(peak.min()),peak_to_peak_max_Pa=float(peak.max()))
    return report


def raster_like_pdf(data, cmap, norm, size):
    fig = plt.figure(figsize=(size/100,size/100),dpi=100)
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(data,origin="lower",cmap=cmap,norm=norm,interpolation="nearest",aspect="auto")
    ax.set_axis_off()
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba()).copy()[...,:3]/255
    plt.close(fig)
    return array


def wenchuan_static(baseline):
    c = config_for("wenchuan")
    run = json.loads((CASES/"wenchuan/run.json").read_text())
    spacing = run["map_spacing_deg"]
    shape = (cal_grid_num(c.obs_lat_range,spacing),cal_grid_num(c.obs_lon_range,spacing))
    arrays = [np.loadtxt(CASES/f"wenchuan/results/{label}/cfs_static_dep_15.00.csv",delimiter=",").reshape(shape)
              for label in ("static_half","static_layer")]
    arrays.append(arrays[1]-arrays[0])
    module = load_plot_module("wenchuan","plot_compare_static_cfs_fix_dep.py")
    c.source_inds = [1,2,3,4,5]
    module.config = c
    module.obs_depth = 15
    module.obs_lat_range,module.obs_lon_range = c.obs_lat_range,c.obs_lon_range
    module.obs_delta_lat = module.obs_delta_lon = spacing
    module.cmap = LinearSegmentedColormap.from_list("case",COLORS)
    module.norm = Normalize(-.25,.25)
    fig, axes = plt.subplots(1,3,figsize=(14,5))
    report = {}
    for i,(ax,data,label,old_id) in enumerate(zip(axes,arrays,["Half-space","Layered","Layered minus half-space"],[1,3,4])):
        module.plot_ax(ax,data,focal_mechanism=[223,47,131] if i==2 else None)
        ax.set_title(label)
        ax.set_xlabel("Longitude (deg)")
        if baseline:
            old = np.asarray(Image.open(BASE/f"wenchuan_static-image{old_id}.png").convert("RGB"))/255
            new = raster_like_pdf(data/1e6,module.cmap,module.norm,old.shape[0])
            report[label] = compare_colors(new,old)
    axes[0].set_ylabel("Latitude (deg)")
    fig.subplots_adjust(left=.06,right=.98,top=.88,bottom=.25,wspace=.2)
    bar = fig.add_axes([.32,.08,.4,.03])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=module.norm,cmap=module.cmap),cax=bar,
                 orientation="horizontal",extend="both").set_label("CFS change (MPa)")
    save(fig,"wenchuan-static-rerun")
    report.update(grid_shape=list(shape))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure",choices=["all","ludian","wenchuan-plane","wenchuan-static"],default="all")
    args = parser.parse_args()
    colors = BASE/"pdf-colors.json"
    reference = json.loads(colors.read_text())["figures"] if colors.exists() else None
    report_file = ROOT/"docs/_build/case-reruns/comparison.json"
    report = json.loads(report_file.read_text()) if report_file.exists() else {
        "baseline":"Repository PDF display colors only; original CSV/NPY unavailable.",
        "date":"2026-09-14","figures":{}}
    functions = {"ludian":ludian,"wenchuan-plane":wenchuan_plane,"wenchuan-static":wenchuan_static}
    for name in functions if args.figure=="all" else [args.figure]:
        report["figures"][name] = functions[name](reference)
        report_file.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(name,json.dumps(report["figures"][name]))


if __name__ == "__main__":
    main()
