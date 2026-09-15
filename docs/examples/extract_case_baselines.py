"""Extract displayed colors from the bundled PDF baselines (requires pypdf, Pillow).

This recovers display colors only, never the original CSV/NPY values.
Run with the documentation environment, then plot_case_studies.py with the
numerical environment. Outputs remain under docs/_build/case-reruns/baseline.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/_build/case-reruns/baseline"


def extract_meshes(page):
    color, clip, stack, vertices = (0., 0., 0.), None, [], []
    rect, pending_clip = None, False
    cells = defaultdict(list)
    for args, op in page.get_contents().operations:
        if op == b"q":
            stack.append((color, clip))
        elif op == b"Q":
            color, clip = stack.pop()
        elif op == b"rg":
            color = tuple(float(x) for x in args)
        elif op == b"g":
            color = (float(args[0]),) * 3
        elif op == b"re":
            rect = tuple(float(x) for x in args)
            vertices = []
        elif op in (b"W", b"W*"):
            pending_clip = True
        elif op == b"n":
            if pending_clip:
                clip = rect
                pending_clip = False
            vertices = []
        elif op == b"m":
            vertices = [tuple(float(x) for x in args)]
        elif op == b"l":
            vertices.append(tuple(float(x) for x in args))
        elif op in (b"f", b"f*", b"B", b"B*"):
            if clip is not None and len(vertices) in (4, 5):
                points = set(vertices)
                xs, ys = sorted(set(v[0] for v in points)), sorted(set(v[1] for v in points))
                if len(points) == 4 and len(xs) == len(ys) == 2:
                    x, y = sum(xs)/2, sum(ys)/2
                    w, h = xs[1]-xs[0], ys[1]-ys[0]
                    if 0 < w < clip[2]/2 and 0 < h < clip[3]/2:
                        cells[clip].append((x,y,w,h,color))
            vertices = []
        elif op in (b"S", b"s"):
            vertices = []
    panels = []
    for clip, group in cells.items():
        mode, count = Counter((round(c[2],4), round(c[3],4)) for c in group).most_common(1)[0]
        if count < 100:
            continue
        group = [c for c in group if abs(c[2]-mode[0]) < 0.001 and abs(c[3]-mode[1]) < 0.001]
        xs = sorted(set(round(c[0],4) for c in group))
        ys = sorted(set(round(c[1],4) for c in group), reverse=True)
        if len(xs)*len(ys) != len(group):
            raise ValueError("Expected a complete rectangular color mesh")
        mapping = {(round(c[0],4),round(c[1],4)): c[4] for c in group}
        panels.append({"clip_pdf_points":clip, "shape":[len(ys),len(xs)],
                       "rgb_top_down":[[mapping[(x,y)] for x in xs] for y in ys]})
    return sorted(panels, key=lambda p: (-round(p["clip_pdf_points"][1],3),p["clip_pdf_points"][0]))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files = {"ludian": "examples/ludian/compare_cfs_ludian_5.00.pdf",
             "wenchuan_dynamic": "examples/wenchuan/plane5_compare_2d.pdf",
             "wenchuan_static": "examples/wenchuan/compare_static_cfs_wenchuan.pdf"}
    report = {"meaning":"Displayed PDF colors; no original numerical values are recovered.", "figures":{}}
    for name, relative in files.items():
        path = ROOT / relative
        page = PdfReader(path).pages[0]
        entry = {"file":relative,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                 "meshes":extract_meshes(page), "images":[]}
        for number, img in enumerate(page.images):
            target = OUT / f"{name}-image{number+1}.png"
            img.image.save(target)
            entry["images"].append({"file":target.name,"size":list(img.image.size)})
        report["figures"][name] = entry
        print(name, [p["shape"] for p in entry["meshes"]], entry["images"])
    assert [p["shape"] for p in report["figures"]["ludian"]["meshes"]] == [[200,200],[200,200]]
    assert len(report["figures"]["wenchuan_dynamic"]["meshes"]) == 21
    (OUT/"pdf-colors.json").write_text(json.dumps(report),encoding="utf-8")


if __name__ == "__main__":
    main()
