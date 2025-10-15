import os
import glob
import platform
import shutil
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py

project_root = os.path.dirname(os.path.abspath(__file__))

# req_file = os.path.join(project_root, "requirements.txt")
# with open(req_file, encoding="utf-8") as f:
#     requirements = [
#         line.strip() for line in f if line.strip() and not line.startswith("#")
#     ]

platform_exec = "exe" if platform.system() == "Windows" else "bin"


def _compile_dir(src_dir: str, out_bin: str, extra_flags: list[str]) -> None:
    build_dir = os.path.join(src_dir, "_build_mod")
    os.makedirs(build_dir, exist_ok=True)

    pats = ["*.f", "*.for", "*.f90", "*.F", "*.F90"]
    files = []
    for p in pats:
        files.extend(glob.glob(os.path.join(src_dir, p)))
    if not files:
        raise RuntimeError(f"[dyncfs] No Fortran sources in {src_dir}")

    remaining = sorted(files)
    compiled_objs = []
    last_err = ""
    for _ in range(10):
        progressed = False
        for f in list(remaining):
            obj = os.path.join(build_dir, os.path.basename(f) + ".o")
            cmd = [
                "gfortran",
                "-c",
                "-O3",
                "-J",
                build_dir,
                "-I",
                build_dir,
                "-I",
                src_dir,
                *extra_flags,
                f,
                "-o",
                obj,
            ]
            proc = subprocess.run(cmd, cwd=src_dir, text=True, capture_output=True)
            if proc.returncode == 0:
                remaining.remove(f)
                compiled_objs.append(obj)
                progressed = True
            else:
                last_err = proc.stderr or proc.stdout
        if not remaining:
            break
        if not progressed:
            raise RuntimeError(
                "[dyncfs] Fortran compile stalled.\n"
                f"Uncompiled: {[os.path.basename(x) for x in remaining]}\n"
                f"Last error:\n{last_err}"
            )

    link_cmd = ["gfortran", *compiled_objs, "-o", out_bin]
    subprocess.run(link_cmd, cwd=src_dir, check=True)


class CustomBuildPy(_build_py):
    def run(self):
        _build_py.run(self)

        if not shutil.which("gfortran"):
            raise ValueError(
                "Please install gfortran and ensure "
                "that command 'gfortran' can be directly called"
            )
        if not shutil.which("jar"):
            raise ValueError(
                "Please install java and ensure that "
                "command 'jar' can be directly called"
            )

        exec_dir = os.path.join(project_root, "dyncfs", "exec")
        os.makedirs(exec_dir, exist_ok=True)

        fortran_src_root = os.path.join(project_root, "fortran_src_codes")
        fortran_subdirs = {
            "edgrn2.0_src": f"edgrn2.{platform_exec}",
            "edcmp2.0_src": f"edcmp2.{platform_exec}",
            "qssp2020_src": f"qssp2020.{platform_exec}",
            "qseis2025_src": f"qseis2025.{platform_exec}",
        }

        for src_folder, bin_name in fortran_subdirs.items():
            fortran_src_dir = os.path.join(fortran_src_root, src_folder)
            output_binary = os.path.join(exec_dir, bin_name)

            extra = []
            env_fflags = os.environ.get("DYNCFS_FFLAGS", "")
            if env_fflags:
                extra += env_fflags.split()
            if src_folder == "qseis2025_src":
                extra += ["-ffixed-line-length-none"]

            print(f"[dyncfs] Compiling {src_folder} -> {output_binary}")
            _compile_dir(fortran_src_dir, output_binary, extra)


setup(cmdclass={"build_py": CustomBuildPy})
