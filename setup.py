import os
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

        fortran_flags = "-O3"
        for src_folder, bin_name in fortran_subdirs.items():
            fortran_src_dir = os.path.join(fortran_src_root, src_folder)
            output_binary = os.path.join(exec_dir, bin_name)
            extra = ""
            if src_folder == "qseis2025_src":
                extra = " -ffixed-line-length-none"
            compile_command = f'gfortran "{fortran_src_dir}"/*.f {fortran_flags}{extra} -o "{output_binary}"'
            print(f"[dyncfs] Compiling {src_folder} -> {output_binary}")
            result = subprocess.run(compile_command, shell=True)
            if result.returncode != 0:
                raise RuntimeError(f"Fortran compilation failed for {src_folder}")


setup(
    name="dyncfs",
    version="1.0.0",
    author="Zhou Jiangcheng",
    author_email="zhoujcpku@outlook.com",
    description="A Python library for computing Coulomb Failure Stress Change.",
    long_description=open(
        os.path.join(project_root, "README.md"), encoding="utf-8"
    ).read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Zhou-Jiangcheng/dyncfs",
    packages=find_packages(),
    include_package_data=True,  # 配合 package_data/MANIFEST.in
    package_data={"dyncfs": ["exec/*"]},
    # install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    entry_points={"console_scripts": ["dyncfs=dyncfs.main:main"]},
    cmdclass={"build_py": CustomBuildPy},  # type: ignore
)
