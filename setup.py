import os
import platform
import sys
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop

# Try to import editable_wheel (for `pip install -e .`)
try:
    from setuptools.command.editable_wheel import editable_wheel as _editable_wheel
except ImportError:
    # If the setuptools version is too old and does not support PEP 660, set it to None
    _editable_wheel = None

project_root = os.path.dirname(os.path.abspath(__file__))


def install_binaries(target_exec_dir):
    """
    Core logic: compile Fortran and copy all binaries/Jar files into conda's bin directory
    target_exec_dir: directory to store compilation outputs (build directory or source directory)
    """
    print(f"[dyncfs] Starting custom installation logic...")

    # Ensure the output directory exists
    os.makedirs(target_exec_dir, exist_ok=True)

    # Get the bin directory of the current conda environment
    if platform.system() == "Windows":
        env_bin_dir = os.path.join(sys.exec_prefix, "Scripts")
    else:
        env_bin_dir = os.path.join(sys.exec_prefix, "bin")
    print(f"[dyncfs] Target environment bin: {env_bin_dir}")


# --- Custom command classes ---


class CustomBuildPy(_build_py):
    """Controls `pip install .`"""

    def run(self):
        _build_py.run(self)
        # Standard install: compile into build/lib/dyncfs/exec
        exec_dir = os.path.join(self.build_lib, "dyncfs", "exec")
        install_binaries(exec_dir)


class CustomDevelop(_develop):
    """Controls `python setup.py develop`"""

    def run(self):
        _develop.run(self)
        # Development install: compile directly into the source directory dyncfs/exec
        exec_dir = os.path.join(project_root, "dyncfs", "exec")
        install_binaries(exec_dir)


# Collect cmdclass
cmd_classes = {
    "build_py": CustomBuildPy,
    "develop": CustomDevelop,
}

# Extra handling for `pip install -e .` (PEP 660)
if _editable_wheel:

    class CustomEditableWheel(_editable_wheel):
        """Controls `pip install -e .`"""

        def run(self):
            _editable_wheel.run(self)
            # Editable install is also treated as development mode; compile into the source directory
            exec_dir = os.path.join(project_root, "dyncfs", "exec")
            install_binaries(exec_dir)

    cmd_classes["editable_wheel"] = CustomEditableWheel


setup(cmdclass=cmd_classes)
