# 安装

运行环境与文档构建环境是两回事。运行计算需要 Python 数值库及
pygrnwang 提供的 Fortran 可执行程序；仅构建 HTML 不需要这些程序。

## 安装程序

本地项目元数据要求 Python 3.9 及以上、
`pygrnwang>=3.0.0` 和 `matplotlib>=3.9.2`。

```bash
python -m pip install dyncfs
python -c "import dyncfs; print(dyncfs.__version__)"
dyncfs --help
```

导入成功只能证明 Python 接口可用。请通过[快速开始](quickstart.md)
检查一次实际静态计算。

## 使用本地源码

将兼容的 pygrnwang 3.x 源码和 dyncfs 放在同一个父目录。
从该父目录运行：

```bash
conda create -n cfs -c conda-forge python=3.12 numpy scipy pandas obspy tqdm matplotlib
conda activate cfs
conda install -c conda-forge gfortran
python -m pip install -e ./pygrnwang
python -m pip install -e ./dyncfs
cd dyncfs
```

编译器和各平台的具体要求以对应 pygrnwang 源码中的安装说明为准。
安装匹配的二进制 wheel 可能不需要编译；从源码构建则需要 Fortran 工具链。

## Windows 注意事项

保持 Conda 环境激活，或在自动化命令中使用：

```powershell
conda run -n cfs python docs/examples/quickstart.py
```

不要直接调用未激活环境下的 `python.exe`，否则可能缺少数值库依赖的 DLL。
确认所需的 `edgrn2.exe`、`edcmp2.exe`、`qseis2025.exe`
或 `qssp2020.exe` 可被当前环境找到。

INI 中使用较短的绝对路径，Windows 可以写成
`C:/work/case/input`，不要额外添加引号。
相对路径以运行目录为基准，不以 INI 所在目录为基准。

动态本机并行不要求 MPI。走时通过 pygrnwang 的 Java 子进程或 ObsPy
后端计算。GIF 功能还需要单独安装 `imageio`。

## 本地构建文档

使用 Python 3.12 安装 `docs/requirements.txt` 后：

```bash
python docs/api/check_coverage.py
python -m sphinx -b html -W --keep-going docs docs/_build/html
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

浏览器访问 `http://127.0.0.1:8000/`，或直接打开
`docs/_build/html/index.html`。完整环境说明见[文档维护](../development.md)。
