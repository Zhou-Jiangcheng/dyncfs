# 快速开始：静态 CFS 小算例

这个算例包含一个 **1 km × 1 km、滑动量 1 m** 的源子断层，以及距离
**30、60、90 km** 的三个接收点。源深度 10 km，接收深度 5 km；
源与接收机制均为走向 30°、倾角 45°、滑动角 90°。
摩擦系数为 0.4，`B_pore=0`。

模型取自仓库中的 `examples/wenchuan/input/model.nd`，
静态建库使用前 24 个数值行。这是安装和计算流程示例，
不代表汶川地震重现，也没有完成网格收敛分析。

## 1. 运行

在仓库根目录、已激活的计算环境中执行：

```bash
python docs/examples/quickstart.py
```

Windows 非交互执行方式：

```powershell
conda run -n cfs python docs/examples/quickstart.py
```

脚本自动准备 CSV、复制模型、写入绝对路径 INI，然后建立 EDGRN/EDCMP
静态库，计算 CFS 并绘图。所有输出位于 `docs/_build/quickstart/`。

输出目录必须是新目录或空目录。重复运行时可指定：

```bash
python docs/examples/quickstart.py --output-dir docs/_build/quickstart-repeat
```

建库深度为 10、11 km，接收深度为 5 km，距离范围 1–121 km、
间隔 10 km。EDGRN 至少需要两个源深度，查询点应位于库的覆盖范围内。

## 2. 查看结果

| 路径，相对于输出目录 | 内容 |
|---|---|
| `quickstart.ini` | 生成的完整配置 |
| `input/` | 输入 CSV 和模型 |
| `grn_s/` | 静态格林函数库 |
| `results/static/stress_tensor_plane1.npy` | 形状 `(3,6)` 的 NED 应力张量，单位 Pa |
| `results/static/cfs_static_plane1.csv` | 三个接收点的 CFS，单位 Pa |
| `static_cfs.png` | 法向、剪切和库仑应力图，显示单位 kPa |
| `summary.json` | 环境、运行时间和检查结果 |

```{figure} ../_static/quickstart.png
:alt: 小型静态算例在三个距离处的法向、剪切和库仑应力变化。
:width: 100%

本地实际计算生成的图。正的法向应力表示张性变化。
```

脚本检查数组形状、有限且非零的数值，以及
`CFS = 剪切应力 + 0.4 × 法向应力`。
实际运行环境和结果列在[验证记录](../validation.md)。

## 3. 使用命令行接口

只准备输入，然后用 CLI 计算：

```bash
python docs/examples/quickstart.py --prepare-only --output-dir docs/_build/prepared
python -m dyncfs.main --config docs/_build/prepared/quickstart.ini --create-static-lib --compute-static-cfs
```

CLI 写出数值结果，图和汇总文件由完整算例脚本生成。
完整脚本及 INI 直接包含在[英文快速开始](../quickstart.md)中，
中英文说明共用同一个可执行示例。

## 4. 扩展到实际任务

更换源断层、接收断层和模型后，重新检查格林函数库的深度与距离覆盖。
动态计算还需要合理的 STF、采样和时间窗。

静态使用面积、滑动量与刚度，动态使用给定地震矩；
比较二者前应保证震源归一化一致。
启用零频校正还需要匹配的静态应力张量和有限的
`max_slowness`。参见[动态计算指南](../guides/dynamic.md)。
