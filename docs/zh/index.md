# 中文入门

DynCFS 根据有限断层模型计算静态和动态库仑破裂应力变化。
当前代码通过 pygrnwang 调用 EDGRN2/EDCMP2、QSEIS2025 和 QSSP2020。

文档沿用 pygrnwang 的 Sphinx、MyST 和 PyData 主题。
本节提供中文入门；参数参考、计算流程和 API 以英文维护。

```{toctree}
:maxdepth: 1

installation
quickstart
```

## 从哪里开始

1. 按[安装说明](installation.md)准备环境。
2. 运行[小型静态算例](quickstart.md)，完成建库、计算和绘图。
3. 阅读[输入文件](../input-files.md)和[科学约定](../conventions.md)。
4. 按[配置参考](../configuration.md)设置实际震源、接收断层和模型。
5. 需要动态结果时，继续阅读[动态计算](../guides/dynamic.md)。

## 使用前确认

- 应力张量顺序为 `[NN, NE, ND, EE, ED, DD]`，坐标采用北、东、下（NED）。
- 应力输出单位为 Pa；完整构造应力在 INI 中用 MPa，读取时转换为 Pa。
- 源断层和接收断层 CSV 均无表头，深度用 km，滑动量用 m。
- 动态并行计算脚本必须有 `if __name__ == "__main__":`。
- `fixed_obs_depth<=0` 时，CLI 和完整流程跳过固定深度计算。
- 这里只提供本地文档和本地构建说明，没有配置上传或发布流程。
