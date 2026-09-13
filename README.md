# 3DGS-LineDrawing

基于 RaDe-GS 复现 Lego 场景，并实现海报思路的交互式特征线渲染。

本仓库直接包含修改后的 RaDe-GS 与 CUDA 光栅器源码，不依赖 Git submodule。上游许可及第三方说明见 `external/RaDe-GS/LICENSE.md` 和 `THIRD_PARTY_NOTICES.md`；其中 Gaussian Splatting 代码仅限研究和评估用途。

已完成：官方 Lego 数据下载、`3dgs` Conda 环境内 CUDA 扩展编译、30,000 次训练和 200 个测试视角渲染。

交互查看器：http://127.0.0.1:17865 。重新启动运行 `scripts\start_viewer.cmd`。

查看器的主画面使用无 PNG 编码的二进制帧传输，并提供实际显示 FPS 与独立 GPU 基准。下方诊断预览固定为初始视角，不参与实时更新。

- [操作说明与线场算法](LINE_DRAWING.md)

- [复现命令、代码版本与兼容修改](RADEGS_RUN.md)
- [环境说明](ENVIRONMENT.md)
- [海报方法解读](REPRODUCTION_NOTES.md)

基础 RGB 的最终 PNG 测试集：PSNR 33.01 dB，SSIM 0.9753。特征线使用不透明度、深度、法线、色调与 Top-k Gaussian 贡献信息；具体公式是基于海报的复现设计，尚未实现跨视角稳定的三维曲线。

数据集、训练结果、PLY 模型、日志和编译产物不会提交到 Git。请按照 `RADEGS_RUN.md` 下载数据并复现模型；viewer 默认读取 `outputs/lego_radegs/point_cloud/iteration_30000/point_cloud.ply`。
