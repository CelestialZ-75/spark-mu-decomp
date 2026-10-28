# CKC 代码来源与课程修改

来源：`FingerForce_Decode/example/gpu_MU_extract`，2026-10-25 引入分解、扩展、预处理、MU 放电率编码和质量检查五个模块。原算法说明与参考名称保留，不复制整个研究仓库；修改仅位于当前课程仓库，原研究文件未改动。

课程适配范围：

- 调整为包内导入，去掉外部电极布局导入并使用文件自带布局，清理未使用变量／导入及排版。
- 保留此前课程加入的 `CKCDecomposer.fit(eSig, ops=None)`：默认使用原 Torch 运算，Spark 路径接相关矩阵、投影、加权求和三个算子。求逆、候选选择、迭代和质量判断仍在 driver，约定见 [BACKEND_CONTRACT](../../../docs/BACKEND_CONTRACT.md)。
- `apply_templates` 增加可选 `polarities=None`。提供时校验每个 MU 的固定 ±1 方向，应用训练数据确定的方向屏蔽相反极性，同时复用训练阈值质心；不传时保持原有方向处理。该参数用于跨试次特征，避免根据测试数据重新选方向。

`SegmentDecomposer`、`apply_templates`、`MURateEncoder` 与质量函数均可从对应模块导入。课程的模板保存／载入、窗口对齐与 MLlib 预测入口由 `mu_course.prepare_prediction`、`mu_course.predict_force` 连接；完整使用方法见 [实操指南](../../../docs/START_HERE.md#step-7)。

这些科研源码供当前私有课程仓库团队运行，本文件不授予新的开源许可；未来公开前另行核对来源、权利与许可。
