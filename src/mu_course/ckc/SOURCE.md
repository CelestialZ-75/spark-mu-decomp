# CKC 代码来源

本目录从 FingerForce_Decode/example/gpu_MU_extract 复制，日期 2026-10-25。
只引入分解、扩展、预处理、放电率编码和质量检查这五个模块，不复制整个研究仓库。
原算法文件中的方法说明与参考名称保留。

ckc_decomposer.py 已包含此前课程实验加入的 fit(eSig, ops=None) 接口，默认使用原 Torch 运算。
本次保留数学流程，只调整包内导入、去掉外部电极布局导入（使用原文件自带布局），
以及清理未使用变量/导入和统一排版。原研究文件没有修改。

SegmentDecomposer、apply_templates、MURateEncoder 和质量函数可从 mu_course.ckc 对应模块导入。
这些辅助函数可用不代表指力预测流水线已完成。

这些科研源码随当前私有课程仓库提供，供团队运行。此文件不授予新的开源许可。
未来公开源码或数据前，应单独核对来源、权利和许可。
