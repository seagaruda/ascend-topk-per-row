# tilelang-ascend 解读

仓库：<https://github.com/tile-ai/tilelang-ascend>（默认分支 `ascendc_pto`，431 stars / 193 forks，
最近推送 2026-11-13，本文件为当日快照）

## 它是什么

TileLang 的昇腾后端：用 TileLang 的 Python DSL 写 kernel，经 TVM 编译到昇腾 NPU。
README 明确了两条后端路线：

| 后端 | 分支 | 产物 |
|---|---|---|
| Ascend C + PTO | `ascendc_pto`（默认） | 生成 Ascend C / PTO 汇编风格代码 |
| AscendNPU IR | `npuir` | 生成 AscendNPU IR |

依赖：CANN ≥ 8.3.RC1、torch-npu ≥ 2.6.0.RC1；测试设备 A2 / A3。

## 成熟度信号（README 新闻流）

- 2025-09-29 开源
- 2025-11-07 自动插入同步指令；11-17 `T.printf` / `T.dump_tensor` 调试；
  11-25 自动 buffer 复用；12-08 `T.Parallel` 自动向量化
- 2026-01-15 `T.Pipelined` 软件流水；01-21 `torch_tl_ascend` PyTorch 集成；01-23 PTO 后端；01-29 编程指南
- 2026-03-03 shmem put/get 核间通信；03-12 ACLGraph 集成；03-16 pip wheel 安装；
  03-28 FlashAttention / SparseFlashAttention benchmark + 优化指南（PR #698 / #665）
- 2026-04-24 DeepSeek-V4 kernels

`examples/` 有 367 个条目，覆盖 GEMM、batch GEMM、elementwise、FA、SFA、linear attention / RNN、
softmax、normalization、activation、reduce、sort（`example_merge_sort.py`）、convolution、
cross-entropy、dispatch/combine、lightning indexer、`topk_selector`、ACLGraph。
`ci/operator_test_manifest.yaml`（276 行）为每个 example 登记对应的 pytest，说明「算子 = 实现 + 测试 +
manifest 条目」是它的贡献契约。

仓库还自带一批 `.agents/skills/`（`npu-arch`、`tilelang-ascend-tile-api`、
`tilelang-api-best-practices`、`tilelang-debug-helper` 等）和 `.opencode/agents/`
（op-analyst / op-developer / op-orchestrator / op-perf-tuner），把「怎么写、怎么调、怎么验收」
固化成了可复用的提示资产。

## 它不做的事

- 它不是推理引擎适配层：不处理 paged KV、不注册 `torch.ops._C_ascend`、不负责在图里替换上游 kernel。
  这些是 vllm-ascend 的职责（`csrc/` 手写 AscendC + aclnn 绑定 + monkey patch）。
- 它不承诺覆盖 CANN 已有算子，目标是「让新算子写得快」。

## 两个仓库的分工

```
        上游 vLLM (CUDA kernel)
                │
   ┌────────────┴─────────────┐
   │                          │
vllm-ascend              tilelang-ascend
（落地：aclnn/AscendC/Triton 替换 + 图捕获）   （造：Python DSL 快速写新 kernel）
   │                          │
   └────────► CANN / NPU ◄────┘
```

## 对 top_k_per_row 的意义

- TileLang 的并行模型是静态 tile（`T.Parallel` / `T.Pipelined`），适合 GEMM、attention、norm 这类
  规整数据流。`top_k_per_row` 的核心是**数据依赖的变长窗口 + 选择/归并 + 索引回填**，
  每个 row 的窗口长度不同、top-k 结果决定后续 gather 位置，不容易摊成静态 tile。
- 所以本仓库用 Ascend C 直写（能精确控制 UB 分块、`AscendC::TopK` 的 tiling、以及
  「位置 → 列索引」的 `Gather` 映射），而不是 TileLang。
- 但 TileLang 版仍有价值：`examples/topk_selector` 与 `examples/sort` 已经证明排序类
  kernel 可以写（原型阶段用它对齐语义比手写 AscendC 快很多），代价是归并路径要自己搭。
  若后续要做，建议顺序：TileLang 原型（正确性）→ AscendC 生产版（性能）→ 用本仓库的
  `tests/reference.py` 做三方对拍。
