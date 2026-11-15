# CUDA → CANN 算子缺口扫描（2026-11-15 快照）

目标：在 vLLM 生态里找出「CUDA 有专用 kernel、CANN 生态没有任何等价实现、且在推理热路径上」的算子。

## 方法

1. 从上游 vLLM 提取所有自定义 CUDA 算子：
   - `vllm/_custom_ops.py` 中 `torch.ops._C.*` / `torch.ops.vllm.*` 的调用名（66 个）；
   - `vllm/csrc/ops.h` 的声明（确认签名与语义）。
2. 在 `vllm-project/vllm-ascend` 全仓库用 GitHub code search 逐个核对是否已有 NPU 实现（aclnn 自定义算子、`csrc/` 手写 AscendC、`vllm_ascend/ops/triton/` 的 Triton 移植、monkey patch）。
3. 交叉验证搜索本身可靠：拿几个已知有实现的算子做对照。

## 对照与结果

| 算子 | vllm-ascend 命中 | 判定 |
|---|---|---|
| `vocab_parallel_embedding` | 35 | 已覆盖（对照项） |
| `hadacore_transform` | 26 | 已覆盖：QuaRot Hadamard 走 Triton |
| `merge_attn_states` | 0（但 `ops/triton/attention_update.py` 是等价物） | 已覆盖 |
| `selective_scan_fwd` | 0（Mamba 走 `ops/triton/mamba_*`） | 部分覆盖 |
| `fused_qk_norm_rope` | 0（`patch/worker/patch_ascend_ops.py` 用两次 `npu_rms_norm` 顶替） | 用拆分绕过，无融合算子 |
| **`top_k_per_row_prefill/decode`** | **0，且无任何等价物** | **缺口** |

上游侧证据：

- `vllm/csrc/topk.cu`：324 行，文件头写明 “Persistent TopK kernel for DeepSeek V3 sparse attention indexer”；
- `vllm/model_executor/layers/indexer_topk.py`：`deep_select` / `cooperative` / `persistent` / `flashinfer` / `flashinfer_gvr2` 五个后端，全部仅 CUDA；
- `vllm/csrc/ops.h`：`top_k_per_row_prefill` 与 `top_k_per_row_decode` 两个入口，参数里带 `stride0/stride1`，直接写进 `raw_topk_indices`（int32，不足 topk 时 -1 填充）。

CANN 侧为什么没有替代品：

- `aclnnTopK` 只处理最后一维、固定 `k`、同时返回 values+indices，无法表达
  「每行一个 `[row_start, row_end)` 变长窗口」+「写回已有 int32 buffer」+「-1 填充」；
- 组合 `aclnnSort` + `aclnnGather` 需要把整行排序（O(W log W)）并分配大临时张量，
  在 `W = 32K…128K`、每层每 step 调用的场景下不可接受；
- 目前 NPU 路径只能整块回退到 CPU（D2H + CPU 计算 + H2D，并强制同步设备队列）。

## 结论

`top_k_per_row` 是这批候选里唯一「CUDA 有专用 kernel / CANN 无任何等价算子 /
位于 DeepSeek-V3.2 与 V4 稀疏注意力 indexer 热路径」的算子，因此作为本仓库的实现对象。

## 保留意见

- 缺口判断是快照，vllm-ascend 迭代很快（近一年 `csrc/` 每月新增多个 AscendC 算子），
  落地前应重跑第 2 步；
- "0 命中"只说明没有同名实现，不等价于没有等价物；上表中 `merge_attn_states`
  就是这种情况，所以每个 0 命中项都做了二次人工核对；
- 若后续 `Ascend/DeepSelect`（TopK kernels for DSA，目前只有 CUDA kernel 源码）
  出现 NPU 后端，本仓库应转为语义对拍基准而非替代实现。
