# Qwen3.8-Flash-Next（Qwen4Exp）SM89 适配记录

本文档记录将 `Qwen3.8-Flash-Next`（`model_type=qwen4_exp`）移植到 `vllm-backport`（SM89/Ada 基线）的过程、踩坑与最终实测结果。官方 vLLM 不提供 SM89 内核，需自行移植。

## 1. 背景与目标

- 模型：`Qwen/Qwen3.8-Flash-Next-FP8`（官方 FP8 checkpoint，约 172.8 GiB / 131 个 safetensors shard）
- 架构：Qwen4Exp，125B MoE + PLE（N-gram embedding）+ QSA（稀疏注意力）混合
  - 每 token 激活 ~6B，48 层（每 4 层一个 full-attention），`ple_layer_ids=[2]`，`ngram_size=3`
  - 512 专家 top10，`moe_intermediate_size=640`，`hidden_size=2560`
  - QSA：`indexer_compress_ratio=4`，`indexer_budget=2048`
- 目标：在 8x RTX 4090 48GB（SM89）上完成移植、编译、实机推理与长上下文验证
- 参考实现：上游 `vllm-qwen38next`（peakcrosser7/vllm:release/qwen38next），并借鉴 SGLang / TokenSpeed 的 Qwen4Exp 适配思路

## 2. 硬件与软件环境

| 项 | 值 |
| --- | --- |
| GPU | 8x NVIDIA RTX 4090 48GB（Ada / SM89） |
| CPU / 内存 | 双路 Xeon Gold 6530，565 GiB RAM |
| 服务器 | 本地 8x RTX 4090 48GB 服务器 |
| CUDA | 13.0（`/data/cuda-13.0`，JIT 用） |
| Python | 3.12 venv `/data/models/vllm-qwen4exp-env`（从生产 vllm-backport-env 克隆后重建 editable 安装） |
| 源码 | `/data/models/vllm-qwen4exp-src`（独立目录，editable） |
| 分支 | `port/qwen4-exp` |

## 3. 移植范围

在 `port/qwen4-exp` 分支上完成，主体提交：

| 提交 | 内容 |
| --- | --- |
| `94bfbca30` | 核心移植：`vllm/models/qwen4_exp` 模型目录、Qwen4ExpConfig + registry + MTP convertor、GDN SigmoidGate kernel + `output_gate_activation`、per-Spec mamba state copy、PLE ngram context、Qwen4ExpMTPProposer |
| `fd8152d00` | `CircularBufferSpec` + `prefix_cacheable`（QSA 环形 KV） |
| `d4070cb6c` | backfill 集成 hooks + 移植单测（config/ple/spec_decode） |
| `a823df40a` | 移植 weight-loading / checkpoint-mapper 测试 |
| `36e9cbdef` | `llm_base_proposer` 多模态 `image_token_index`（Qwen4Exp 作 drafter） |

移植后 72 个上游单测全部通过（spec_decode 6 + config 24 + ple 12 + weight_loading 30）。

### 3.1 编译期改动

- `csrc`：GDN fused kernel 增加 `SigmoidGate` 模板参数与 `output_gate_activation` 运行时参数（SM89 可编译）
- QSA 用 `persistent_topk`（SM89），非 SM90+ 的 `cooperative_topk`
- `TORCH_CUDA_ARCH_LIST=8.9`，`VLLM_VERSION_OVERRIDE=0.5.2`

### 3.2 模型/集成层

- `vllm/config/compilation.py`：`_attention_ops` 加入 3 个 Qwen4Exp 切割点 op
- `vllm/model_executor/models/config.py`：`MODELS_CONFIG_MAP` 注册 Qwen4Exp 架构（`_strip_qwen4_exp_mrope` 等）
- `vllm/model_executor/layers/vocab_parallel_embedding.py`：支持 `quant_method` 参数与 FP8 masked_fill（PLE FP8 embedding 的 TP 归约）

## 4. 实机验证中发现的移植遗漏与修复

本轮验证按"启动 → 小样本 → 长上下文"推进，逐层暴露并修复了以下问题。

### 4.1 FP8 MoE block-scale refinement（TP 分片整除性）

- 现象：`--tensor-parallel-size 8` 启动即失败：gate/up 每个 TP shard 输出 `640/8=80`，不能被 FP8 `block_n=128` 整除。
- 修复（3 个文件协同，对齐上游）：
  - `fused_moe/oracle/fp8.py`：新增 `refine_fp8_moe_block_shape`（`gcd` 推导无损细粒度块）
  - `quantization/fp8.py`：`Fp8MoEMethod` 按 refined block 建 scale 网格、QuantKey 与 kernel block shape
  - `fused_moe/routed_experts.py`：加载时把 checkpoint 的 128x128 scale 无损展开到 refined 网格
- 说明：Qwen3.8 的 80 维分片 `gcd(128,128,80,2560)=16 < 32` 无法精化——正确做法不是降 TP 或放宽检查，而是使用 **EP8**（每卡 64 个完整专家，专家维保持 640）。三套参考实现（vLLM/SGLang/TokenSpeed）对 Qwen3.8 均以"EP 承载专家、TP 承载注意力"为执行拓扑。

### 4.2 QSA KV cache dtype

- 现象：`--kv-cache-dtype fp8` 启动失败：`Qwen4Exp QSA requires a BF16 main KV cache`。
- 修复：QSA 层主 KV 必须为 BF16（`--kv-cache-dtype auto` 即可），上游 NVIDIA/AMD 实现均只声明支持 `auto`/`bfloat16`。

### 4.3 `CircularBufferSpec` 注册缺失

- 现象：`ValueError: Unsupported KV cache spec type ... CircularBufferSpec. Please register it using @register_kv_cache_spec`。
- 修复：`vllm/v1/core/single_type_kv_cache_manager.py` 补齐三处：
  - 移植 `CircularBufferManager`（每请求固定 1 个 ring block，禁用 prefix cache）
  - `register_all_kvcache_specs` 注册 `CircularBufferSpec → CircularBufferManager`
  - 避免 ring block 被普通 attention cache zeroing 路径处理

### 4.4 QSA 环形 KV slot mapping 越界（长上下文崩溃根因）

- 现象：`CUDA_LAUNCH_BLOCKING=1` 下，>12K 的 chunked prefill 在第二个 chunk 触发 `Triton Error [CUDA]: an illegal memory access`，全部 worker 死亡、服务退出。栈指向 `block_table.py compute_slot_mapping`。
- 根因：backport 移植时遗漏上游 `gpu_model_runner._get_slot_mapping_mode()`。上游对 `CircularBufferSpec`（QSA 环形 raw-key KV，`max_num_blocks_per_req=1`）返回 `SlotMappingMode.NONE`；backport 内联逻辑只对 `MAMBA` 设 NONE，QSA 环形组被误归为 `TOKEN_TO_KV_SLOT`，通用 `ComputeSlotMappingKernel` 按 `pos//block_size` 索引只有 1 列的 block table → 越界读。
- 修复：移植 `_get_slot_mapping_mode()`（`MAMBA` 或 `CircularBufferSpec` → `NONE`），并补 `CircularBufferSpec` import。QSA 环形槽位本就由 QSA 自己的 metadata builder（`build_qsa_metadata`）管理，无需通用 token→slot。

### 4.5 262K 上下文容量

- 现象：`--max-model-len 262144` + `--gpu-memory-utilization 0.90` 启动失败：可用 KV 仅 1.26 GiB，低于 262K 所需的 3.22 GiB。
- 修复：`--gpu-memory-utilization 0.96`（模型权重 ~23 GiB/卡 + 激活后，KV 池 ~21 GiB/卡，可容纳 262K）。

## 5. 部署配置（验证用）

```bash
/data/models/vllm-qwen4exp-env/bin/vllm serve /data/models/Qwen3.8-Flash-Next-FP8 \
  --served-model-name qwen3.8-flash-next qwen3.8-flash-next-fp8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.96 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --api-key sk-6b1e... \
  --host 0.0.0.0 --port 18080
```

关键点：
- **必须 `--enable-expert-parallel`**（EP8：每卡 64/512 专家，专家维 640 保持完整，规避 FP8 block 整除问题）
- **QSA 要求 BF16 主 KV**：`--kv-cache-dtype auto`
- SM89 自动选择 `TRITON Fp8 MoE backend`

## 6. 长上下文性能实测

单请求 needle 召回基准（streaming、temp 0、max_tokens 128），每个档位注入唯一 marker，校验输出是否准确召回：

| Prompt tokens | TTFT | Prefill tok/s | Decode tok/s | Needle | finish |
| --- | --- | --- | --- | --- | --- |
| 7,829 | 2.1s | 3,806 | 40.1 | 命中 | stop |
| 31,336 | 8.4s | 3,725 | 40.4 | 命中 | stop |
| 62,679 | 15.7s | 3,995 | 40.1 | 命中 | stop |
| 125,366 | 21.6s | 5,803 | 74.0 | 命中 | stop |
| 191,297 | 45.4s | 4,214 | 41.4 | 命中 | stop |
| 229,558 | 54.9s | 4,180 | 41.5 | 命中 | stop |
| 250,601 | 45.3s | 5,535 | 87.6 | 命中 | stop |

注：前 5 档在 `CUDA_LAUNCH_BLOCKING=1`（同步 kernel）下测得，略慢；后两档为异步真实速度。

结论：
- **无退化**：全部档位 marker 精确召回，`finish_reason=stop`，无重复、无截断、无输出退化
- **解码稳定**：长上下文下 decode 40~88 tok/s（对比 DeepSeek-V4 SGLang 200K 仅 18 tok/s）
- **prefill 稳定**：3700~5800 tok/s，与上下文长度弱相关
- 显存全程稳定 ~47.7 GiB/卡，无泄漏

## 7. 相关脚本

- `D:\env\scripts\qwen38_context_bench.py`：分档长上下文基准（精确 token 构造 + 流式 TTFT/decode）
- `D:\env\scripts\qwen38_probe.py`：单请求二分探测
- `D:\env\scripts\serve_qwen4exp.sh`：验证服务启动脚本
