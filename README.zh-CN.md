# VLLM Backport（中文说明）

一个专注于在 Ampere 架构上运行 DeepSeek V4 Flash 0731 的 VLLM 分支。

当前在 8xA6000（TP4PP2）上可实现 3435 tps 的 prefill 与 948 tps 的解码，理论上也适用于 A100。

> 本文件为 README 的中文版本，内容与 [README.md](README.md) 保持一致。

## 实测性能（8x RTX 4090 48GB，SM89）

验证环境：`deepseek-server`（172.18.12.5），8x NVIDIA RTX 4090 48GB（Ada / SM89），双路 Xeon Gold 6530，565 GiB 内存，CUDA 13.0，Python 3.12 venv，8 卡 TP8。

### DeepSeek-V4-Flash-0731（生产服务，`deepseek-v4-flash-0731`）

官方 FP4 checkpoint，`--kv-cache-dtype fp8_ds_mla`，TP8，524288 max-model-len，4 路并发，端口 18080。**生产环境已禁用 DSpark greedy draft**：在长上下文 / 多客户端场景下它会放大输出重复（长上下文下 draft 接受率崩塌至 0-5.6%），因此服务未带 `--speculative-config` 运行。

| 指标 | 数值 |
| --- | --- |
| 解码速度（关闭 DSpark） | ~80 tok/s |
| 峰值显存 | ~44.7 GiB / 卡 |
| 上下文 | 最高 512k（fp8_ds_mla KV） |

### Qwen3.8-Flash-Next（`qwen4_exp` 移植，`qwen3.8-flash-next`）

Qwen4Exp 125B MoE + PLE + QSA，官方 FP8 checkpoint，`--enable-expert-parallel`（TP8 + EP8，每卡 64/512 专家），BF16 KV cache（QSA 要求），原生 262144 max-model-len，chunked prefill 4096，端口 18080。单请求 needle 召回基准（流式、temp 0）：每一档都精确召回注入标记且 `finish_reason=stop`——无重复、无截断、无输出退化。

| Prompt tokens | TTFT | Prefill tok/s | Decode tok/s | 结果 |
| --- | --- | --- | --- | --- |
| 7.8k | 2.1s | 3806 | 40.1 | OK |
| 31.3k | 8.4s | 3725 | 40.4 | OK |
| 62.7k | 15.7s | 3995 | 40.1 | OK |
| 125.4k | 21.6s | 5803 | 74.0 | OK |
| 191.3k | 45.4s | 4214 | 41.4 | OK |
| 229.6k | 54.9s | 4180 | 41.5 | OK |
| 250.6k | 45.3s | 5535 | 87.6 | OK |

模型权重约 23 GiB/卡，稳态总占用约 47.7 GiB/卡。前 5 行在 `CUDA_LAUNCH_BLOCKING=1`（同步 kernel，略慢）下测得，后两行为异步真实速度。移植与排障完整记录见 [docs/qwen3.8-flash-next-adaptation.md](docs/qwen3.8-flash-next-adaptation.md)。

## Docker 用法

每次推送都会将预构建镜像发布到 Docker Hub：

| 镜像 | 目标 GPU |
| --- | --- |
| `lazymio/vllm-backport:latest-sm86`（同 `:latest`） | Ampere sm86（A6000、RTX 30xx） |
| `lazymio/vllm-backport:latest-sm80` | Ampere sm80（A100） |
| `lazymio/vllm-backport:latest-sm89` | Ada sm89（RTX 4090、L40S） |
| `lazymio/vllm-backport:v0.6.4-sm86` / `-sm80` / `-sm89` | 固定版本构建 |

镜像是单架构构建（不含 FA3/Hopper 内核），请选择与你的 GPU 匹配的标签。入口命令为 `vllm serve`。

在 sm89 上，需要在容器环境中设置 `VLLM_TEST_FORCE_FP8_MARLIN=1`。

`:latest*` 标签跟踪 main 分支；每个发布也提供 `:v0.6.1-sm86` 之类的固定版本标签。

### Docker Compose

```yaml
services:
  vllm:
    image: lazymio/vllm-backport:latest-sm86  # A100: 使用 :latest-sm80
    command: >
      deepseek-ai/DeepSeek-V4-Flash-0731
      --tensor-parallel-size 8
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}
    ipc: host
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

然后：

```bash
docker compose up -d
curl http://localhost:8000/v1/models
```

按你的环境调整模型与 `--tensor-parallel-size`；多卡张量并行必须使用 `ipc: host`。

## 推荐配置

```bash
vllm serve /path/to/your/deepseek \
  --tensor-parallel-size 8 \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8_ds_mla \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64],"max_cudagraph_capture_size":64}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --host 0.0.0.0 --port 8000 \
  --hf-overrides '{"head_dtype": "float32"}' \
  --served-model-name deepseek-v4-flash
```

配合以下环境变量（`FULL_AND_PIECEWISE` 需要，参见下面的 CUDA graph 提示）：

```bash
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
```

提示：

- `FULL_AND_PIECEWISE`（v0.6.x 默认推荐，原 `PIECEWISE`）把整个 decode 步骤——attention、MoE dispatch、NCCL all-reduce 以及 DSpark draft 循环——全部捕获进一张 CUDA graph。在 4x/8x A6000 实测：单流解码 45.6 -> 67-70 tok/s（+47%）；prefill 不变（计算密集）。两个前提：
  - 用 `NCCL_ALGO=Ring NCCL_PROTO=Simple` 固定 NCCL（并建议加 `--disable-custom-all-reduce`）。graph 重放必须重发完全相同的已捕获集合通信；NCCL 的按尺寸自适应算法切换正是 FULL 捕获在 Ampere 上"崩溃"的原因——Ampere 本身没有问题。
  - 按示例限制 `cudagraph_capture_sizes`。FULL graph 持有私有内存池，把每个 batch size 都捕获到 `--max-num-seqs` 会在高 `--gpu-memory-utilization` 下额外占用每卡 800+MB 甚至 OOM warmup。
- 请相应调整 TP（`--tensor-parallel-size`）与 PP（`--pipeline-parallel-size`）。
- `head_dtype` 覆盖有助于减少垃圾输出。
- PP>1 时 DSpark 效果不佳。
- `--speculative-config` 的 JSON 外面必须加单引号——否则 bash 会把花括号处的逗号做花括号展开，vLLM 收到字面量 `method:dspark`（报 `Value method:dspark cannot be converted`）。
- Ampere 上 `num_speculative_tokens` 固定为 5。低于 5（checkpoint 的 `dspark_block_size`）会被拒绝；7 需要约 200KB 共享内存，超过 Ampere 的 163KB 限制（`triton OutOfResources` 错误）。6 能启动，但越过原生块的 draft 位置几乎不会被接受（实测 3-13%），只会浪费 draft 计算——输出质量与速度与 5 相同。
- 请求若既不设 `thinking` 也不设 `reasoning_effort`，现在会进入高 effort 的 thinking 模式，与官方 0731 API 映射一致（`reasoning_effort: "none"` 恢复普通聊天模式）。Agent/工具调用类客户端应在会话第一轮就显式传 `reasoning_effort`——不带 effort 前缀的会话会逐渐停止思考并进入自我强化的推理循环。

## 环境变量（警告：大量 AI 生成内容！）

本分支相对原版 vLLM 增加的全部旋钮。默认值即镜像内置值；通常无需改动。

### 默认开启（正确性 / batch 不变性）

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `VLLM_DETERMINISTIC_MOE_ALIGN` | `1` | 确定性 MoE token 分组（稳定排序替代原子序）。`0` 恢复历史 CUDA kernel。 |
| `VLLM_DSV4_FIXED_DECODE_SPLITS` | `16` | 固定稀疏解码 attention 的 split-k，使请求数值不随共批内容变化。`0` 恢复 batch 自适应启发式。 |
| `VLLM_TOKEN_BUCKET_PAD` | `1` | 把 batch 填充到固定 token 桶（16/32/64/128/256，之后 ×256），使 GEMM 分块不再随精确 batch size 变化。`0` 关闭。 |
| `VLLM_DSPARK_FUSED_MARKOV` | `1` | 融合的 DSpark Markov draft 采样链。`0` 回退到 eager 算子链。 |
| `VLLM_DSV4_LOGITS_ROW_CHUNK` | `128` | 对稀疏 indexer prefill logits 做行分块，使 `[chunk_rows, context/4]` fp32 临时量在长上下文下保持有界（修复 ~134k token 以上的崩溃；256k+ 需要）。`0` 恢复整块分配；不带前缀的 `DSV4_LOGITS_ROW_CHUNK` 拼写同样有效。 |

### 可选性能旋钮（默认 `0`——请先在自己的拓扑上实测）

| 变量 | 含义 |
| --- | --- |
| `VLLM_MHC_PRENORM_SHARD` | 跨 TP rank 切分 mHC prenorm GEMM（TP8 收益明显，TP4 反而变差）。 |
| `VLLM_MHC_POST_FUSE_SQRSUM` | 把 mHC prenorm 的行 sqrsum 折叠进 `mhc_post`。 |
| `VLLM_UNREPLICATE_ATTN_GEMMS` | 去重跨 TP rank 复制的 attention GEMM。 |
| `VLLM_INDEXER_QUERY_SHARD` / `VLLM_INDEXER_QUERY_SHARD_QPATH` | 跨 TP rank 切分稀疏 indexer query 投影。 |
| `VLLM_SPARSE_RAGGED_FAST_SCAN` | 稀疏 prefill 中更快的 ragged-index 扫描。 |
| `VLLM_SPARSE_PREFILL_EXACT_TILE` | 面向精确 tile 形状的无 mask 稀疏 prefill kernel 特化。 |
| `VLLM_DSPARK_VOCAB_SHARD` | 词表分片 DSpark greedy draft 选择（减少 draft 侧通信）。 |
| `VLLM_MARLIN_FP8_DEQUANT_BF16` | 让稠密 block-fp8 GEMM 走 cuBLAS（反量化→bf16）而非 Marlin。 |
| `VLLM_HIER_ALL_REDUCE` | 面向多 PCIe island 机箱的 island 感知分层 all-reduce。 |
| `VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE` | 覆盖自定义 all-reduce 负载上限（MB）。 |
| `VLLM_MHC_FIXED_NUM_SPLIT` | 固定 mHC TileLang GEMM split-k（仅 DeepGEMM 能力的 GPU 可达；sm86/sm80 无效果）。 |

### 运维 / 调试

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `VLLM_MQ_MAX_CHUNK_BYTES_MB` | `16` | Worker 消息队列分块大小。容器 `/dev/shm` 较小且无法用 `--ipc=host` 时调小（如 `1`）。 |
| `VLLM_DISABLE_MULTI_STREAM_PARALLEL` | `0` | 调试总开关：把辅助流工作串行放到默认流上执行。 |

并发下如需严格的 temperature-0 稳定性，可考虑 `--hf-overrides '{"head_dtype": "float32"}'`（fp32 logits head）——这是 CLI 参数，不是环境变量。
