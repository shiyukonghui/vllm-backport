# Qwen3.8-Flash-Next master 分支部署与验证记录（2026-08-29）

本文档记录将 Qwen3.8-Flash-Next（`qwen4_exp`）从 `port/qwen4-exp` 自移植分支切换到 **master 官方实现** 的部署过程与验证结果。自移植（port 分支）的记录见 `port/qwen4-exp` 分支的 `docs/qwen3.8-flash-next-adaptation.md`。

## 1. 为什么切换到 master

- master 官方主干已合并完整的 `qwen4_exp` 实现（`vllm/models/qwen4_exp/`、`vllm/v1/ple_offload/`、`tests/models/qwen4_exp/` 等），且**已内置自移植分支的全部关键修复**：
  - `gpu_model_runner._get_slot_mapping_mode()`（QSA 环形 KV → `SlotMappingMode.NONE`，修复 >12K chunked prefill 越界）
  - `CircularBufferSpec` + `CircularBufferManager` + registry 注册 + ring block 不零化
  - FP8 MoE block-scale refinement（`refine_fp8_moe_block_shape`，master 实现强制 Triton backend）
- master 独有而 port 没有的特性：**PLE CPU offload**（51B n-gram embedding 卸载到主机 RAM）、更新的基础库与更全的测试。
- 自移植分支独有的差异仅为 vllm-qwen38next 的 fused-shared-expert `enabled` 参数（性能优化，非正确性必需）。

## 2. 部署切换步骤

```
1. 服务器停 qwen4exp 服务
2. 备份旧源码：vllm-qwen4exp-src -> vllm-qwen4exp-src.port.bak（682M，可随时回滚）
3. 本地 git archive master -> tar.gz，scp 上传解压到 /data/models/vllm-qwen4exp-src
4. 用 setup_qwen4exp_env.sh 重建 editable + 编译 C 扩展（venv /data/models/vllm-qwen4exp-env 复用）
```

环境：8x RTX 4090 48GB（SM89）、双路 Xeon Gold 6530（128 逻辑核）、CUDA 13.0、Python 3.12 venv、VLLM_VERSION_OVERRIDE=0.5.2、TORCH_CUDA_ARCH_LIST=8.9。

## 3. 编译线程问题（重要踩坑）

- **现象**：`setup.py` 打印 `Using MAX_JOBS=3`，但 ninja 实际 `-j 1`，183 个目标单线程串行编译（约 1-1.25 分钟/目标，全程需 2.5h+）。
- **根因**：vLLM `setup.py::compute_num_jobs()` 中
  `num_jobs = max(1, MAX_JOBS // NVCC_THREADS)`。
  原脚本 `MAX_JOBS=3` + `NVCC_THREADS=2` → `3 // 2 = 1`。
- **修复**：`MAX_JOBS=128` + `NVCC_THREADS=2` → `128 // 2 = 64` 个 ninja job × 每个 nvcc 2 线程 = **128 线程**。183 个目标并行编译约 **20 分钟**完成。
- **注意**：手动续跑 ninja 不可行——build-temp 由 uv 临时目录管理，进程退出即被清理；且 ninja 的 regen 规则引用原临时路径 + 需要 `VLLM_PYTHON_EXECUTABLE` 等 setup 注入的环境。全量重编（并行）是最省事路径。
- 经验：改并行度后**务必确认实际 ninja 命令行**（`pgrep -af ninja` 看 `-j N`），不要只看日志的 `Using MAX_JOBS`。

## 4. 服务配置（master 定稿）

```bash
vllm serve /data/models/Qwen3.8-Flash-Next-FP8 \
  --served-model-name qwen3.8-flash-next qwen3.8-flash-next-fp8 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --dtype bfloat16 --kv-cache-dtype auto \
  --max-model-len 524288 --gpu-memory-utilization 0.96 \
  --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --hf-overrides '{"rope_scaling":{"type":"yarn","factor":2.0,"original_max_position_embeddings":262144}}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --api-key sk-6b1e... --host 0.0.0.0 --port 18080
```

关键点：

- **`VLLM_PLE_CPU_OFFLOAD=1`**：51B n-gram embedding（fp8 ~51GiB）经独立 `PleOffloadWorker` 驻留主机 pinned RAM。收益：权重占用 23 → **18.07 GiB/卡**，KV 池 21 → **26.63 GiB/卡**（容量 1,097,682 → **2,158,253 tokens**）。
- **500k 上下文**：原生 `max_position_embeddings=262144`，需 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` + YaRN override（`rope_scaling type=yarn factor=2.0 original_max_position_embeddings=262144`）。不加 `VLLM_ALLOW_LONG_MAX_MODEL_LEN` 会被 ModelConfig 直接拒绝（524288 > 262144）。
- **工具调用**：vLLM parser 列表**没有 `qwen3`**；Qwen3 系工具为 XML 格式，需 `--tool-call-parser qwen3_xml --enable-auto-tool-choice`。不启用时带 `tools` 的请求直接 400。

## 5. 验证结果

### 5.1 500k 上下文 agent 场景（456,250 prompt tokens + 工具定义）

| 指标 | 结果 |
| --- | --- |
| prompt tokens | 456,250（目标 480k） |
| TTFT | 98.7s（prefill 4623 tok/s） |
| 解码速度 | **67.9 tok/s** |
| 退化检查 | 无崩溃、无重复循环、无截断、无短输出 |
| 工具调用 | 场景内未触发（任务未给城市，模型合理拒绝）；单独验证 `get_weather(city=北京)` → `finish_reason=tool_calls` ✓ |

解码速度随上下文：短 74-87 tok/s → 262K 71.7 → **500k 67.9 tok/s**，长上下文仅轻微衰减。

### 5.2 多模态（Qwen4ExpForConditionalGeneration 视觉 tower）

| 图片 | 模型识别结果 |
| --- | --- |
| cat.jpeg | 戴墨镜、穿粉色连帽皮衣的猫，粉色背景 ✓ |
| dog.jpeg | 穿橙色连帽衫、戴蓝色墨镜的灰狗，蓝色背景 ✓ |
| hato.jpg | 鸽子在人行道，背景砖墙/树木/行人 ✓ |
| image1.png | OCR "Hello, AI world!" ✓ |
| image2.png | OCR "But, Safe is important!" ✓ |

视觉 tower 完整可用（含 OCR）。测试图片经 OpenAI `data:` URI（base64）传入。

### 5.3 回归对照

- 262K 档：needle 命中、finish=stop、prefill 5553 tok/s、decode 71.7 tok/s（与 port 版相当）
- 8K~256K 各档此前在 port 分支全通过；master 为同一模型权重，未再逐档重测

## 6. 环境与回滚

- 服务器：`/data/models/vllm-qwen4exp-src`（master）、`/data/models/vllm-qwen4exp-src.port.bak`（port 备份，682M）、venv `/data/models/vllm-qwen4exp-env`
- 回滚：把 `.port.bak` 换回 `vllm-qwen4exp-src`，重跑 `setup_qwen4exp_env.sh` 即可
- 构建脚本：`D:\env\scripts\setup_qwen4exp_env.sh`（已改 MAX_JOBS=128）、`D:\env\scripts\serve_qwen4exp.sh`
- 测试脚本：`D:\env\scripts\qwen38_agent500k.py`（500k agent+测速）、`qwen38_multimodal.py`（多模态）、`qwen38_context_bench.py`（分档基准）、`qwen38_probe.py`（二分探测）
