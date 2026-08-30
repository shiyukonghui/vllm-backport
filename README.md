# vLLM Backport

A vLLM fork that focuses on running frontier models on older cards like A6000, 3090 and A100.

Status:

| Supported Models | Quantization | Status |
| --- | --- | --- |
| `DeepSeek-v4-Flash-0731` | Native FP4 | Fully Supported (0.6.0+) |
| `Qwen3.8-27B` | BF16, AWQ W4A16 | Fully Supported (v0.8.0+) |
| `Qwen3.8-Flash-Next` | FP8, [AWQ W4A16](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16) | Fully Supported (v0.9.0+) |
| `GLM-5.3-Flash` | [AWQ W4A16](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16) | Fully Supported (v0.11.0+) |

Note we have a paired [LMCache](https://github.com/wtdcode/LMCache/tree/vllm-backport) fork for production kvcache serving, **which is also built into our docke images.**

## Docker Usage

Prebuilt images are published to Docker Hub on every push:

| Image | Target GPUs |
| --- | --- |
| `lazymio/vllm-backport:latest-sm86` (also `:latest`) | Ampere sm86 (A6000, RTX 30xx) |
| `lazymio/vllm-backport:latest-sm80` | Ampere sm80 (A100) |
| `lazymio/vllm-backport:latest-sm89` | Ada sm89 (RTX 4090, L40S) |
| `lazymio/vllm-backport:v0.9.0-sm86` / `-sm80` / `-sm89` | pinned release builds |

Images are single-arch builds (no FA3/Hopper kernels), so pick the tag matching your GPU. The entrypoint is `vllm serve`.

`:latest*` tags track the main branch; each release also ships versioned tags like `:v0.6.1-sm86` if you want to pin.

### Docker Compose

```yaml
services:
  vllm:
    image: lazymio/vllm-backport:latest-sm86  # A100: use :latest-sm80
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

Then:

```bash
docker compose up -d
curl http://localhost:8000/v1/models
```

Adjust the model and `--tensor-parallel-size` to your setup; `ipc: host` is required for multi-GPU tensor parallelism.

## Recommend Setup

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

with these environment variables (required for `FULL_AND_PIECEWISE`, see the cudagraph tip below):

```bash
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
```

Tips:

- `FULL_AND_PIECEWISE` (v0.6.x default recommendation, previously `PIECEWISE`) captures the whole decode step — attention, MoE dispatch, NCCL all-reduce and the DSpark draft loop — into one CUDA graph. Measured on 4x/8x A6000: single-stream decode 45.6 -> 67-70 tok/s (+47%); prefill is unchanged (compute-bound). Two prerequisites:
  - Pin NCCL with `NCCL_ALGO=Ring NCCL_PROTO=Simple` (and prefer `--disable-custom-all-reduce`). Graph replay must re-issue the exact captured collective; NCCL's size-adaptive algorithm switching is what made FULL capture "crash on Ampere" — Ampere itself is fine.
  - Bound `cudagraph_capture_sizes` as shown. FULL graphs keep private memory pools; capturing every batch size up to `--max-num-seqs` can cost >800 MB per GPU and OOM warmup at high `--gpu-memory-utilization`.
- Adjust your TP (--tensor-parallel-size) and PP (--pipeline-parallel-size) accordingly.
- The single quotes around the `--speculative-config` JSON are required — without them bash brace-expands the braces at the comma and vLLM receives the literal `method:dspark` (`Value method:dspark cannot be converted` error).

### Model Specific setups

#### Qwen3.8-Flash-Next (v0.9.0+)

- `VLLM_PLE_CPU_OFFLOAD=1` keeps the 51B n-gram embedding (fp8, ~51 GiB) in pinned host RAM via a separate `PleOffloadWorker` process. Without it the TP-sharded embedding adds ~12.8 GiB per GPU and KV memory goes negative on 48 GB cards.
- `--enable-expert-parallel` is required, not optional: with plain TP the 640-wide expert intermediate becomes 160 per rank, which is not a multiple of the 128x128 fp8 block, and vLLM then forces the Triton fp8 MoE kernel (no fp8 tensor cores on sm86). With EP the experts stay whole and the Marlin W8A16 backend is used.
- AWQ W4A16 ([`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16), compressed-tensors `pack-quantized`, routed experts INT4 g128, everything else BF16): MTP speculative decoding works (the BF16 MTP draft is kept unquantized automatically). Verified on 4x A100-80GB: `VLLM_PLE_CPU_OFFLOAD=1 vllm serve wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 --tensor-parallel-size 4 --enable-expert-parallel --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' --speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. This achives up to 936 tps.

#### DeepSeek V4 Flash

- Keep `num_speculative_tokens` at 5 on Ampere. Values below 5 (the checkpoint's `dspark_block_size`) are rejected, and 7 needs ~200 KB of shared memory vs the 163 KB Ampere limit (`triton OutOfResources` error). 6 does start, but draft positions past the native block are almost never accepted (3–13% in our measurements), so it only wastes draft compute — output quality and speed are the same as 5.
- Requests that set neither `thinking` nor `reasoning_effort` now get thinking mode with high effort, matching the official 0731 API mapping (`reasoning_effort: "none"` restores plain chat mode). Agentic/tool-calling clients should pass a `reasoning_effort` explicitly from the first turn of a session — sessions that run without the effort prefix gradually stop thinking and can enter self-reinforcing reasoning loops.

