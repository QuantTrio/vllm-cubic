# vLLM Cubic

vLLM Cubic is a CUDA-oriented fork of
[vLLM](https://github.com/vllm-project/vllm) that implements the Cubic
quantization format as a native vLLM quantization method. It keeps weights in
packed 1--8-bit Cubic form and selects optimized A16 or dynamic-A8 kernels at
runtime.

The implementation is format-oriented rather than model-specific. Kimi K3 is
the first large-scale validation target and is included as a conversion and
serving example.

## Highlights

- Native packed Cubic weights from W1 through W8, with per-group FP32 scale
  and FP16 curve parameters.
- Weight-only A16 execution by default.
- Opt-in dynamic A8 execution for Cubic Linear and routed-expert layers.
- CUDA and Triton kernel families with startup calibration against the actual
  model shapes and the current GPU.
- Calibration sharing across equivalent devices and persistence in Triton's
  cache, including multi-node and heterogeneous deployments.
- Query-preserving `fp8_q16` MLA KV cache and experimental `cubic8` MLA KV
  cache.
- Chunked-prefill scheduling improvements for hybrid attention/Mamba models.
- OpenAI-compatible serving through the standard vLLM CLI and API.

This is an experimental inference project. Validate model quality and
performance for your own checkpoint and hardware before production use.

## Installation

Create an isolated Python 3.12 environment:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

For the supported binary configuration, install the wheel attached to the
GitHub release. The release notes contain the exact command and compatibility
matrix. A release wheel is the fastest installation path and does not compile
CUDA locally.

To build from source directly from a tagged revision:

```bash
uv pip install "git+https://github.com/QuantTrio/vllm-cubic.git@v0.26.1+cubic.20260805"
```

The source command requires a CUDA toolkit and a compiler toolchain and can
take substantial time. See [INSTALL.md](INSTALL.md) for prerequisites,
reproducible source-build settings, and binary compatibility details.

## Serving a Cubic checkpoint

Cubic checkpoints declare `quant_method: "cubic"` in `config.json`, so the
quantization method is normally detected without a CLI flag.

```bash
export VLLM_CUBIC_DYNAMIC_A8=1

vllm serve /path/to/cubic-model \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-chunked-prefill
```

`VLLM_CUBIC_DYNAMIC_A8=0` selects weight-only A16 execution. Dynamic A8 is an
execution choice and does not change the checkpoint packing format.

For MLA models, `fp8_q16` stores KV in FP8 while retaining query computation
in the model dtype:

```bash
vllm serve /path/to/cubic-model --kv-cache-dtype fp8_q16
```

The backend is selected automatically. FlashMLA is used where supported and
the implementation falls back to the compatible Triton MLA path otherwise.
`cubic8` is also available as an experimental MLA KV-cache dtype.

## Kernel calibration and cache

Device calibration is enabled by default. It benchmarks the Cubic kernel
families needed by the loaded checkpoint, using the real tensor shapes and
serving token buckets before CUDA graph capture. Equivalent devices sharing a
cache calibrate each unique task once; heterogeneous devices calibrate their
own applicable tasks.

The results use Triton's configured cache location. Set `TRITON_CACHE_DIR` to
choose it:

```bash
export TRITON_CACHE_DIR="$HOME/.triton/cache"
```

Removing that directory resets both Triton and Cubic tactic caches. Set
`VLLM_CUBIC_AUTOTUNE=0` only for startup diagnostics; safe fallback tactics
remain available.

## Checkpoint conversion

The generic converter and Kimi K3 validation tools are under
[`examples/quantization`](examples/quantization). The Kimi K3 data-free
quantizer is documented in that directory and supports mixed W1--W8 schedules
without changing the runtime packing format.

## Compatibility

- Cubic weight execution currently requires an NVIDIA GPU with compute
  capability 8.0 or newer. The reference CUDA wheel includes native Cubic
  kernels for SM80, SM86, SM89, SM90, SM100 and SM120.
- Dynamic A8 performance and the selected tactic depend on the GPU, tensor
  shape, group size, bit width, and batch/token bucket.
- `fp8_q16` requires reliable E4M3 support; unsupported configurations are
  rejected or routed to a compatible backend rather than silently changing
  query precision.
- Binary wheels are tied to their documented Python, PyTorch, CUDA and system
  ABI. Use a matching wheel or build from source.

## Upstream and license

This repository is based on vLLM and preserves its Apache-2.0 license. See
[NOTICE](NOTICE) for attribution and [LICENSE](LICENSE) for the complete
license text. General vLLM usage and API documentation is available at
[docs.vllm.ai](https://docs.vllm.ai/).
