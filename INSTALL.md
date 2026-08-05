# Installation

## Binary wheel

Release `v0.26.1+cubic.20260805` provides the reference Linux x86_64 wheel
built with:

- Python 3.12
- PyTorch 2.13.0+cu132
- CUDA Toolkit 13.0.88
- NVIDIA driver compatible with CUDA 13.2

A clean uv installation was also validated with PyTorch 2.13.0+cu130. The
stable-libtorch extension is not tied to the PyTorch CUDA patch suffix used by
the build environment.

Create a clean environment and install the wheel URL shown in the GitHub
release notes:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "https://github.com/QuantTrio/vllm-cubic/releases/download/v0.26.1%2Bcubic.20260805/vllm-0.26.1%2Bcubic.20260805-cp312-cp312-linux_x86_64.whl"
vllm --version
```

## Build directly from Git

Install the tagged source revision with uv:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS="$(nproc)"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"
uv pip install "git+https://github.com/QuantTrio/vllm-cubic.git@v0.26.1+cubic.20260805"
```

The source build compiles CUDA and Rust extensions. It therefore requires a
working CUDA toolkit, C/C++ compiler, Rust toolchain, CMake and Ninja, and is
not expected to be quick. Limit `MAX_JOBS` on memory-constrained hosts.

The reference architecture list covers A100 (SM80), RTX 30-series (SM86), RTX
40-series (SM89), H100/H200 (SM90), B100/B200-class data-center Blackwell
(SM100), and RTX 50-series Blackwell (SM120). The Cubic CUDA sources receive
native cubins for every requested architecture. A local build may set a
narrower list for a smaller binary and shorter build.

## Development checkout

```bash
git clone https://github.com/QuantTrio/vllm-cubic.git
cd vllm-cubic
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

For reproducible development builds, use the exact dependency revisions in
the repository's CMake and requirements files. Generated build products,
model checkpoints, calibration caches and benchmark logs are intentionally
not committed.

## Verify Cubic support

```bash
python - <<'PY'
from vllm.model_executor.layers.quantization import get_quantization_config

config = get_quantization_config("cubic")
print(config.__name__)
PY
```

On a CUDA host, loading a Cubic checkpoint also verifies that the compiled
Cubic operators are available before the server accepts requests.
