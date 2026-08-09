# Cubic checkpoint tools

`quantize_k3.py` is the validated data-free Kimi K3 conversion pipeline. It
reads the public MXFP4 checkpoint, fits the configured per-group Cubic curves,
packs the result without intermediate model copies, writes sharded
safetensors, and audits the complete output before publishing it atomically.

Install the converter dependencies in an environment with CUDA-enabled
PyTorch:

```bash
uv pip install torch safetensors
```

Inspect the default mixed-precision schedule without writing a checkpoint:

```bash
python examples/quantization/quantize_k3.py \
  --source /path/to/Kimi-K3 \
  --output /path/to/Kimi-K3-Cubic-2.5Bit \
  --plan
```

Run conversion on the desired worker GPUs:

```bash
python -u examples/quantization/quantize_k3.py \
  --source /path/to/Kimi-K3 \
  --output /path/to/Kimi-K3-Cubic-2.5Bit \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7
```

Dynamic-A8 carrier correction is enabled by default and incorporates the
`round(127*q)/127` carrier grid into the data-free fitting objective. Pass
`--disable-a8-correction` only when producing a checkpoint intended solely for
continuous A16 reconstruction comparisons.

The converter does not overwrite an existing output. It writes to an
`.incomplete` directory, produces one `cubic_quantization_report.json`
containing the manifest, per-bit/per-group loss statistics and audit result,
then atomically renames the directory after the audit succeeds.

The other scripts in this directory provide lower-level conversion,
checkpoint auditing, and offline/online validation helpers.
