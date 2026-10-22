# pi0.5 LIBERO on Atlas

This port uses the Atlas GLM integration branch (`34399129`, including the
deployed `5d652f8e` implementation). The engine is Rust/CUDA and uses Atlas's
`GpuBackend`, resident weights, and cuBLASLt. It does not launch Python,
OpenPi, ONNX Runtime, or TensorRT during inference.

## Supported profile

The explicit profile is `crates/spark-pi05/pi05_libero.json`: batch one,
two 224×224 RGB images, a PaliGemma 2B prefix, Gemma 300M action expert,
ten denoising steps, and ten actions with seven LIBERO output dimensions.
The checkpoint stores BF16 weights. Patch embeddings, action/time
projections, adaptive normalization, and Euler integration use FP32 as in
OpenPi. An explicit [NVFP4 language-prefix recipe](PI05_NVFP4.md) is also
supported, with an optional [FP8 expert overlay and faster attention](PI05_PERFORMANCE.md).
Other pi0/pi0.5 profiles and undeclared quantization recipes are rejected.

The state vector is required by the observation contract but is not consumed
by this checkpoint, which sets `discrete_state_input=false`. The masked third
camera and prompt padding are omitted from computation. Prefix KV is computed
once per observation; each denoising step replaces its suffix KV. Prefix KV
does not persist across observations.

## Build and assets

Build on a DGX Spark with Rust from `rust-toolchain.toml` and CUDA 13:

```sh
CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=13000 \
  ATLAS_SKIP_BUILD=1 ATLAS_PI05_BUILD=1 \
  cargo build --locked --release -p spark-server
```

This is a **policy-only build**: `ATLAS_SKIP_BUILD` skips the existing chat
kernel matrix, while `ATLAS_PI05_BUILD=1` builds the native policy kernels.
For a binary with both policy and chat kernels, use the normal Atlas build
with the desired `ATLAS_TARGET_*` chat target and omit `ATLAS_SKIP_BUILD`.
The policy kernel module is compiled by `spark-pi05/build.rs`; it is separate
from the autoregressive chat model registry.

Required assets are the converted OpenPi `pi05_libero` BF16
`model.safetensors`, its `assets/physical-intelligence/libero/norm_stats.json`,
the original `big_vision/paligemma_tokenizer.model`, and the Atlas profile JSON.
The tested OpenPi source is `15a9616a00943ada6c20a0f158e3adb39df2ccac`
with the Jetson AI Lab deployment overlay. Weight conversion and reference
evaluation are offline Python harness operations; the native runtime has no
Python dependency.

## Serve actions

```sh
./target/release/spark policy serve \
  --checkpoint /models/pi05_libero_pytorch/model.safetensors \
  --config crates/spark-pi05/pi05_libero.json \
  --tokenizer /models/paligemma_tokenizer.model \
  --norm-stats /models/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json \
  --gpu-ordinal 0 --host 127.0.0.1 --port 8000 \
  --queue-capacity 4 --max-body-bytes 4194304
```

`GET /healthz` reports readiness after model loading. `POST /v1/actions`
accepts JSON with these required fields:

| Field | Representation |
| --- | --- |
| `observation/image` | 150528 RGB bytes, flattened HWC |
| `observation/wrist_image` | Same layout for the wrist camera |
| `observation/state` | Eight finite numbers |
| `prompt` | Task instruction |
| `noise` | 320 finite numbers, flattened `[10,32]` initial Gaussian sample |

Explicit noise makes repeated and cross-backend comparisons reproducible.
Responses contain `actions` shaped `[10,7]`, `timing_ms`, and
`backend="atlas-native-bf16"`. Timing includes preprocessing, synchronized
GPU inference, and action denormalization. One GPU worker serializes requests;
an occupied queue returns HTTP 429. Invalid inputs return 400, oversized
bodies 413, and an unavailable worker 503.

This endpoint is an Atlas JSON action API, not the OpenPi MessagePack
WebSocket protocol. A robot client must map observations and actions to this
contract. No robot is controlled by the benchmark harness.

For one request, use `spark policy infer` with the same model arguments,
`--input request.json --max-body-bytes 4194304`. JSON goes to stdout and
diagnostics go to stderr.

## Validation

CPU preprocessing tests can run without CUDA:

```sh
cargo test -p spark-pi05 --no-default-features
```

On a Spark, build/run the `policy_bench` and `policy_trace` examples.
`tests/pi05/export_reference.py` creates fixed-input/fixed-noise OpenPi
references. `tests/pi05/ops_parity.py` checks CUDA primitives against PyTorch.
Stage tracing synchronizes and downloads tensors; never use it for latency
measurements. Benchmark loading, warmup, synchronized local inference, and
HTTP round trips separately. Numerical agreement is not robot task success.

## Measured on the two Sparks (2026-10-22)

Both nodes run the same native binary on GB10 (SM 12.1), driver 580.126.09,
CUDA 13.0, and cuBLASLt 13.1. The original synthetic observation and initial
noise are reused for this comparison. Times are milliseconds per ten-action
chunk, not per individual action:

| Backend / measurement | gx10-4386 mean | gx10-4386 p95 | gx10-ec6d mean | gx10-ec6d p95 |
| --- | ---: | ---: | ---: | ---: |
| Earlier OpenPi compiled BF16, local | 145.92 | 146.96 | 145.36 | 146.62 |
| Atlas native BF16, local | 110.94 | 112.29 | 111.37 | 112.79 |
| Earlier TensorRT FP8 + NVFP4, local | 59.82 | 60.45 | 60.05 | 60.58 |

Native BF16 reduces latency by about 24% against the earlier BF16 baseline.
It does not match the quantized TensorRT result. The tutorial's approximately
[49 ms result is for FP8 + NVFP4 on Jetson AGX Thor](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/#performance),
not a BF16 or DGX Spark measurement.

The native run cycles seven fixed observations/noise pairs, with 21 warmup
calls and 1,400 measured calls per node (200 per case). Across all seven,
means are 111.74 / 112.11 ms and p95 values are 114.13 / 114.36 ms.
Loading takes 78 / 88 seconds from the shared checkpoint mount and is excluded.
CUDA graphs retain up to eight prefix shapes; every request uploads fresh
images, tokens, and noise. Graph capture and compilation are excluded by warmup.

All seven paired action comparisons pass cosine ≥0.999 and maximum absolute
error ≤0.02; the worst observed cosine is 0.9999645 and maximum error 0.01356.
Native actions match exactly between nodes. Three observations come from the
LIBERO calibration subset; the other four are synthetic. These establish
numerical agreement, not held-out task accuracy or robot success.
The 23 CUDA primitive cases pass, including BF16-rounded rotary frequencies
through position 703. Eight cuBLASLt bias/no-bias cases match PyTorch exactly.

For the live JSON API, 20 warmup plus 100 sequential measured requests per
node give 129.74 / 129.80 ms mean round-trip latency from the management host.
Server-reported inference means are 110.61 / 111.22 ms. The request is
1,080,945 bytes; the client opens a new HTTP connection each time. All responses
match the local native actions exactly. This protocol differs from the earlier
OpenPi MessagePack WebSocket benchmark, so their network overheads are not
directly comparable. Reproduce with `tests/pi05/benchmark_http.py` and an
explicit request, native benchmark reference, warmup count, and run count.

## Deployment and review notes

The working branch is `feat/pi05-native`, based on the GLM integration above.
The tested implementation is an uncommitted worktree; the external benchmark
bundle records source hashes rather than claiming a clean commit benchmark.
Both Sparks run `atlas-pi05-native.service` as a transient user unit, exposing
`/healthz` and `/v1/actions` on `192.168.8.181:8000` and `192.168.8.187:8000`.
Each has an independent policy replica; inference does not span the two GPUs.
GLM and its memory guards are stopped. The launch is not enabled across reboot.

Local operational assets and raw results are in the sibling
`../pi05-spark-benchmark/` directory: `launch-native.sh`, `restore-glm.sh`,
`cache/results/native-final-*.json`, `native-http-*.json`, and
`native-summary.json`. `restore-glm.sh` stops native policy services before
restoring the original GLM services. Model loading requires the shared asset
mount to remain available.

The shared-runtime change adds a BF16 cuBLASLt fused-bias entry point. The
existing no-bias API uses the same implementation with no epilogue; its GPU
oracle passes. The policy has its own CUDA module and does not alter chat
model selection. The offline checkpoint conversion saves an unused randomly
initialized expert LM head, so the restored whole-file hash differs from the
old conversion. Atlas skips that tensor; current native/reference comparisons
use the same restored checkpoint. Restoration hashes are recorded separately.

Validation completed: workspace Clippy; workspace tests (7,630 passed, zero
failed, 222 ignored, including doctests); Rust formatting; policy/server
rustdoc; license headers; and kernel shadow checks. CPU workspace tests use
the repository's fail-fast GPU stub libraries. `NO_COLOR` must be unset for
the existing terminal color assertions; the initial three failures under
`NO_COLOR=1` pass with that variable unset and no source change. Real GPU
policy tests, primitive comparisons, paired model outputs, and HTTP checks
run separately on the Sparks. This work does not rerun the full chat-model
serve matrix or validate robot control.
