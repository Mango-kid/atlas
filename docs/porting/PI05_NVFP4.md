# Native NVFP4 pi0.5

This extends the [native pi0.5 port](PI05_NATIVE.md) on the same Atlas GLM
integration branch. Inference uses Rust/CUDA and cuBLASLt FP4 Tensor Core
matrix multiplication. Python and ModelOpt are used offline for calibration
and reference validation; no TensorRT engine is loaded by Atlas.

The subsequent [performance work](PI05_PERFORMANCE.md) adds an optional FP8
expert overlay, batches camera affine operations, and accelerates BF16
attention. The measurements below preserve the original NVFP4-only baseline.

## Precision recipe

The supported recipe is `nvfp4-prefix-bf16` for `pi05_libero`, batch one,
ten denoising steps. All 126 linear weight matrices in the 18-layer PaliGemma
language prefix use packed E2M1 weights and E2M1 input activations. Each
16-element block has an E4M3 scale; each tensor has a calibrated FP32 global
scale. Accumulation is FP32 and linear outputs are BF16. Every inference
recomputes activation block scales and packed values, including graph replay.

The vision encoder, multimodal projector, action expert, and attention remain
on the validated BF16 path. Patch/action/time projections, adaptive
normalization, and Euler integration retain their existing FP32 behavior.
This is the tutorial's NVFP4 language-prefix region, with the surrounding
native BF16 implementation. It is not the tutorial's complete FP8 + NVFP4
recipe. Other precision maps are rejected rather than silently substituted.

This is an **opt-in experimental quantization profile**. On the three tested
real LIBERO observations, native NVFP4 action cosines versus native BF16 are
0.99984–0.99997. Those observations overlap calibration data. Random-image
synthetic observations can change gripper modes, including sign flips; the
ModelOpt fake-quantized reference also exhibits this sensitivity. Increasing
activation headroom removes clipping but does not resolve it, so no headroom
variant is promoted. BF16 remains the CLI default. Neither calibration-set
agreement nor primitive arithmetic tests establish robot task success or
held-out generalization.

Packed weights remain packed on the GPU. The native GEMM uses
`CUDA_R_4F_E2M1` and `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`, with
NVIDIA's 128-row × four-block scale layout. See the
[cuBLAS block-scaling documentation](https://docs.nvidia.com/cuda/archive/13.0.3/cublas/index.html).
The existing BF16 GEMM path is preserved.

## Build and select

Use the existing policy-only build:

```sh
CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=13000 \
  ATLAS_SKIP_BUILD=1 ATLAS_PI05_BUILD=1 \
  cargo build --locked --release -p spark-server -p spark-pi05 \
    --no-default-features --features cuda --bin spark --examples
```

Add the explicit quantization manifest to either `spark policy serve` or
`spark policy infer`:

```sh
./target/release/spark policy serve \
  --checkpoint /models/pi05_libero_pytorch/model.safetensors \
  --config crates/spark-pi05/pi05_libero.json \
  --tokenizer /models/paligemma_tokenizer.model \
  --norm-stats /models/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json \
  --nvfp4-manifest /models/pi05_libero_native_nvfp4/manifest.json \
  --gpu-ordinal 0 --host 127.0.0.1 --port 8000 \
  --queue-capacity 4 --max-body-bytes 4194304
```

Omitting the manifest selects the existing BF16 implementation. An invalid
manifest fails loading. Successful NVFP4 responses and health checks identify
the backend as `atlas-native-nvfp4`. The observation/noise/action JSON contract
is unchanged. Benchmark and trace examples accept the same optional trailing
`--nvfp4-manifest PATH` argument.

## Checkpoint and calibration contract

The original converted BF16 checkpoint supplies retained tensors. The
version-one manifest names a relative packed payload, the exact 126 quantized
weight names and shapes, offsets, FP32 global scales, and calibration
provenance. Each packed tensor and scale tensor has a SHA-256 digest. All
consumed retained BF16 tensors are also hashed before promotion or upload.
The original whole-checkpoint digest is provenance; loading verifies the
consumed tensor digests rather than rereading the unused BF16 prefix weights.
Missing/extra tensors, incompatible shapes, overlapping/truncated payloads,
invalid scales, and digest mismatches fail loading.

`tests/pi05/export_nvfp4.py` uses the pinned OpenPi reference environment and
ModelOpt max calibration over the retained 32 real LIBERO observations. During
calibration, quantizers collect statistics with quantization disabled. Static
global activation scales are calibrated `amax/(6*448)`; block scales remain
dynamic. The exporter verifies packed-weight dequantization against ModelOpt,
creates paired fixed-noise action references and affine fixtures, and publishes
the manifest only after those checks finish.

ModelOpt's Triton fake-quantizer uses approximate FP32 division; its separate
packed-tensor API uses precise division. At FP4 midpoints these can select
adjacent codes. The exporter recovers block scales and codes from the calibrated
fake-quantized weights, retaining original scales for all-zero blocks. It requires exact BF16
reconstruction of every weight; any change from the packed API must be an
adjacent code at a midpoint within one FP32 ULP. This discrepancy and the
encoding method are recorded in the calibration artifact. Native activation
quantization uses precise division; the full-model reference check measures
any resulting boundary differences.

`tests/pi05/export_nvfp4_ops.py` provides an independent PyTorch E2M1 codec and
FP32 dequantized GEMM oracle. `nvfp4_check` exercises the actual Rust/CUDA path,
including ragged prompt lengths, maximum prefix length, padded scale storage,
zero/tiny blocks, optional bias, and CUDA graph execution. Numerical action
agreement and calibration-set comparisons do not establish robot task success.

`tests/pi05/export_nvfp4_precise_reference.py` separately decodes the actual
packed checkpoint and applies FP32 GEMMs before global scaling and BF16 output
conversion. Set `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` before importing PyTorch;
otherwise the container's override can defeat `allow_tf32=False`. The final
fourteen-GEMM audit uses this setting and passes, with maximum mean absolute
error 7.15e-8 and remaining differences within one BF16 ULP. Activation codes
and block scales match the independent codec byte for byte, and graph replay
uses changed observations correctly.

Full-model agreement is narrower: the three real observations have cosine
0.999955–0.999968 against the precise arithmetic reference, but the
`synthetic_7` case changes gripper sign. All seven cases do **not** pass the strict
action agreement gate. BF16 fake quantization introduces additional rounding
that can move these discontinuities; it is not an exact hardware FP4 oracle.
Changing calibration headroom does not fix the observed synthetic sensitivity.
Promotion to the default requires held-out real-observation agreement and
robot-task evaluation; neither has been established here.

## Validation and reproducibility

Both Sparks ran 21 warmups followed by 1,400 calls, equally distributed across
seven fixed observation/noise pairs. These are synchronized preprocessing,
GPU inference, and action postprocessing timings; loading and HTTP are excluded.

| Spark | Native BF16 mean | Native NVFP4 mean | NVFP4 p95 | Latency reduction |
|---|---:|---:|---:|---:|
| gx10-4386 | 111.74 ms | 97.95 ms | 99.37 ms | 12.34% |
| gx10-ec6d | 112.11 ms | 98.62 ms | 100.10 ms | 12.03% |

Actions match exactly between nodes on all seven fixtures. Separate HTTP
checks used 20 warmups and 100 measured requests per node, with exact action
agreement against the native outputs. Mean round-trip latency was 114.12 ms
on gx10-4386 and 117.14 ms on gx10-ec6d. Each serving process used 4,290 MiB
of GPU memory, compared with 6,998 MiB for the previous BF16 deployment.

This does not reach the tutorial's 49 ms Thor FP8 + NVFP4 figure. The earlier
TensorRT full FP8 + NVFP4 recipe on these Sparks measured about 60 ms on the
original synthetic fixture; it is a different precision recipe and runtime.

The 2026-10-22 build passed 7,639 workspace tests (222 ignored), workspace
Clippy, rustdoc, formatting, typo and license checks. The unquantized path produces
exactly the previous BF16 actions on all seven fixtures. Packed weight
reconstruction matches ModelOpt for all 1,981,808,640 quantized weight values.
The three actual-model affine checks have cosine at least 0.999992 against
ModelOpt's BF16 fake-quantized outputs; these are separate from full-model
action agreement above.

Results and checkpoint artifacts are under the sibling
`pi05-spark-benchmark/cache/` directory. `native-nvfp4-summary.json` in
`cache/results/` records timings, hashes, quality limitations, and raw report
paths. The experiment root contains `native-nvfp4-source-snapshot.tar.gz`,
`native-nvfp4-source-manifest.json`, and `native-nvfp4-tracked-changes.patch`.
These identify an **uncommitted** `feat/pi05-native` worktree based on
`34399129d815844c503402645b4ce0b045b2a88f`; the base commit alone does not
identify the benchmarked implementation. Previous BF16 recovery artifacts
remain separate.

The test service is `atlas-pi05-native.service` on each Spark. Start it with
`launch-native-nvfp4.sh <Spark LAN IP>` from the experiment directory. Stop the
service before switching to `launch-native.sh` for BF16, or run
`restore-glm.sh` to restore the prior GLM services. Test services are transient
and are not enabled across reboot; GLM boot enablement is unchanged.
