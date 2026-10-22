# Native pi0.5 performance

This extends the [NVFP4 port](PI05_NVFP4.md) in the same Atlas GLM integration
worktree. The optimization target is the approximately 60 ms TensorRT
FP8 + NVFP4 result previously measured on these DGX Sparks, with both images,
ten denoising steps, and the full ten-action output preserved.

## What NVIDIA does

The [NVIDIA tutorial](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)
uses FP8 vision and action-expert linears, NVFP4 language-prefix linears,
quantized attention, and TensorRT compilation. Its export also supports
batched vision views, fused expert QKV and gate/up projections, and folding
fixed time-conditioning computations. Atlas already folds time-conditioning,
reuses prefix KV across all denoising steps, skips unused final-prefix output,
and captures CUDA graphs. Quantizing the prefix alone left the dominant
vision, expert, and attention costs in place.

An Nsight Systems trace of three warmed native NVFP4 graphs on gx10-ec6d
measured approximately 99 ms per inference:

| GPU stage | Time |
|---|---:|
| Vision | 24.32 ms |
| Language prefix | 24.35 ms |
| Ten expert steps | 50.25 ms |

Across those stages, expert BF16 GEMMs consumed 32.82 ms, vision BF16 GEMMs
17.87 ms, and attention kernels 24.83 ms. The trace and kernel totals are saved
under `pi05-spark-benchmark/cache/results/native-nvfp4-baseline-*`.

## Changes

Camera affine operations now batch the two views as 512 rows. Attention is
still independent within each 256-token camera view. This alone measured
87.07 ms over 140 calls versus 97.95 ms for the previous implementation;
all seven fixed-input action outputs were identical.

Compatible Q/K/V and gate/up projections share one quantized activation.
Reuse requires identical input dimensions, precision, and activation scale;
each GEMM retains its individual weight scale, bias, and output shape.
Incompatible groups retain separate quantization. This removes 52 NVFP4
quantizations per inference, plus 54 vision and 540 expert FP8 quantizations
when their respective overlays are enabled.

FP8 overlays contain row-major E4M3 weights and calibrated scalar weight and
activation dequantization scales. Inference quantizes fresh BF16 activations,
uses native FP8 Tensor Core GEMMs with FP32 accumulation, applies the scale
product and bias, and rounds the output to BF16. The overlays support 126 expert matrices, 163 vision/projector matrices, or
both regions. The selected deployment uses only the expert overlay: adding
FP8 vision yielded less than 1 ms extra speedup and introduced an additional
synthetic gripper-mode mismatch in the tested configuration.
Sensitive FP32 projections remain unchanged.

An overlay is bound to the exact NVFP4 manifest and selected original tensor
hashes. Payload offsets, inventories, shapes, finite scales, finite E4M3
values, and SHA-256 hashes are validated before use. Calibration uses the
same 32 real LIBERO observations with quantization disabled. This does not
establish held-out robot task quality.

Select an overlay explicitly alongside the prefix manifest:

```sh
spark policy serve <existing required policy arguments> \
  --nvfp4-manifest /models/pi05_libero_native_nvfp4/manifest.json \
  --fp8-manifest /models/pi05_libero_native_fp8/expert/manifest.json
```

The backend identifies successful mixed-precision loading as
`atlas-native-fp8-nvfp4`. Omitting the FP8 overlay retains the existing NVFP4
recipe; omitting both manifests selects BF16. A missing, mismatched, or invalid
overlay fails explicitly.

Prefix and expert attention now flatten the eight query heads against their
shared K/V head into two cuBLAS BF16 GEMMs. K/V are padded to a multiple of
16; padding is excluded before softmax. Dot-product rounding, scale, and
probabilities retain BF16 semantics. The vision and explicitly masked paths
retain the previous implementation. Twenty-four GPU shape/value comparisons
passed; prefix cases were bit-identical and expert differences were within
one BF16 ULP. The initial full-model run measured 55.43 ms, versus 70.81 ms
with the old attention path and the same expert-only FP8 recipe.

## Numerical reference details

FP8 activation encoding uses precise FP32 division by its calibrated scale
before E4M3 conversion. A Python scalar divisor in PyTorch can instead become
reciprocal multiplication and choose a different E4M3 code at an exact
midpoint. The independent oracle therefore uses a CUDA FP32 tensor divisor.
Previous scalar-division reports remain archived separately; neither the
native codec nor previously published packed weight bytes were changed.

The earlier NVFP4-only profile already exhibited synthetic gripper-mode
sensitivity. That limitation is evaluated separately from primitive arithmetic
and latency; performance improvements do not establish robot-task success.

## Initial two-Spark result, 2026-10-22

Both independent replicas run identical binaries and the expert-only FP8
overlay. Each run uses 21 warmups and 1,400 synchronized calls distributed
equally across seven fixed observations/noise inputs. Loading and graph
capture are excluded; preprocessing, GPU execution, and output decoding are
included. Both cameras, all ten denoising steps, and all ten actions remain.

| DGX Spark | Previous NVFP4 mean | New mean | New p95 | Latency reduction |
|---|---:|---:|---:|---:|
| gx10-4386 | 97.95 ms | 55.53 ms | 56.29 ms | 43.30% |
| gx10-ec6d | 98.62 ms | 55.68 ms | 56.44 ms | 43.54% |

On the original synthetic input used for the prior TensorRT measurement,
Atlas averages 55.27/55.38 ms versus the recorded TensorRT FP8 + NVFP4
59.82/60.05 ms on the respective Sparks. These are local pipeline timings,
not network round trips or the tutorial's separate Thor measurement. The
seven final action outputs are identical across the two Sparks.

The warmed Nsight kernel totals on the same real observation explain the
improvement (three graph replays, distinct from wall-clock measurements):

| GPU stage | Before | After |
|---|---:|---:|
| Vision | 24.32 ms | 13.79 ms |
| Language prefix | 24.35 ms | 18.93 ms |
| Ten expert steps | 50.25 ms | 22.73 ms |

The selected recipe removes 592 repeated activation-quantization launches per
inference. Expert attention, including K/V padding, falls from 14.09 ms to
2.37 ms; expert dense GEMMs fall from 32.82 ms to 16.61 ms.

## Accuracy and validation

Sixteen additional LIBERO observations come from distinct episodes excluded
from all 32 calibration episodes, spanning 13 instructions. Selection and
source hashes are recorded against dataset revision
`a4336d589d589045d1c56423ffdf3b88a0e19b1f`. Actual native HTTP outputs versus
independent BF16 references have minimum full-action cosine 0.999829,
minimum arm-only cosine 0.998800, and maximum absolute difference 0.04137.
There are zero gripper sign changes across their 160 action steps. Against
the independent mixed-precision reference, minimum cosine is 0.999934 and
maximum absolute difference is 0.02212, also with zero sign changes.

The original random-image `synthetic_7` fixture still has one gripper sign
change versus BF16 and the independent quantized reference. The strict
seven-fixture action-agreement gate therefore remains unmet. The accelerated
quantized profile stays opt-in; BF16 remains the CLI default. These checks
measure numerical behavior, not closed-loop robot task success.

The unquantized BF16 regression passes the existing seven-fixture gate
(cosine at least 0.999 and maximum absolute difference at most 0.02).
GPU validation includes 24 FP8 GEMM cases, 16 calibrated affine checks,
24 attention shape/value cases, exact quantization bytes, and graph replay
with changed inputs. Workspace validation passes 7,648 tests, with 222
ignored; formatting, clippy, rustdoc, mdBook, license headers, typos, and
kernel-shadow checks pass. All 51 changed/new Rust files meet the 500-line
cap. CI GPU stubs include fail-fast profiler symbols for the profiling example.

Results live under `pi05-spark-benchmark/cache/results/native-fast-*`, with
held-out requests, paired references, and HTTP outputs under
`native-fp8-heldout16/`. `native-fast-summary.json` records final HTTP latency,
binary/manifest hashes, accuracy, and validation. The source manifest,
recovery archive, and tracked patch are `native-fast-source-*` and
`native-fast-tracked-changes.patch` in the experiment root. This is the
`feat/pi05-native` implementation committed as `28b24c60`; the archive also
preserves the source used for these measurements.

Both port-8000 services use `launch-native-fast.sh` and report
`atlas-native-fp8-nvfp4`. They are transient user services. The prior NVFP4
and BF16 binaries, snapshots, launch scripts, and GLM restoration script
remain available; GLM services are stopped and their boot enablement is
unchanged.

## Exact-output optimization follow-up

The follow-up keeps the same checkpoint, quantization manifests, two cameras,
ten denoising steps, and ten decoded actions. It changes native GPU execution;
HTTP transport is outside the measurement.

GELU and activation packing now share a kernel for NVFP4 prefix and FP8 expert
projections. Expert adaptive normalization also packs FP8 activations directly.
Both fusions retain the original intermediate BF16 rounding and quantization
scales. Vision attention uses a larger output tile and a warp softmax with the
same sum reduction order. Owned, padded K/V caches eliminate repeated copies;
cached rotary tables combine query/key rotation and clear both cache tails on
every graph replay. Generic attention and incompatible projection groups retain
their existing paths.

Fresh paired runs use 21 warmups and 1,400 calls per Spark, equally distributed
across seven fixed inputs. Timing includes preprocessing, synchronized GPU
execution, and action decoding; loading and graph capture are excluded.

| DGX Spark | Fresh baseline mean | Optimized mean | Optimized p95 | Latency reduction |
|---|---:|---:|---:|---:|
| gx10-4386 | 55.64 ms | 49.41 ms | 50.19 ms | 11.20% |
| gx10-ec6d | 55.56 ms | 49.63 ms | 50.37 ms | 10.67% |

Both also exceed a 10% reduction against the previously recorded 55.53/55.68 ms
baselines. Identical binaries run on both Sparks.

Decoded actions are bit-identical to the committed baseline on 55 fixed cases:
the original seven inputs, 16 held-out observations, and 32 additional noise
and prompt variants. The BF16 path separately preserves all seven original
outputs exactly. `tests/pi05/compare_exact.py` enforces finite 10-by-7 actions,
matching ordered requests and backend, and identical decoded floating-point
bits, including signed zero. These checks establish no additional numerical
drift on the tested inputs; the earlier quantized-profile limitations remain.

GPU checks cover exact NVFP4 GELU packing, FP8 normalization/GELU packing,
vision attention, cached attention, rotary tables, and fused rotation/tail
clearing. They include changed-input graph replay and poisoned padding.
Workspace validation passes 7,649 tests with 222 ignored, plus formatting,
workspace/example clippy, rustdoc, mdBook, license headers, typos, and kernel
shadow checks. Changed Rust files remain within the 500-line cap.

Paired native results are saved as `native-exact-baseline-*.json` and
`native-exact-final-*.json` in the experiment's `cache/results/` directory.
The 55-case comparisons are in `native-exact-quality/`, with BF16 parity in
`native-exact-bf16-parity.json` and kernel checks in `native-exact-*-ops.json`.
