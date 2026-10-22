# Native pi0.5 integration

Base: `integrate/glm53-fp4local-20261020` at `34399129`, containing the
`5d652f8e` native GLM implementation deployed on both DGX Sparks.

1. Define a strictly validated `pi05_libero` checkpoint profile and load the
   converted OpenPi safetensors, preserving mixed BF16/FP32 compute types.
2. Add a native policy crate using Atlas's GPU I/O boundary and cuBLASLt,
   with SigLIP, bidirectional PaliGemma prefix attention, cached prefix KV,
   adaptive RMSNorm, and the ten-step flow-matching action expert.
3. Add pure Rust LIBERO preprocessing, SentencePiece tokenization, action
   denormalization, and a policy command beside Atlas's chat command.
4. Compare fixed-input/fixed-noise native actions with the existing OpenPi
   reference; investigate discrepancies before reporting performance.
5. Benchmark synchronized warm inference on both Sparks against the saved
   OpenPi BF16 and TensorRT measurements. Keep load and warmup separate.
6. Run formatting, lint, unit, documentation, and GPU checks; document the
   actual supported profile, protocol, precision, and measured limitations.

The engine remains Rust/CUDA. Python is used only for reference fixtures and
benchmark harnesses. The existing GLM deployment and its rollback remain intact.

## Native NVFP4 extension

1. Preserve the validated BF16 source/binary/results and inspect the tutorial's
   calibrated quantization recipe, exported weights, and scale conventions.
2. Define an explicit versioned quantization manifest. Retain sensitive FP32
   operations and make every quantized linear and activation scale auditable.
3. Implement and GPU-test packed E2M1 weights, E4M3 block scales, activation
   quantization, and native Blackwell FP4 matrix multiplication. Do not label
   BF16 dequantization or an external TensorRT engine as native NVFP4 compute.
4. Load quantized weights through the existing I/O boundary and dispatch from
   policy linear layers, preserving the BF16 path and CUDA graph correctness.
5. Validate primitive quantization/GEMM against independent references, then
   compare paired full-model actions against the quantized reference and BF16.
6. Benchmark both Sparks using the existing observations/noise and synchronized
   timing, check HTTP behavior, document precision and measured limitations,
   and retain recovery snapshots and a working rollback path.

## Latency optimization toward the NVIDIA pipeline

1. Preserve the 98 ms native NVFP4 build and compare NVIDIA's actual export,
   precision, shapes, fusions, and measured TensorRT layer profile.
2. Profile native graph execution on a Spark to rank vision, prefix, attention,
   and the ten expert steps by GPU time and distinguish launch/transfer costs.
3. Optimize the largest measured costs with native kernels, projection fusion,
   and calibrated FP8 where justified. Keep both image observations, all ten
   denoising steps, and the full action horizon; do not cache changing inputs.
4. Validate primitives and paired full-model actions after each numerical
   change. Separate speed improvements from quantization quality limitations.
5. Benchmark the selected implementation on both Sparks with the same seven
   fixtures and HTTP checks, retaining source/binary provenance and rollback.

Lesson from the initial NVFP4 milestone: implementing one quantized region is
not the performance target. Compare the complete precision map and the dominant
GPU stages against NVIDIA's pipeline before treating the optimization as done.
