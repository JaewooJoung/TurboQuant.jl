[English](README.md) | [한국어](README.ko-KR.md) | [Svenska](README.sv-SE.md) | [简体中文](README.zh-CN.md)

# TurboQuant.jl

[![JuliaHub](https://juliahub.com/docs/General/TurboQuant/stable/version.svg)](https://juliahub.com/ui/Packages/General/TurboQuant)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-blueviolet?logo=julia)](https://julialang.org/)
[![pkgeval](https://juliahub.com/docs/General/TurboQuant/stable/pkgeval.svg)](https://juliahub.com/ui/Packages/General/TurboQuant)
[![deps](https://juliahub.com/docs/General/TurboQuant/stable/deps.svg)](https://juliahub.com/ui/Packages/General/TurboQuant)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JaewooJoung.github.io/TurboQuant.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JaewooJoung.github.io/TurboQuant.jl/dev)
[![Build Status](https://github.com/JaewooJoung/TurboQuant.jl/workflows/CI/badge.svg)](https://github.com/JaewooJoung/TurboQuant.jl/actions)
[![Coverage](https://codecov.io/gh/JaewooJoung/TurboQuant.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JaewooJoung/TurboQuant.jl)
[![License](https://img.shields.io/github/license/JaewooJoung/TurboQuant.jl)](https://github.com/JaewooJoung/TurboQuant.jl/blob/main/LICENSE)


A Julia implementation of advanced vector quantization based on [TurboQuant (arXiv:2504.19874v1)](https://arxiv.org/abs/2504.19874v1). Achieves near information-theoretically optimal compression without any calibration data or codebook learning.

## Key Features

- **Online / data-oblivious** — no preprocessing, calibration, or codebook learning required
- **Near-optimal distortion** — within 2.72x of Shannon's lower bound, O(1/4^b) rate
- **Unbiased inner products** — critical for attention mechanisms and nearest-neighbor search
- **Fully vectorizable** — designed for GPU/TPU-friendly execution

## Installation

```julia
using Pkg
Pkg.add("TurboQuant")
```

Or install directly from the repository:

```julia
using Pkg
Pkg.add(url="https://github.com/JaewooJoung/TurboQuant.jl")
```

## Quick Start

### Using the Module in Code

```julia
using TurboQuant
using LinearAlgebra

# --- MSE Quantizer ---
d = 128    # vector dimension
b = 4      # bits per coordinate
tq = setup(TurboQuantMSE, d, b)

x = randn(d)                          # original vector
comp = quantize(tq, x)                # compress
x_hat = dequantize(tq, comp)          # decompress
println("MSE: ", sum((x .- x_hat[:,1]).^2) / d)

# --- Batch quantization ---
X = randn(d, 1000)                    # 1000 vectors
comp = quantize(tq, X)
X_hat = dequantize(tq, comp)

# --- Unbiased Inner Product Quantizer ---
tq_prod = setup(TurboQuantProd, d, 4)
comp = quantize(tq_prod, X)
y = randn(d)
approx_ips = inner_product(tq_prod, comp, y)   # E[result] = X' * y (unbiased)
```

### File Compression (.tqt format)

Compress any file into `.tqt` format and decompress it back:

```julia
include("examples/tqt_file_io.jl")

# Lossless compression (default) — exact reconstruction guaranteed
compress_file("myfile.txt", "myfile.tqt"; bit_width=4, dim=64, lossless=true)
decompress_file("myfile.tqt", "myfile_restored.txt")
verify_roundtrip("myfile.txt", "myfile_restored.txt")
# → Result: LOSSLESS (perfect reconstruction)

# Lossy compression — smaller output, approximate reconstruction
compress_file("myfile.txt", "myfile.tqt"; bit_width=4, dim=64, lossless=false)
```

**How lossless mode works**: TurboQuant compresses the data with lossy vector
quantization, then computes the byte-level residual (original - reconstructed)
and stores it with RLE encoding. On decompression, the residual correction is
added back for exact reconstruction. Higher `bit_width` means smaller residuals
and thus better overall compression.

## Examples

### Run the Demo

The demo shows both lossless and lossy comparisons across bit-widths:

```bash
# Use built-in sample text
julia examples/example_compress.jl

# Compress your own file
julia examples/example_compress.jl path/to/myfile.txt

# Custom bit-width (6) and vector dimension (128)
julia examples/example_compress.jl myfile.txt 6 128
```

Example output:

```
=== Lossless Mode: Bit-width Comparison ===
Bits | Compressed | Ratio  | TQT part | Residual part | Exact?
-----|------------|--------|----------|---------------|-------
  1  |        ... |  ...×  |      ... |           ... | YES
  4  |        ... |  ...×  |      ... |           ... | YES
  8  |        ... |  ...×  |      ... |           ... | YES

=== Lossy Mode: Bit-width Comparison ===
Bits | Compressed | Ratio  | Exact Match %
-----|------------|--------|-------------
  1  |        ... |  ...×  | ~0.7%
  4  |        ... |  ...×  | ~3.5%
  8  |        ... |  ...×  | ~43.5%
```

### CLI Tool

A standalone command-line tool for compress / decompress / inspect:

```bash
# Lossless compression (default)
julia examples/tqt_tool.jl compress input.png output.tqt 4 128

# Lossy compression
julia examples/tqt_tool.jl compress input.png output.tqt 2 64 --lossy

# Decompress (auto-detects lossless/lossy from header)
julia examples/tqt_tool.jl decompress output.tqt restored.png

# Inspect a .tqt file header
julia examples/tqt_tool.jl info output.tqt
```

`info` output shows mode (lossless/lossy), original size, vector dimensions, bit-width, codebook centroids, residual size, and effective bits-per-byte.

## Module Components

### Codebook (`src/codebook.jl`)

Lloyd-Max optimal scalar quantizer for the Beta distribution that arises when projecting unit vectors via random rotation.

```julia
cb = solve_codebook(128, 4)       # dimension=128, 4-bit
# cb.centroids  — 16 optimal centroid values
# cb.boundaries — 15 decision boundaries
```

### Rotation (`src/rotation.jl`)

Two rotation modes:

| Mode | Complexity | Usage |
|---|---|---|
| `RandomRotation(d, seed)` | O(d²) | Exact orthogonal, best quality |
| `HadamardRotation(d, seed)` | O(d log d) | Structured, faster for large d |

```julia
R = RandomRotation(128, UInt64(42))
y = rotate(R, x)          # forward
x_back = rotate_back(R, y) # inverse
```

### MSE Quantizer (`src/mse_quantizer.jl`)

MSE-optimal vector quantizer: random rotation + per-coordinate Lloyd-Max.

```julia
tq = setup(TurboQuantMSE, d, b; use_hadamard=false)
comp = quantize(tq, X)         # returns CompressedVectorMSE
X_hat = dequantize(tq, comp)
ratio = compression_ratio(comp)
```

### Inner Product Quantizer (`src/prod_quantizer.jl`)

Two-stage pipeline (MSE + QJL) guaranteeing unbiased inner products.

```julia
tq = setup(TurboQuantProd, d, b)   # b >= 2 required
comp = quantize(tq, X)
X_hat = dequantize(tq, comp)

# Compute inner products without full dequantization
scores = inner_product(tq, comp, query_vector)
```

### KV Cache (`src/kv_cache.jl`)

Drop-in quantized KV cache for transformer attention:

```julia
cache = KVCache(n_heads, head_dim, bit_width; max_seq_len=131072)

# Stream tokens
for token in tokens
    compress_kv!(cache, K_token, V_token)
end

# Compute attention with quantized cache
output = attention_with_quantized_kv(cache, Q)

# Memory comparison
println("Quantized: ", memory_usage(cache), " bytes")
println("FP16:      ", fp16_memory_usage(cache), " bytes")
```

- Keys use `TurboQuantProd` (unbiased inner products for attention scores)
- Values use `TurboQuantMSE` (MSE sufficient after softmax weighting)
- Automatic outlier head detection with extra bit allocation
- FIFO eviction at `max_seq_len`

### Nearest Neighbor Search (`src/nn_search.jl`)

Zero-training-time approximate nearest neighbor search:

```julia
# Build index (instant — no k-means)
index = build_index(database_vectors, 4; use_hadamard=true)

# Query
indices, scores = search(index, query, k)

# Batch query
idx_mat, score_mat = batch_search(index, queries, k)

# Evaluate recall
recall = recall_at_k(index, queries, ground_truth, k)
```

## .tqt File Format (v3)

Binary layout with bit-packed indices:

```
[Header]  38 bytes
  Magic:          "TQT\x03"          (4 bytes)
  Flags:          UInt8               (1 byte, bit 0 = lossless)
  Original size:  UInt64              (8 bytes)
  Dimension:      UInt32              (4 bytes)
  Bit-width:      UInt8               (1 byte)
  Num vectors:    UInt64              (8 bytes)
  Padding:        UInt32              (4 bytes)
  Seed:           UInt64              (8 bytes)

[Codebook]
  Num centroids:  UInt32              (4 bytes)
  Centroids:      Float64 × 2^b      (8 × 2^b bytes)

[Data] × n_vectors
  Norm:           Float32             (4 bytes)
  Indices:        ceil(dim×b/8) bytes (bit-packed, b bits per index)

[Residual — lossless mode only]
  Residual size:  UInt32              (4 bytes)
  Residual data:  variable            (RLE-encoded zigzag residuals)
```

**Storage math**: At `bit_width=2, dim=64`, each vector takes 4 (norm) + 16 (packed indices) = 20 bytes instead of the 72 bytes (8 + 64) in v2. For `bit_width=1`, it's just 4 + 8 = 12 bytes per vector.

## Distortion Performance

| Bit-width | MSE Distortion | Ratio to Shannon Bound |
|---|---|---|
| 1 | 0.36 | ~1.45x |
| 2 | 0.117 | ~1.87x |
| 3 | 0.030 | ~1.92x |
| 4 | 0.009 | ~2.30x |
| General b | (sqrt(3pi)/2) · 4^(-b) | ≤ 2.72x |

## Running Tests

```julia
using Pkg
Pkg.test("TurboQuant")
```

## Parameters Guide

| Parameter | Recommended | Notes |
|---|---|---|
| `bit_width` | 3-4 for quality, 1-2 for aggressive compression | Higher = better quality, less compression |
| `dim` | 64-128 | Must divide evenly into data; larger = better statistical properties |
| `use_hadamard` | `true` for d > 256 | O(d log d) vs O(d²) rotation |
| `seed` | any UInt64 | Same seed = reproducible compression |

## References

- Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization for Near-Optimal Quantized Retrieval, KV Cache Quantization, and Beyond.* arXiv:2504.19874v1, April 2025.

## License

MIT
