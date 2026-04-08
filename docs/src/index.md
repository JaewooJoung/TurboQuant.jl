# TurboQuant.jl

A Julia implementation of advanced vector quantization based on [TurboQuant (arXiv:2504.19874v1)](https://arxiv.org/abs/2504.19874v1).

Achieves near information-theoretically optimal compression without any calibration data or codebook learning.

## Key Features

- **Online / data-oblivious** — no preprocessing, calibration, or codebook learning required
- **Near-optimal distortion** — within 2.72× of Shannon's lower bound
- **Unbiased inner products** — critical for attention mechanisms and nearest-neighbor search
- **Fully vectorizable** — designed for GPU/TPU-friendly execution

## Installation

```julia
using Pkg
Pkg.add("TurboQuant")
```

## Quick Start

### MSE Quantizer

```julia
using TurboQuant
using LinearAlgebra

d = 128    # vector dimension
b = 4      # bits per coordinate
tq = setup(TurboQuantMSE, d, b)

x = randn(d)                          # original vector
comp = quantize(tq, x)                # compress
x_hat = dequantize(tq, comp)          # decompress
println("MSE: ", sum((x .- x_hat[:,1]).^2) / d)
```

### Batch Quantization

```julia
X = randn(d, 1000)                    # 1000 vectors
comp = quantize(tq, X)
X_hat = dequantize(tq, comp)
```

### Unbiased Inner Product Quantizer

```julia
tq_prod = setup(TurboQuantProd, d, 4)
comp = quantize(tq_prod, X)
y = randn(d)
approx_ips = inner_product(tq_prod, comp, y)   # E[result] = X' * y (unbiased)
```

### Nearest Neighbor Search

```julia
index = build_index(X, 4; use_hadamard=true)
indices, scores = search(index, y, 10)
```

### KV Cache

```julia
cache = KVCache(8, 64, 3; max_seq_len=131072)
K = randn(64, 8); V = randn(64, 8)
compress_kv!(cache, K, V)
Q = randn(64, 8)
output = attention_with_quantized_kv(cache, Q)
```

## References

- Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization for Near-Optimal Quantized Retrieval, KV Cache Quantization, and Beyond.* arXiv:2504.19874v1, April 2025.
