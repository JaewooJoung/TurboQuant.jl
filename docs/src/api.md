# API Reference

## Codebook

```@docs
LloydMaxCodebook
solve_codebook
quantize_scalar
dequantize_scalar
precompute_codebooks
```

## Rotation

```@docs
RandomRotation
HadamardRotation
rotate
rotate_back
```

## MSE Quantizer

```@docs
TurboQuantMSE
CompressedVectorMSE
setup(::Type{TurboQuantMSE}, ::Int, ::Int)
quantize(::TurboQuantMSE, ::AbstractMatrix)
dequantize(::TurboQuantMSE, ::CompressedVectorMSE)
compression_ratio
mse_distortion
```

## Inner Product Quantizer

```@docs
TurboQuantProd
CompressedVectorProd
setup(::Type{TurboQuantProd}, ::Int, ::Int)
quantize(::TurboQuantProd, ::AbstractMatrix)
dequantize(::TurboQuantProd, ::CompressedVectorProd)
inner_product
inner_product_bias
```

## KV Cache

```@docs
KVCache
compress_kv!
attention_with_quantized_kv
memory_usage
fp16_memory_usage
```

## Nearest Neighbor Search

```@docs
TurboQuantIndex
build_index
search(::TurboQuantIndex, ::AbstractVector, ::Int)
batch_search
recall_at_k
```
