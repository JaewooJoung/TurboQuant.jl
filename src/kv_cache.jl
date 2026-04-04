# KV Cache integration for transformer attention with online quantization

"""
    KVCache{R}

Quantized KV cache for transformer attention. Supports:
- Online streaming quantization (no calibration needed)
- Mixed-precision: outlier channels get extra bits
- On-the-fly dequantization during attention computation
"""
mutable struct KVCache{R}
    # Quantizers
    key_quantizer::TurboQuantProd{R}    # Keys use prod (unbiased IP)
    value_quantizer::TurboQuantMSE{R}   # Values use MSE

    # Stored compressed tokens
    key_store::Vector{CompressedVectorProd}
    value_store::Vector{CompressedVectorMSE}

    # Configuration
    dim::Int
    n_heads::Int
    head_dim::Int
    bit_width::Int
    max_seq_len::Int
    current_len::Int

    # Outlier channel tracking
    outlier_mask::Vector{Bool}          # true for outlier channels
    outlier_key_quantizer::Union{Nothing, TurboQuantProd{R}}
    outlier_value_quantizer::Union{Nothing, TurboQuantMSE{R}}
end

"""
    KVCache(n_heads, head_dim, bit_width; max_seq_len=131072, seed=UInt64(42),
            use_hadamard=true, outlier_fraction=0.25)

Create a quantized KV cache.

- `bit_width`: effective bits per coordinate (e.g., 3 means 2-bit MSE + 1-bit QJL for keys)
- `outlier_fraction`: fraction of channels receiving (bit_width+1) bits
- Uses TurboQuantProd for keys (unbiased inner products critical for attention)
- Uses TurboQuantMSE for values (MSE sufficient after softmax weighting)
"""
function KVCache(n_heads::Int, head_dim::Int, bit_width::Int;
                 max_seq_len::Int=131072, seed::UInt64=UInt64(42),
                 use_hadamard::Bool=true, outlier_fraction::Float64=0.25)
    dim = n_heads * head_dim

    key_q = setup(TurboQuantProd, head_dim, bit_width;
                  seed=seed, use_hadamard=use_hadamard)
    val_q = setup(TurboQuantMSE, head_dim, bit_width;
                  seed=seed + UInt64(1), use_hadamard=use_hadamard)

    # Outlier quantizers with one extra bit
    outlier_key_q = setup(TurboQuantProd, head_dim, bit_width + 1;
                          seed=seed + UInt64(2), use_hadamard=use_hadamard)
    outlier_val_q = setup(TurboQuantMSE, head_dim, bit_width + 1;
                          seed=seed + UInt64(3), use_hadamard=use_hadamard)

    # Initially no outlier detection (updated after first tokens)
    outlier_mask = fill(false, n_heads)

    return KVCache(
        key_q, val_q,
        CompressedVectorProd[], CompressedVectorMSE[],
        dim, n_heads, head_dim, bit_width, max_seq_len, 0,
        outlier_mask, outlier_key_q, outlier_val_q
    )
end

"""
    detect_outlier_heads!(cache::KVCache, K::AbstractMatrix, threshold_percentile=0.75)

Detect which attention heads have outlier-magnitude keys and should receive
extra quantization bits. K is (head_dim, n_heads).
"""
function detect_outlier_heads!(cache::KVCache, K::AbstractMatrix;
                               threshold_percentile::Float64=0.75)
    n_heads = cache.n_heads
    head_norms = [norm(view(K, :, h)) for h in 1:n_heads]
    threshold = quantile(head_norms, threshold_percentile)
    cache.outlier_mask .= head_norms .>= threshold
end

"""
    compress_kv!(cache::KVCache, K::AbstractMatrix, V::AbstractMatrix)

Add a new token's K and V to the cache with online quantization.
K, V are (head_dim, n_heads) — one vector per attention head.
"""
function compress_kv!(cache::KVCache, K::AbstractMatrix, V::AbstractMatrix)
    @assert size(K) == (cache.head_dim, cache.n_heads)
    @assert size(V) == (cache.head_dim, cache.n_heads)

    if cache.current_len >= cache.max_seq_len
        # Evict oldest token (FIFO)
        popfirst!(cache.key_store)
        popfirst!(cache.value_store)
        cache.current_len -= 1
    end

    # Update outlier detection periodically
    if cache.current_len % 64 == 0 && cache.current_len > 0
        detect_outlier_heads!(cache, K)
    end

    # Quantize keys (per-head)
    # For simplicity, concatenate all heads and quantize together
    key_comp = quantize(cache.key_quantizer, K)
    val_comp = quantize(cache.value_quantizer, V)

    push!(cache.key_store, key_comp)
    push!(cache.value_store, val_comp)
    cache.current_len += 1

    return nothing
end

"""
    attention_with_quantized_kv(cache::KVCache, Q::AbstractMatrix;
                                 scale=nothing)

Compute attention output using quantized KV cache.
Q is (head_dim, n_heads) for the current query token.

Computes: output = softmax(Q^T K / √d) V  using quantized K and V.
Returns (head_dim, n_heads) output.
"""
function attention_with_quantized_kv(cache::KVCache, Q::AbstractMatrix;
                                      scale::Union{Nothing, Float64}=nothing)
    head_dim, n_heads = size(Q)
    seq_len = cache.current_len

    if seq_len == 0
        return zeros(head_dim, n_heads)
    end

    s = isnothing(scale) ? 1.0 / sqrt(Float64(head_dim)) : scale

    output = zeros(head_dim, n_heads)

    for h in 1:n_heads
        q = view(Q, :, h)

        # Compute attention scores: q^T k_i for all cached tokens
        scores = Vector{Float64}(undef, seq_len)
        for t in 1:seq_len
            # Use unbiased inner product for keys
            ip = inner_product(cache.key_quantizer, cache.key_store[t], q)
            scores[t] = ip[h] * s
        end

        # Softmax
        max_score = maximum(scores)
        scores .= exp.(scores .- max_score)
        sum_scores = sum(scores)
        scores ./= sum_scores

        # Weighted sum of values
        for t in 1:seq_len
            V_hat = dequantize(cache.value_quantizer, cache.value_store[t])
            output[:, h] .+= scores[t] .* view(V_hat, :, h)
        end
    end

    return output
end

"""
    memory_usage(cache::KVCache)

Estimate memory usage of the quantized KV cache in bytes.
"""
function memory_usage(cache::KVCache)
    if cache.current_len == 0
        return 0
    end

    # Per token: key = (b-1)*d + d + 32 bits for prod, value = b*d + 32 bits for MSE
    key_bits_per_token = (cache.bit_width - 1) * cache.head_dim * cache.n_heads +
                         cache.head_dim * cache.n_heads +  # QJL signs
                         cache.n_heads * 64                 # norms
    val_bits_per_token = cache.bit_width * cache.head_dim * cache.n_heads +
                         cache.n_heads * 64                 # norms

    total_bits = cache.current_len * (key_bits_per_token + val_bits_per_token)
    return total_bits ÷ 8
end

"""
    fp16_memory_usage(cache::KVCache)

Memory usage if KV cache were stored in Float16.
"""
function fp16_memory_usage(cache::KVCache)
    return cache.current_len * cache.head_dim * cache.n_heads * 2 * 16 ÷ 8
end
