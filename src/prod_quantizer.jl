# TurboQuant_Prod: Unbiased inner product quantizer (MSE + QJL residual)

"""
    TurboQuantProd{R}

Unbiased inner product quantizer using two-stage pipeline:
1. TurboQuant_MSE at (b-1) bits for base approximation
2. QJL (Quantized Johnson-Lindenstrauss) 1-bit projection on residual

Guarantees E[⟨y, x̃⟩] = ⟨y, x⟩ (unbiased inner product estimation).
"""
struct TurboQuantProd{R}
    mse_quantizer::TurboQuantMSE{R}
    S::Matrix{Float64}    # Random Gaussian projection matrix for QJL
    dim::Int
    bit_width::Int        # Total effective bit-width (b-1 + 1 = b)
    seed_qjl::UInt64
end

"""
    CompressedVectorProd

Compressed representation using the two-stage prod quantizer.
"""
struct CompressedVectorProd
    mse_part::CompressedVectorMSE   # (b-1)-bit MSE quantization
    qjl_signs::Matrix{Int8}         # (d, N) sign bits of projected residual
    residual_norms::Vector{Float64} # (N,) L2 norms of residuals
    dim::Int
    n_vectors::Int
    bit_width::Int
end

"""
    setup(::Type{TurboQuantProd}, d, b; seed=UInt64(42), seed_qjl=UInt64(123), use_hadamard=false)

Initialize the unbiased inner product quantizer.
Uses (b-1) bits for MSE stage + 1 bit for QJL residual stage.
Requires b ≥ 2.
"""
function setup(::Type{TurboQuantProd}, d::Int, b::Int;
               seed::UInt64=UInt64(42), seed_qjl::UInt64=UInt64(123),
               use_hadamard::Bool=false)
    @assert b >= 2 "TurboQuantProd requires bit-width b ≥ 2 (uses b-1 for MSE + 1 for QJL)"

    mse_q = setup(TurboQuantMSE, d, b - 1; seed=seed, use_hadamard=use_hadamard)

    # Generate random Gaussian projection matrix for QJL
    rng = MersenneTwister(seed_qjl)
    S = randn(rng, d, d) ./ sqrt(d)

    return TurboQuantProd(mse_q, S, d, b, seed_qjl)
end

"""
    quantize(tq::TurboQuantProd, X::AbstractMatrix)

Two-stage quantization:
1. MSE quantize at (b-1) bits
2. Compute residual, apply QJL (sign of random projection)
"""
function quantize(tq::TurboQuantProd, X::AbstractMatrix)
    d, N = size(X)
    @assert d == tq.dim "Dimension mismatch: expected $(tq.dim), got $d"

    # Stage 1: MSE quantization at (b-1) bits
    mse_comp = quantize(tq.mse_quantizer, X)
    X_mse = dequantize(tq.mse_quantizer, mse_comp)

    # Compute residual
    R = X .- X_mse

    # Stage 2: QJL on residual
    residual_norms = [norm(view(R, :, j)) for j in 1:N]

    # Project residuals: S · r for each vector
    SR = tq.S * R  # (d, N)

    # Take signs
    qjl_signs = Matrix{Int8}(undef, d, N)
    for j in 1:N
        for i in 1:d
            qjl_signs[i, j] = SR[i, j] >= 0 ? Int8(1) : Int8(-1)
        end
    end

    return CompressedVectorProd(mse_comp, qjl_signs, residual_norms, d, N, tq.bit_width)
end

"""
    quantize(tq::TurboQuantProd, x::AbstractVector)

Quantize a single vector.
"""
function quantize(tq::TurboQuantProd, x::AbstractVector)
    return quantize(tq, reshape(x, :, 1))
end

"""
    dequantize(tq::TurboQuantProd, comp::CompressedVectorProd)

Two-stage dequantization:
x̃ = x̃_mse + √(π/2)/d · ‖r‖ · S^T · qjl_signs
"""
function dequantize(tq::TurboQuantProd, comp::CompressedVectorProd)
    d, N = comp.dim, comp.n_vectors

    # Stage 1: MSE dequantization
    X_mse = dequantize(tq.mse_quantizer, comp.mse_part)

    # Stage 2: QJL reconstruction
    # x̃_qjl = √(π/2) / d · γ · S^T · qjl
    scale = sqrt(π / 2) / d
    qjl_float = Matrix{Float64}(comp.qjl_signs)
    X_qjl = tq.S' * qjl_float  # (d, N)

    for j in 1:N
        X_qjl[:, j] .*= scale * comp.residual_norms[j]
    end

    return X_mse .+ X_qjl
end

"""
    inner_product(tq::TurboQuantProd, comp::CompressedVectorProd, y::AbstractVector)

Compute approximate inner products ⟨x̃_i, y⟩ for all quantized vectors,
without full dequantization. Returns a vector of length N.

This is unbiased: E[result] = ⟨x, y⟩ for each vector.
"""
function inner_product(tq::TurboQuantProd, comp::CompressedVectorProd, y::AbstractVector)
    d, N = comp.dim, comp.n_vectors
    @assert length(y) == d

    # MSE part inner products
    X_mse = dequantize(tq.mse_quantizer, comp.mse_part)
    ip_mse = X_mse' * y  # (N,)

    # QJL part inner products: ⟨y, S^T · qjl⟩ · scale · γ
    # = ⟨S · y, qjl⟩ · scale · γ
    Sy = tq.S * y  # (d,)
    scale = sqrt(π / 2) / d

    result = Vector{Float64}(undef, N)
    for j in 1:N
        ip_qjl = 0.0
        for i in 1:d
            ip_qjl += Sy[i] * comp.qjl_signs[i, j]
        end
        result[j] = ip_mse[j] + scale * comp.residual_norms[j] * ip_qjl
    end

    return result
end

"""
    inner_product_bias(tq::TurboQuantProd, X::AbstractMatrix, y::AbstractVector; n_trials=100)

Estimate the bias of inner product estimation by repeated quantization.
Returns (mean_bias, std_bias) where bias = estimated_ip - true_ip.
"""
function inner_product_bias(tq::TurboQuantProd, X::AbstractMatrix, y::AbstractVector;
                            n_trials::Int=100)
    true_ip = X' * y
    biases = zeros(length(true_ip), n_trials)

    for t in 1:n_trials
        comp = quantize(tq, X)
        est_ip = inner_product(tq, comp, y)
        biases[:, t] .= est_ip .- true_ip
    end

    mean_bias = mean(biases, dims=2)[:]
    std_bias = std(biases, dims=2)[:]
    return mean_bias, std_bias
end
