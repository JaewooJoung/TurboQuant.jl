# TurboQuant_MSE: MSE-optimal vector quantizer via random rotation + Lloyd-Max

"""
    TurboQuantMSE{R}

MSE-optimal vector quantizer using:
1. Random rotation Π to normalize coordinate distribution
2. Per-coordinate Lloyd-Max scalar quantization
3. Norm preservation via separate float storage

Type parameter `R` is the rotation type (RandomRotation or HadamardRotation).
"""
struct TurboQuantMSE{R}
    rotation::R
    codebook::LloydMaxCodebook
    dim::Int
    bit_width::Int
end

"""
    CompressedVectorMSE

Compressed representation of a batch of vectors using MSE quantization.
"""
struct CompressedVectorMSE
    indices::Matrix{UInt8}   # (d, N) quantization indices, 1-based
    norms::Vector{Float64}   # (N,) original vector norms
    dim::Int
    n_vectors::Int
    bit_width::Int
end

"""
    setup(::Type{TurboQuantMSE}, d, b; seed=UInt64(42), use_hadamard=false)

Initialize the MSE quantizer for vectors of dimension `d` at bit-width `b`.
"""
function setup(::Type{TurboQuantMSE}, d::Int, b::Int;
               seed::UInt64=UInt64(42), use_hadamard::Bool=false)
    if use_hadamard
        rotation = HadamardRotation(d, seed)
    else
        rotation = RandomRotation(d, seed)
    end
    codebook = solve_codebook(d, b)
    return TurboQuantMSE(rotation, codebook, d, b)
end

"""
    quantize(tq::TurboQuantMSE, X::AbstractMatrix)

Quantize a batch of vectors. X is (d, N) where each column is a vector.
Returns a CompressedVectorMSE.
"""
function quantize(tq::TurboQuantMSE, X::AbstractMatrix)
    d, N = size(X)
    @assert d == tq.dim "Dimension mismatch: expected $(tq.dim), got $d"

    # Store norms and normalize
    norms = [norm(view(X, :, j)) for j in 1:N]
    X_normalized = similar(X, Float64)
    for j in 1:N
        n = norms[j]
        if n > 1e-15
            X_normalized[:, j] .= X[:, j] ./ n
        else
            X_normalized[:, j] .= 0.0
        end
    end

    # Rotate
    Y = rotate(tq.rotation, X_normalized)

    # Quantize each coordinate
    indices = Matrix{UInt8}(undef, d, N)
    for j in 1:N
        for i in 1:d
            indices[i, j] = UInt8(quantize_scalar(Y[i, j], tq.codebook))
        end
    end

    return CompressedVectorMSE(indices, norms, d, N, tq.bit_width)
end

"""
    quantize(tq::TurboQuantMSE, x::AbstractVector)

Quantize a single vector. Returns a CompressedVectorMSE with N=1.
"""
function quantize(tq::TurboQuantMSE, x::AbstractVector)
    return quantize(tq, reshape(x, :, 1))
end

"""
    dequantize(tq::TurboQuantMSE, comp::CompressedVectorMSE)

Dequantize a compressed representation back to approximate vectors.
Returns a (d, N) matrix.
"""
function dequantize(tq::TurboQuantMSE, comp::CompressedVectorMSE)
    d, N = comp.dim, comp.n_vectors

    # Lookup centroids
    Y_hat = Matrix{Float64}(undef, d, N)
    for j in 1:N
        for i in 1:d
            Y_hat[i, j] = dequantize_scalar(Int(comp.indices[i, j]), tq.codebook)
        end
    end

    # Rotate back
    X_hat = rotate_back(tq.rotation, Y_hat)

    # Restore norms
    for j in 1:N
        X_hat[:, j] .*= comp.norms[j]
    end

    return X_hat
end

"""
    compression_ratio(comp::CompressedVectorMSE)

Compute the compression ratio relative to Float64 storage.
"""
function compression_ratio(comp::CompressedVectorMSE)
    original_bits = comp.dim * comp.n_vectors * 64  # Float64
    compressed_bits = comp.dim * comp.n_vectors * comp.bit_width + comp.n_vectors * 64  # indices + norms
    return original_bits / compressed_bits
end

"""
    mse_distortion(tq::TurboQuantMSE, X::AbstractMatrix)

Compute the mean squared error of quantize→dequantize roundtrip.
"""
function mse_distortion(tq::TurboQuantMSE, X::AbstractMatrix)
    comp = quantize(tq, X)
    X_hat = dequantize(tq, comp)
    return mean(sum((X .- X_hat).^2, dims=1)) / size(X, 1)
end
