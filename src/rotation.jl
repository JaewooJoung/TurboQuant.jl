# Random rotation modules: dense orthogonal and structured Hadamard

"""
    RandomRotation

Dense random orthogonal rotation matrix generated via QR decomposition
of a Gaussian random matrix. O(d²) storage and O(d²) multiply.
"""
struct RandomRotation
    Q::Matrix{Float64}
    seed::UInt64
end

"""
    RandomRotation(d, seed)

Create a random orthogonal matrix Π ∈ R^{d×d} from the given seed.
Uses QR decomposition of a random Gaussian matrix.
"""
function RandomRotation(d::Int, seed::UInt64=UInt64(42))
    rng = MersenneTwister(seed)
    G = randn(rng, d, d)
    F = qr(G)
    # Ensure proper orthogonal matrix (det = +1 adjustment)
    Q = Matrix(F.Q)
    # Fix sign ambiguity: multiply columns by sign of diagonal of R
    R_diag = diag(F.R)
    for j in 1:d
        if R_diag[j] < 0
            Q[:, j] .*= -1
        end
    end
    return RandomRotation(Q, seed)
end

"""
    rotate(R::RandomRotation, x)

Apply forward rotation: y = Π · x.
Supports vectors (d,) and matrices (d, N) for batch processing.
"""
function rotate(R::RandomRotation, x::AbstractVecOrMat)
    return R.Q * x
end

"""
    rotate_back(R::RandomRotation, y)

Apply inverse rotation: x = Π^T · y.
"""
function rotate_back(R::RandomRotation, y::AbstractVecOrMat)
    return R.Q' * y
end

# ─────────────────────────────────────────────────────────────────────────────
# Structured Hadamard rotation: O(d log d) complexity
# ─────────────────────────────────────────────────────────────────────────────

"""
    HadamardRotation

Structured random rotation using randomized Hadamard transform:
Π = D₃ H D₂ H D₁  (three rounds of Hadamard with random diagonal signs).

This achieves O(d log d) rotation instead of O(d²), with similar
distributional properties for sub-Gaussian tail bounds.
"""
struct HadamardRotation
    d::Int          # original dimension
    d_pad::Int      # padded to next power of 2
    signs::Vector{Vector{Int8}}  # 3 rounds of random ±1 diagonals
    seed::UInt64
end

"""
    HadamardRotation(d, seed; rounds=3)

Create a structured Hadamard rotation for dimension `d`.
Pads to next power of 2 internally.
"""
function HadamardRotation(d::Int, seed::UInt64=UInt64(42); rounds::Int=3)
    d_pad = nextpow(2, d)
    rng = MersenneTwister(seed)
    signs = [Int8.(2 .* (rand(rng, Bool, d_pad)) .- 1) for _ in 1:rounds]
    return HadamardRotation(d, d_pad, signs, seed)
end

"""
    fwht!(x)

Fast Walsh-Hadamard Transform (in-place, unnormalized).
"""
function fwht!(x::AbstractVector)
    n = length(x)
    h = 1
    while h < n
        for i in 1:2h:n
            for j in 0:(h-1)
                a = x[i + j]
                b = x[i + j + h]
                x[i + j] = a + b
                x[i + j + h] = a - b
            end
        end
        h <<= 1
    end
    return x
end

"""
    rotate(R::HadamardRotation, x::AbstractVector)

Apply structured Hadamard rotation to a vector.
"""
function rotate(R::HadamardRotation, x::AbstractVector{T}) where T
    d_pad = R.d_pad
    y = zeros(Float64, d_pad)
    y[1:R.d] .= x

    for round_signs in R.signs
        y .*= round_signs
        fwht!(y)
        y ./= sqrt(d_pad)
    end

    return y[1:R.d]
end

"""
    rotate(R::HadamardRotation, X::AbstractMatrix)

Batch rotation: each column is a vector to rotate.
"""
function rotate(R::HadamardRotation, X::AbstractMatrix{T}) where T
    d, n = size(X)
    Y = zeros(Float64, d, n)
    for j in 1:n
        Y[:, j] = rotate(R, view(X, :, j))
    end
    return Y
end

"""
    rotate_back(R::HadamardRotation, y::AbstractVector)

Apply inverse structured Hadamard rotation.
The inverse of D_i H / √d is H D_i / √d (since H and D are self-inverse up to scaling).
"""
function rotate_back(R::HadamardRotation, y::AbstractVector{T}) where T
    d_pad = R.d_pad
    x = zeros(Float64, d_pad)
    x[1:R.d] .= y

    for round_signs in reverse(R.signs)
        fwht!(x)
        x ./= sqrt(d_pad)
        x .*= round_signs
    end

    return x[1:R.d]
end

function rotate_back(R::HadamardRotation, Y::AbstractMatrix{T}) where T
    d, n = size(Y)
    X = zeros(Float64, d, n)
    for j in 1:n
        X[:, j] = rotate_back(R, view(Y, :, j))
    end
    return X
end
