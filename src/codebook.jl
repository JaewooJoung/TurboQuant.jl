# Lloyd-Max optimal codebook solver for Beta/Gaussian distributed coordinates

"""
    LloydMaxCodebook

Stores precomputed optimal centroids and decision boundaries for a given
bit-width and target distribution (Beta distribution of rotated coordinates).
"""
struct LloydMaxCodebook
    bit_width::Int
    centroids::Vector{Float64}
    boundaries::Vector{Float64}
    dimension::Int  # original vector dimension (affects Beta distribution shape)
end

# Use C standard library lgamma (available on all platforms)
_lgamma(x::Float64) = ccall(:lgamma, Float64, (Float64,), x)

"""
    beta_pdf(x, d)

PDF of a single coordinate of a uniformly random unit vector in R^d.
f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}  for x ∈ [-1, 1]

For large d, this converges to N(0, 1/d).
"""
function beta_pdf(x::Float64, d::Int)
    if abs(x) >= 1.0
        return 0.0
    end
    if d >= 50
        # Gaussian approximation: N(0, 1/d)
        σ² = 1.0 / d
        return exp(-x^2 / (2σ²)) / sqrt(2π * σ²)
    end
    # Exact Beta distribution on [-1, 1]
    log_norm = _lgamma(d / 2.0) - 0.5 * log(π) - _lgamma((d - 1) / 2.0)
    log_val = log_norm + ((d - 3) / 2.0) * log(max(1.0 - x^2, 1e-300))
    return exp(log_val)
end

"""
    solve_codebook(d, b; max_iter=1000, tol=1e-10, n_quad=10000)

Solve the Lloyd-Max optimal quantizer for the Beta distribution arising from
random rotation in dimension `d`, at bit-width `b`.

Returns a `LloydMaxCodebook` with 2^b centroids and 2^b-1 decision boundaries.
"""
function solve_codebook(d::Int, b::Int; max_iter::Int=1000, tol::Float64=1e-10, n_quad::Int=10000)
    K = 1 << b  # number of centroids = 2^b

    # Integration grid over [-1, 1] (or truncated Gaussian range)
    if d >= 50
        σ = 1.0 / sqrt(d)
        lo, hi = -5σ, 5σ
    else
        lo, hi = -0.9999, 0.9999
    end
    xs = range(lo, hi, length=n_quad)
    dx = xs[2] - xs[1]
    pdf_vals = [beta_pdf(Float64(x), d) for x in xs]

    # Initialize centroids uniformly
    centroids = collect(range(lo * 0.8, hi * 0.8, length=K))
    boundaries = zeros(K - 1)

    prev_distortion = Inf
    for iter in 1:max_iter
        # Update boundaries (midpoints between adjacent centroids)
        for i in 1:(K-1)
            boundaries[i] = (centroids[i] + centroids[i+1]) / 2.0
        end

        # Update centroids using conditional expectation
        new_centroids = zeros(K)
        distortion = 0.0

        for k in 1:K
            lb = k == 1 ? lo : boundaries[k-1]
            ub = k == K ? hi : boundaries[k]

            num = 0.0
            den = 0.0
            dist_k = 0.0

            for (i, x) in enumerate(xs)
                if lb <= x <= ub
                    w = pdf_vals[i] * dx
                    num += x * w
                    den += w
                    dist_k += (x - centroids[k])^2 * w
                end
            end

            new_centroids[k] = den > 1e-15 ? num / den : centroids[k]
            distortion += dist_k
        end

        centroids .= new_centroids

        if abs(prev_distortion - distortion) < tol * max(abs(distortion), 1e-15)
            break
        end
        prev_distortion = distortion
    end

    sort!(centroids)

    # Final boundaries
    for i in 1:(K-1)
        boundaries[i] = (centroids[i] + centroids[i+1]) / 2.0
    end

    return LloydMaxCodebook(b, centroids, boundaries, d)
end

"""
    quantize_scalar(val, cb::LloydMaxCodebook)

Quantize a single scalar value using the precomputed codebook.
Returns the index (1-based) of the nearest centroid.
"""
function quantize_scalar(val::Float64, cb::LloydMaxCodebook)
    # Binary search on boundaries
    lo, hi = 1, length(cb.boundaries)
    idx = length(cb.centroids)  # default to last centroid

    while lo <= hi
        mid = (lo + hi) >> 1
        if val <= cb.boundaries[mid]
            idx = mid
            hi = mid - 1
        else
            lo = mid + 1
        end
    end
    return idx
end

@inline function quantize_scalar(val::Float32, cb::LloydMaxCodebook)
    return quantize_scalar(Float64(val), cb)
end

"""
    dequantize_scalar(idx, cb::LloydMaxCodebook)

Look up the centroid value for index `idx`.
"""
@inline function dequantize_scalar(idx::Int, cb::LloydMaxCodebook)
    return cb.centroids[idx]
end

"""
    precompute_codebooks(d; max_bits=8)

Precompute Lloyd-Max codebooks for bit-widths 1 through `max_bits`.
Returns a Dict{Int, LloydMaxCodebook}.
"""
function precompute_codebooks(d::Int; max_bits::Int=8)
    codebooks = Dict{Int, LloydMaxCodebook}()
    for b in 1:max_bits
        codebooks[b] = solve_codebook(d, b)
    end
    return codebooks
end
