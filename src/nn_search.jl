# Nearest Neighbor Search using TurboQuant compression

"""
    TurboQuantIndex{R}

A nearest neighbor search index using TurboQuant compression.
No codebook learning required — vectors are quantized online
using random rotation + Lloyd-Max quantization.
"""
struct TurboQuantIndex{R}
    quantizer::TurboQuantProd{R}
    compressed::Vector{CompressedVectorProd}
    n_vectors::Int
    dim::Int
    bit_width::Int
end

"""
    build_index(X::AbstractMatrix, bit_width::Int;
                seed=UInt64(42), use_hadamard=true, batch_size=1000)

Build a nearest neighbor search index from database vectors.
X is (d, N) where each column is a database vector.

This is essentially instantaneous compared to PQ (no k-means training).
"""
function build_index(X::AbstractMatrix, bit_width::Int;
                     seed::UInt64=UInt64(42), use_hadamard::Bool=true,
                     batch_size::Int=1000)
    d, N = size(X)
    quantizer = setup(TurboQuantProd, d, bit_width;
                      seed=seed, use_hadamard=use_hadamard)

    compressed = CompressedVectorProd[]

    # Quantize in batches
    for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        batch = X[:, start:stop]
        push!(compressed, quantize(quantizer, batch))
    end

    return TurboQuantIndex(quantizer, compressed, N, d, bit_width)
end

"""
    search(index::TurboQuantIndex, query::AbstractVector, k::Int)

Find the k approximate nearest neighbors of `query` in the index.
Returns (indices, scores) where indices are 1-based into the original database
and scores are approximate inner products (higher = more similar).
"""
function search(index::TurboQuantIndex, query::AbstractVector, k::Int)
    @assert length(query) == index.dim

    # Compute inner products with all database vectors
    all_scores = Float64[]
    all_indices = Int[]

    offset = 0
    for comp in index.compressed
        scores = inner_product(index.quantizer, comp, query)
        for (i, s) in enumerate(scores)
            push!(all_scores, s)
            push!(all_indices, offset + i)
        end
        offset += comp.n_vectors
    end

    # Find top-k
    k_actual = min(k, length(all_scores))
    perm = partialsortperm(all_scores, 1:k_actual, rev=true)

    return all_indices[perm], all_scores[perm]
end

"""
    batch_search(index::TurboQuantIndex, queries::AbstractMatrix, k::Int)

Search for k nearest neighbors for each query column in `queries`.
Returns (indices_matrix, scores_matrix) each of size (k, n_queries).
"""
function batch_search(index::TurboQuantIndex, queries::AbstractMatrix, k::Int)
    d, n_queries = size(queries)
    @assert d == index.dim

    result_indices = Matrix{Int}(undef, k, n_queries)
    result_scores = Matrix{Float64}(undef, k, n_queries)

    for j in 1:n_queries
        idxs, scores = search(index, view(queries, :, j), k)
        k_actual = length(idxs)
        result_indices[1:k_actual, j] .= idxs
        result_scores[1:k_actual, j] .= scores
        if k_actual < k
            result_indices[k_actual+1:k, j] .= 0
            result_scores[k_actual+1:k, j] .= -Inf
        end
    end

    return result_indices, result_scores
end

"""
    recall_at_k(index::TurboQuantIndex, queries::AbstractMatrix,
                ground_truth::AbstractMatrix, k::Int)

Compute recall@1@k: fraction of queries where the true nearest neighbor
is among the top-k results from the approximate search.

`ground_truth` is (1, n_queries) containing the index of each query's true NN.
"""
function recall_at_k(index::TurboQuantIndex, queries::AbstractMatrix,
                     ground_truth::AbstractVector{Int}, k::Int)
    result_indices, _ = batch_search(index, queries, k)
    n_queries = size(queries, 2)

    hits = 0
    for j in 1:n_queries
        if ground_truth[j] in view(result_indices, :, j)
            hits += 1
        end
    end

    return hits / n_queries
end
