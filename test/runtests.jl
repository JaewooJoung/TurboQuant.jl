using Test
using LinearAlgebra
using Random
using Statistics
using TurboQuant

@testset "TurboQuant" begin

    @testset "Codebook: Lloyd-Max solver" begin
        # Test codebook generation for various bit-widths
        for b in 1:4
            cb = solve_codebook(128, b)
            @test length(cb.centroids) == 2^b
            @test length(cb.boundaries) == 2^b - 1
            @test issorted(cb.centroids)
            @test issorted(cb.boundaries)

            # Centroids should be symmetric around 0 for symmetric distribution
            for i in 1:2^(b-1)
                @test isapprox(cb.centroids[i], -cb.centroids[end - i + 1], atol=1e-6)
            end
        end

        # Test scalar quantization round-trip
        cb = solve_codebook(128, 2)
        for c in cb.centroids
            idx = quantize_scalar(c, cb)
            @test isapprox(dequantize_scalar(idx, cb), c, atol=1e-10)
        end
    end

    @testset "Rotation: Dense random orthogonal" begin
        d = 64
        R = RandomRotation(d, UInt64(42))

        # Check orthogonality: Q^T Q = I
        @test isapprox(R.Q' * R.Q, I(d), atol=1e-10)
        @test isapprox(R.Q * R.Q', I(d), atol=1e-10)

        # Round-trip: rotate then rotate_back should be identity
        x = randn(d)
        y = rotate(R, x)
        x_back = rotate_back(R, y)
        @test isapprox(x, x_back, atol=1e-10)

        # Norm preservation
        @test isapprox(norm(y), norm(x), atol=1e-10)

        # Batch rotation
        X = randn(d, 10)
        Y = rotate(R, X)
        X_back = rotate_back(R, Y)
        @test isapprox(X, X_back, atol=1e-10)
    end

    @testset "Rotation: Structured Hadamard" begin
        d = 64
        R = HadamardRotation(d, UInt64(42))

        # Round-trip
        x = randn(d)
        y = rotate(R, x)
        x_back = rotate_back(R, y)
        @test isapprox(x, x_back, atol=1e-8)

        # Norm should be approximately preserved
        @test isapprox(norm(y), norm(x), rtol=0.1)
    end

    @testset "MSE Quantizer: basic roundtrip" begin
        d = 64
        N = 50
        b = 4

        tq = setup(TurboQuantMSE, d, b; seed=UInt64(42))
        X = randn(d, N)

        comp = quantize(tq, X)
        @test comp.n_vectors == N
        @test comp.dim == d
        @test size(comp.indices) == (d, N)

        X_hat = dequantize(tq, comp)
        @test size(X_hat) == (d, N)

        # MSE should be reasonable for 4-bit
        mse = mean(sum((X .- X_hat).^2, dims=1)) / d
        @test mse < 0.1  # 4-bit should have low distortion
    end

    @testset "MSE Quantizer: distortion scaling" begin
        d = 128
        N = 100
        X = randn(d, N)
        X ./= sqrt.(sum(X.^2, dims=1))  # normalize to unit vectors

        # Distortion should decrease with more bits
        prev_mse = Inf
        for b in 1:4
            tq = setup(TurboQuantMSE, d, b; seed=UInt64(42))
            mse = mse_distortion(tq, X)
            @test mse < prev_mse
            prev_mse = mse
        end
    end

    @testset "MSE Quantizer: single vector" begin
        d = 32
        tq = setup(TurboQuantMSE, d, 3; seed=UInt64(42))
        x = randn(d)

        comp = quantize(tq, x)
        @test comp.n_vectors == 1

        x_hat = dequantize(tq, comp)
        @test size(x_hat) == (d, 1)
    end

    @testset "MSE Quantizer: compression ratio" begin
        d = 128
        N = 1000
        X = randn(d, N)

        for b in [2, 4, 8]
            tq = setup(TurboQuantMSE, d, b; seed=UInt64(42))
            comp = quantize(tq, X)
            ratio = compression_ratio(comp)
            # At b bits per dim + overhead, ratio ≈ 64/b
            @test ratio > 1.0
        end
    end

    @testset "Prod Quantizer: unbiased inner products" begin
        d = 64
        N = 20
        b = 3

        tq = setup(TurboQuantProd, d, b; seed=UInt64(42))
        X = randn(d, N)
        y = randn(d)

        true_ip = X' * y

        # Check that inner product estimates are correlated with true values
        comp = quantize(tq, X)
        est_ip = inner_product(tq, comp, y)

        @test length(est_ip) == N

        # Estimates should have same sign pattern as true IPs (most of the time)
        sign_agreement = sum(sign.(est_ip) .== sign.(true_ip))
        @test sign_agreement >= N ÷ 2  # at least half should agree

        # Correlation between estimated and true IPs should be positive
        mean_true = mean(true_ip)
        mean_est = mean(est_ip)
        cov_val = mean((true_ip .- mean_true) .* (est_ip .- mean_est))
        var_true = mean((true_ip .- mean_true).^2)
        var_est = mean((est_ip .- mean_est).^2)
        if var_true > 1e-10 && var_est > 1e-10
            correlation = cov_val / sqrt(var_true * var_est)
            @test correlation > 0.3  # positive correlation
        end

        # Estimates should be finite
        @test all(isfinite, est_ip)
    end

    @testset "Prod Quantizer: dequantize roundtrip" begin
        d = 32
        N = 10
        b = 3

        tq = setup(TurboQuantProd, d, b; seed=UInt64(42))
        X = randn(d, N)

        comp = quantize(tq, X)
        X_hat = dequantize(tq, comp)

        @test size(X_hat) == (d, N)

        # Dequantized vectors should be in the right ballpark
        for j in 1:N
            @test isapprox(norm(X_hat[:, j]), norm(X[:, j]), rtol=0.5)
        end
    end

    @testset "KV Cache: basic operations" begin
        n_heads = 4
        head_dim = 32
        bit_width = 3

        cache = KVCache(n_heads, head_dim, bit_width; max_seq_len=100)

        @test cache.current_len == 0

        # Add tokens
        for t in 1:10
            K = randn(head_dim, n_heads)
            V = randn(head_dim, n_heads)
            compress_kv!(cache, K, V)
        end

        @test cache.current_len == 10
        @test length(cache.key_store) == 10
        @test length(cache.value_store) == 10
    end

    @testset "KV Cache: attention computation" begin
        n_heads = 2
        head_dim = 16
        bit_width = 3

        cache = KVCache(n_heads, head_dim, bit_width; max_seq_len=100)

        # Add some tokens
        for t in 1:5
            K = randn(head_dim, n_heads) * 0.1
            V = randn(head_dim, n_heads) * 0.1
            compress_kv!(cache, K, V)
        end

        Q = randn(head_dim, n_heads) * 0.1
        output = attention_with_quantized_kv(cache, Q)

        @test size(output) == (head_dim, n_heads)
        # Output should be finite
        @test all(isfinite, output)
    end

    @testset "KV Cache: eviction at max_seq_len" begin
        n_heads = 2
        head_dim = 16
        bit_width = 3

        cache = KVCache(n_heads, head_dim, bit_width; max_seq_len=5)

        for t in 1:10
            K = randn(head_dim, n_heads)
            V = randn(head_dim, n_heads)
            compress_kv!(cache, K, V)
        end

        @test cache.current_len == 5
    end

    @testset "KV Cache: memory usage" begin
        n_heads = 8
        head_dim = 64
        bit_width = 3

        cache = KVCache(n_heads, head_dim, bit_width; max_seq_len=1000)

        for t in 1:100
            K = randn(head_dim, n_heads)
            V = randn(head_dim, n_heads)
            compress_kv!(cache, K, V)
        end

        quant_mem = memory_usage(cache)
        fp16_mem = fp16_memory_usage(cache)

        # Quantized should use less memory than FP16
        @test quant_mem < fp16_mem
    end

    @testset "NN Search: index and search" begin
        d = 32
        N = 200
        bit_width = 4

        # Generate random database
        rng = MersenneTwister(42)
        X_db = randn(rng, d, N)

        # Build index
        index = build_index(X_db, bit_width; seed=UInt64(42), batch_size=50)
        @test index.n_vectors == N
        @test index.dim == d

        # Search
        query = randn(rng, d)
        indices, scores = search(index, query, 10)

        @test length(indices) == 10
        @test length(scores) == 10
        @test issorted(scores, rev=true)

        # All indices should be valid
        @test all(1 .<= indices .<= N)
    end

    @testset "NN Search: batch search" begin
        d = 32
        N = 100
        n_queries = 5
        k = 5

        rng = MersenneTwister(42)
        X_db = randn(rng, d, N)
        queries = randn(rng, d, n_queries)

        index = build_index(X_db, 3; seed=UInt64(42))
        idx_mat, score_mat = batch_search(index, queries, k)

        @test size(idx_mat) == (k, n_queries)
        @test size(score_mat) == (k, n_queries)
    end

    @testset "NN Search: recall improves with bits" begin
        d = 32
        N = 200
        n_queries = 20
        k = 10

        rng = MersenneTwister(42)
        X_db = randn(rng, d, N)
        queries = randn(rng, d, n_queries)

        # Compute ground truth (exact NN by inner product)
        gt = zeros(Int, n_queries)
        for j in 1:n_queries
            ips = X_db' * queries[:, j]
            gt[j] = argmax(ips)
        end

        # Recall should generally improve with more bits
        recalls = Float64[]
        for b in [2, 4]
            index = build_index(X_db, b; seed=UInt64(42))
            r = recall_at_k(index, queries, gt, k)
            push!(recalls, r)
        end

        # Higher bits should give at least as good recall (with high probability)
        @test recalls[2] >= recalls[1] - 0.15  # allow small statistical slack
    end

    @testset "Hadamard rotation mode" begin
        d = 64
        N = 20
        b = 3

        tq = setup(TurboQuantMSE, d, b; seed=UInt64(42), use_hadamard=true)
        X = randn(d, N)

        comp = quantize(tq, X)
        X_hat = dequantize(tq, comp)

        mse = mean(sum((X .- X_hat).^2, dims=1)) / d
        @test mse < 0.5  # Should still work reasonably
    end

end
