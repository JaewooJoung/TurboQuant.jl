# TQT File Format — Binary file compression using TurboQuant vector quantization
#
# .tqt file layout:
#   [Header]
#     magic:          4 bytes  "TQT\x01"
#     original_size:  8 bytes  UInt64 (original file size in bytes)
#     dim:            4 bytes  UInt32 (vector dimension used)
#     bit_width:      1 byte   UInt8
#     n_vectors:      8 bytes  UInt64
#     padding_bytes:  4 bytes  UInt32 (zero-padding added to last vector)
#     seed:           8 bytes  UInt64 (rotation seed for reproducibility)
#   [Codebook]
#     n_centroids:    4 bytes  UInt32
#     centroids:      n_centroids × 8 bytes  Float64[]
#   [Per-vector data] × n_vectors
#     norm:           8 bytes  Float64
#     indices:        dim bytes UInt8[] (quantization indices)

include(joinpath(@__DIR__, "..", "src", "TurboQuant.jl"))
using .TurboQuant
using LinearAlgebra

const TQT_MAGIC = UInt8[0x54, 0x51, 0x54, 0x01]  # "TQT\x01"

"""
    compress_file(input_path, output_path; bit_width=4, dim=64, seed=UInt64(42))

Compress any file to `.tqt` format using TurboQuant MSE quantization.

The file bytes are chunked into vectors of length `dim`, each normalized and
quantized. The norm is stored separately for faithful reconstruction.
"""
function compress_file(input_path::String, output_path::String;
                       bit_width::Int=4, dim::Int=64, seed::UInt64=UInt64(42))
    # Read the entire file as raw bytes
    raw = read(input_path)
    original_size = length(raw)

    if original_size == 0
        error("Input file is empty")
    end

    println("Read $(original_size) bytes from: $input_path")

    # Convert bytes to Float64 and chunk into vectors of length `dim`
    floats = Float64.(raw)

    # Pad to make length a multiple of dim
    padding_bytes = (dim - length(floats) % dim) % dim
    if padding_bytes > 0
        append!(floats, zeros(Float64, padding_bytes))
    end

    n_vectors = length(floats) ÷ dim
    X = reshape(floats, dim, n_vectors)  # (dim, N)

    println("Vectors: $n_vectors × $dim, bit_width=$bit_width, padding=$padding_bytes")

    # Setup quantizer
    tq = setup(TurboQuantMSE, dim, bit_width; seed=seed)

    # Quantize
    comp = quantize(tq, X)

    # Write .tqt file
    open(output_path, "w") do io
        # Header
        write(io, TQT_MAGIC)
        write(io, UInt64(original_size))
        write(io, UInt32(dim))
        write(io, UInt8(bit_width))
        write(io, UInt64(n_vectors))
        write(io, UInt32(padding_bytes))
        write(io, UInt64(seed))

        # Codebook
        n_centroids = length(tq.codebook.centroids)
        write(io, UInt32(n_centroids))
        for c in tq.codebook.centroids
            write(io, Float64(c))
        end

        # Per-vector: norm + indices
        for j in 1:n_vectors
            write(io, Float64(comp.norms[j]))
            write(io, comp.indices[:, j])  # UInt8 array of length dim
        end
    end

    compressed_size = filesize(output_path)
    ratio = original_size / compressed_size
    println("Compressed: $(compressed_size) bytes (ratio: $(round(ratio, digits=2))×)")
    println("Wrote: $output_path")

    return output_path
end

"""
    decompress_file(tqt_path, output_path)

Decompress a `.tqt` file back to the original file.
"""
function decompress_file(tqt_path::String, output_path::String)
    open(tqt_path, "r") do io
        # Read & verify header
        magic = read(io, 4)
        if magic != TQT_MAGIC
            error("Not a valid .tqt file (bad magic bytes)")
        end

        original_size = read(io, UInt64)
        dim           = Int(read(io, UInt32))
        bit_width     = Int(read(io, UInt8))
        n_vectors     = Int(read(io, UInt64))
        padding_bytes = Int(read(io, UInt32))
        seed          = read(io, UInt64)

        println("TQT header: $(original_size) bytes, $(n_vectors)×$(dim), $(bit_width)-bit, seed=$(seed)")

        # Read codebook
        n_centroids = Int(read(io, UInt32))
        centroids = [read(io, Float64) for _ in 1:n_centroids]

        # Reconstruct codebook and quantizer
        boundaries = [(centroids[i] + centroids[i+1]) / 2.0 for i in 1:(n_centroids-1)]
        codebook = TurboQuant.LloydMaxCodebook(bit_width, centroids, boundaries, dim)

        if dim <= 256
            rotation = TurboQuant.RandomRotation(dim, seed)
        else
            rotation = TurboQuant.HadamardRotation(dim, seed)
        end

        tq = TurboQuant.TurboQuantMSE(rotation, codebook, dim, bit_width)

        # Read per-vector data and dequantize
        all_floats = Vector{Float64}(undef, n_vectors * dim)

        indices = Matrix{UInt8}(undef, dim, n_vectors)
        norms = Vector{Float64}(undef, n_vectors)

        for j in 1:n_vectors
            norms[j] = read(io, Float64)
            read!(io, view(indices, :, j))
        end

        # Build compressed struct and dequantize
        comp = TurboQuant.CompressedVectorMSE(indices, norms, dim, n_vectors, bit_width)
        X_hat = dequantize(tq, comp)

        # Flatten back to bytes
        floats = reshape(X_hat, :)

        # Remove padding and convert back to bytes
        total_needed = original_size
        raw_out = Vector{UInt8}(undef, total_needed)

        for i in 1:total_needed
            # Clamp to valid byte range and round
            val = clamp(round(Int, floats[i]), 0, 255)
            raw_out[i] = UInt8(val)
        end

        write(output_path, raw_out)
        println("Decompressed: $(length(raw_out)) bytes")
        println("Wrote: $output_path")
    end

    return output_path
end

"""
    verify_roundtrip(original_path, decompressed_path)

Compare original and decompressed files, reporting byte-level accuracy.
"""
function verify_roundtrip(original_path::String, decompressed_path::String)
    orig = read(original_path)
    decomp = read(decompressed_path)

    if length(orig) != length(decomp)
        println("WARNING: Size mismatch — original=$(length(orig)), decompressed=$(length(decomp))")
        return
    end

    n = length(orig)
    exact_matches = sum(orig .== decomp)
    off_by_one = sum(abs.(Int.(orig) .- Int.(decomp)) .<= 1)

    println("\n=== Roundtrip Verification ===")
    println("File size:       $n bytes")
    println("Exact matches:   $exact_matches / $n ($(round(100.0 * exact_matches / n, digits=2))%)")
    println("Within ±1 byte:  $off_by_one / $n ($(round(100.0 * off_by_one / n, digits=2))%)")

    if exact_matches == n
        println("Result: LOSSLESS (perfect reconstruction)")
    else
        max_err = maximum(abs.(Int.(orig) .- Int.(decomp)))
        mean_err = mean(abs.(Float64.(orig) .- Float64.(decomp)))
        println("Max byte error:  $max_err")
        println("Mean byte error: $(round(mean_err, digits=3))")
        println("Result: LOSSY (this is expected — TurboQuant is a lossy quantizer)")
    end
end
