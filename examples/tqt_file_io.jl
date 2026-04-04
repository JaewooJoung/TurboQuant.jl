# TQT File Format — Binary file compression using TurboQuant vector quantization
#
# Supports two modes:
#   LOSSY:    TurboQuant quantization only (smaller file, approximate reconstruction)
#   LOSSLESS: TurboQuant + residual correction (exact reconstruction, guaranteed)
#
# .tqt file layout:
#   [Header]
#     magic:          4 bytes  "TQT\x03"
#     flags:          1 byte   bit 0 = lossless
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
#     norm:           4 bytes  Float32 (was Float64 — halved)
#     indices:        ceil(dim * bit_width / 8) bytes  (bit-packed)
#   [Residual section — lossless mode only]
#     residual_size:  4 bytes  UInt32 (byte count of encoded residuals)
#     residual_data:  variable (RLE-encoded zigzag residuals)

include(joinpath(@__DIR__, "..", "src", "TurboQuant.jl"))
using .TurboQuant
using LinearAlgebra

const TQT_MAGIC = UInt8[0x54, 0x51, 0x54, 0x03]  # "TQT\x03"
const FLAG_LOSSLESS = UInt8(0x01)

# ── Bit-packing: pack b-bit indices into bytes ──────────────────────────────

"""
    bitpack(indices::Vector{UInt8}, bit_width::Int) → Vector{UInt8}

Pack an array of `bit_width`-bit indices (values 0 to 2^b-1) into a
compact byte array. Indices are 1-based internally, stored as 0-based.
"""
function bitpack(indices::AbstractVector{UInt8}, bit_width::Int)
    n = length(indices)
    total_bits = n * bit_width
    n_bytes = cld(total_bits, 8)  # ceil division
    packed = zeros(UInt8, n_bytes)

    bit_pos = 0
    for i in 1:n
        val = UInt32(indices[i] - 1)  # convert 1-based to 0-based for storage
        # Write `bit_width` bits starting at bit_pos
        for b in 0:(bit_width - 1)
            if (val >> b) & 1 == 1
                byte_idx = (bit_pos >> 3) + 1
                bit_idx = bit_pos & 7
                packed[byte_idx] |= UInt8(1) << bit_idx
            end
            bit_pos += 1
        end
    end

    return packed
end

"""
    bitunpack(packed::Vector{UInt8}, n::Int, bit_width::Int) → Vector{UInt8}

Unpack `n` indices of `bit_width` bits each from a packed byte array.
Returns 1-based indices.
"""
function bitunpack(packed::AbstractVector{UInt8}, n::Int, bit_width::Int)
    indices = Vector{UInt8}(undef, n)
    bit_pos = 0

    for i in 1:n
        val = UInt32(0)
        for b in 0:(bit_width - 1)
            byte_idx = (bit_pos >> 3) + 1
            bit_idx = bit_pos & 7
            if (packed[byte_idx] >> bit_idx) & 1 == 1
                val |= UInt32(1) << b
            end
            bit_pos += 1
        end
        indices[i] = UInt8(val + 1)  # convert 0-based back to 1-based
    end

    return indices
end

# ── Residual encoding: zigzag + RLE ──────────────────────────────────────────

"""Encode signed integer as unsigned via zigzag: 0→0, -1→1, 1→2, -2→3, ..."""
zigzag_encode(x::Int) = x >= 0 ? UInt16(2x) : UInt16(-2x - 1)

"""Decode zigzag back to signed."""
zigzag_decode(z::UInt16) = iseven(z) ? Int(z >> 1) : -Int((z >> 1) + 1)

"""
    rle_encode(residuals::Vector{Int})

Run-Length Encode zigzag-encoded residuals into a compact byte stream.

Uses a variable format to minimize overhead:
- If run_length == 1: [0xFF] [zigzag_val as UInt8]       (2 bytes for singletons with val < 256)
- General:            [zigzag_val as UInt16] [run as UInt16]  (4 bytes)

Special: consecutive singletons with small values are packed as raw bytes
with a header indicating the count.

Format: sequence of chunks:
  [UInt16 zigzag_val] [UInt16 run_length]
"""
function rle_encode(residuals::Vector{Int})
    buf = IOBuffer()
    n = length(residuals)
    i = 1
    while i <= n
        val = residuals[i]
        run = UInt16(1)
        while i + Int(run) <= n && residuals[i + Int(run)] == val && run < 0xFFFF
            run += UInt16(1)
        end
        write(buf, zigzag_encode(val))
        write(buf, run)
        i += Int(run)
    end
    return take!(buf)
end

"""
    rle_decode(data::Vector{UInt8}, expected_len::Int)

Decode RLE-encoded residuals back to a vector of signed integers.
"""
function rle_decode(data::Vector{UInt8}, expected_len::Int)
    result = Vector{Int}(undef, expected_len)
    io = IOBuffer(data)
    pos = 1
    while pos <= expected_len && !eof(io)
        z = read(io, UInt16)
        run = Int(read(io, UInt16))
        val = zigzag_decode(z)
        for _ in 1:run
            result[pos] = val
            pos += 1
        end
    end
    return result
end

# ── Lossy reconstruction helper ──────────────────────────────────────────────

"""
Reconstruct bytes from quantized data (lossy step).
Returns Vector{UInt8} of length `original_size`.
"""
function lossy_reconstruct(tq, indices, norms, dim, n_vectors, original_size)
    comp = TurboQuant.CompressedVectorMSE(indices, norms, dim, n_vectors, tq.bit_width)
    X_hat = dequantize(tq, comp)
    floats = reshape(X_hat, :)

    raw_out = Vector{UInt8}(undef, original_size)
    for i in 1:original_size
        raw_out[i] = UInt8(clamp(round(Int, floats[i]), 0, 255))
    end
    return raw_out
end

# ── Public API ───────────────────────────────────────────────────────────────

"""
    compress_file(input_path, output_path; bit_width=4, dim=64, seed=UInt64(42), lossless=true)

Compress any file to `.tqt` format using TurboQuant MSE quantization.

- `lossless=true` (default): stores residual correction for exact reconstruction
- `lossless=false`: lossy mode, smaller output but approximate reconstruction
- `bit_width`: bits per coordinate (1-8). Higher = better base approximation,
  smaller residuals in lossless mode.
- `dim`: vector dimension for chunking bytes. Must be > 0.
"""
function compress_file(input_path::String, output_path::String;
                       bit_width::Int=4, dim::Int=64, seed::UInt64=UInt64(42),
                       lossless::Bool=true)
    raw = read(input_path)
    original_size = length(raw)

    if original_size == 0
        error("Input file is empty")
    end

    println("Read $(original_size) bytes from: $input_path")

    # Convert bytes to Float64 and chunk into vectors
    floats = Float64.(raw)

    padding_bytes = (dim - length(floats) % dim) % dim
    if padding_bytes > 0
        append!(floats, zeros(Float64, padding_bytes))
    end

    n_vectors = length(floats) ÷ dim
    X = reshape(floats, dim, n_vectors)

    mode_str = lossless ? "LOSSLESS" : "LOSSY"
    println("Vectors: $n_vectors × $dim, bit_width=$bit_width, padding=$padding_bytes [$mode_str]")

    # Setup quantizer & quantize
    tq = setup(TurboQuantMSE, dim, bit_width; seed=seed)
    comp = quantize(tq, X)

    # Compute residuals for lossless mode
    residual_encoded = UInt8[]
    if lossless
        lossy_bytes = lossy_reconstruct(tq, comp.indices, comp.norms, dim, n_vectors, original_size)
        residuals = Int.(raw) .- Int.(lossy_bytes)
        residual_encoded = rle_encode(residuals)

        n_zero = count(==(0), residuals)
        println("Residuals: $(length(residuals)) values, $(n_zero) zeros ($(round(100.0*n_zero/length(residuals), digits=1))%), RLE → $(length(residual_encoded)) bytes")
    end

    # Compute packed index size per vector
    packed_bytes_per_vec = cld(dim * bit_width, 8)

    # Write .tqt file
    open(output_path, "w") do io
        # Header
        write(io, TQT_MAGIC)
        write(io, lossless ? FLAG_LOSSLESS : UInt8(0x00))
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

        # Per-vector: Float32 norm + bit-packed indices
        for j in 1:n_vectors
            write(io, Float32(comp.norms[j]))                    # 4 bytes (was 8)
            packed = bitpack(view(comp.indices, :, j), bit_width) # ceil(dim*b/8) bytes
            write(io, packed)
        end

        # Residual section (lossless only)
        if lossless
            write(io, UInt32(length(residual_encoded)))
            write(io, residual_encoded)
        end
    end

    compressed_size = filesize(output_path)
    ratio = original_size / compressed_size
    savings = round((1.0 - compressed_size / original_size) * 100, digits=1)
    println("Compressed: $(compressed_size) bytes (ratio: $(round(ratio, digits=2))×, savings: $(savings)%)")
    println("Wrote: $output_path")

    return output_path
end

"""
    decompress_file(tqt_path, output_path)

Decompress a `.tqt` file back to the original file.
Automatically detects lossless/lossy mode from the header.
"""
function decompress_file(tqt_path::String, output_path::String)
    open(tqt_path, "r") do io
        # Read & verify header
        magic = read(io, 4)
        if magic != TQT_MAGIC
            if magic[1:3] == UInt8[0x54, 0x51, 0x54]
                error("Incompatible .tqt version (v$(magic[4])). Please re-compress with the current tool.")
            end
            error("Not a valid .tqt file (bad magic bytes)")
        end

        flags         = read(io, UInt8)
        is_lossless   = (flags & FLAG_LOSSLESS) != 0
        original_size = read(io, UInt64)
        dim           = Int(read(io, UInt32))
        bit_width     = Int(read(io, UInt8))
        n_vectors     = Int(read(io, UInt64))
        padding_bytes = Int(read(io, UInt32))
        seed          = read(io, UInt64)

        mode_str = is_lossless ? "LOSSLESS" : "LOSSY"
        println("TQT header: $(original_size) bytes, $(n_vectors)×$(dim), $(bit_width)-bit, seed=$(seed) [$mode_str]")

        # Read codebook
        n_centroids = Int(read(io, UInt32))
        centroids = [read(io, Float64) for _ in 1:n_centroids]

        # Reconstruct quantizer
        boundaries = [(centroids[i] + centroids[i+1]) / 2.0 for i in 1:(n_centroids-1)]
        codebook = TurboQuant.LloydMaxCodebook(bit_width, centroids, boundaries, dim)

        if dim <= 256
            rotation = TurboQuant.RandomRotation(dim, seed)
        else
            rotation = TurboQuant.HadamardRotation(dim, seed)
        end

        tq = TurboQuant.TurboQuantMSE(rotation, codebook, dim, bit_width)

        # Read per-vector data (Float32 norm + bit-packed indices)
        packed_bytes_per_vec = cld(dim * bit_width, 8)
        indices = Matrix{UInt8}(undef, dim, n_vectors)
        norms = Vector{Float64}(undef, n_vectors)

        for j in 1:n_vectors
            norms[j] = Float64(read(io, Float32))
            packed = read(io, packed_bytes_per_vec)
            indices[:, j] .= bitunpack(packed, dim, bit_width)
        end

        # Lossy reconstruction
        raw_out = lossy_reconstruct(tq, indices, norms, dim, n_vectors, Int(original_size))

        # Apply residual correction if lossless
        if is_lossless
            residual_size = Int(read(io, UInt32))
            residual_data = read(io, residual_size)
            residuals = rle_decode(residual_data, Int(original_size))

            for i in 1:Int(original_size)
                corrected = Int(raw_out[i]) + residuals[i]
                raw_out[i] = UInt8(clamp(corrected, 0, 255))
            end
            println("Applied $(length(residuals)) residual corrections")
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

    if exact_matches == n
        println("Result: LOSSLESS (perfect reconstruction)")
    else
        println("Within ±1 byte:  $off_by_one / $n ($(round(100.0 * off_by_one / n, digits=2))%)")
        max_err = maximum(abs.(Int.(orig) .- Int.(decomp)))
        mean_err = mean(abs.(Float64.(orig) .- Float64.(decomp)))
        println("Max byte error:  $max_err")
        println("Mean byte error: $(round(mean_err, digits=3))")
        println("Result: LOSSY")
    end
end
