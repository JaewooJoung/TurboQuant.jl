#!/usr/bin/env julia
#
# Example: Compress and decompress a file using TurboQuant (.tqt format)
#
# Usage:
#   julia examples/example_compress.jl                          # uses built-in sample
#   julia examples/example_compress.jl myfile.txt               # compress your file
#   julia examples/example_compress.jl myfile.txt 4 64          # custom bit_width & dim

include(joinpath(@__DIR__, "tqt_file_io.jl"))
using Statistics

function create_sample_file(path::String)
    text = """
    The quick brown fox jumps over the lazy dog.
    Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Pack my box with five dozen liquor jugs.
    How vexingly quick daft zebras jump!
    0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz
    The five boxing wizards jump quickly.
    """
    write(path, text)
    println("Created sample file: $path ($(filesize(path)) bytes)")
    return path
end

function main()
    println("=" ^ 60)
    println("  TurboQuant File Compression Example (.tqt)")
    println("=" ^ 60)
    println()

    # Parse arguments
    if length(ARGS) >= 1
        input_path = ARGS[1]
        if !isfile(input_path)
            println("Error: File not found: $input_path")
            return
        end
    else
        input_path = joinpath(@__DIR__, "sample.txt")
        create_sample_file(input_path)
    end

    bit_width = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
    dim       = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 64

    # Paths
    base = splitext(input_path)[1]
    tqt_path = base * ".tqt"
    out_path = base * ".restored" * splitext(input_path)[2]

    # ── Lossless compression (default) ──
    println("\n--- Lossless Compress ---")
    compress_file(input_path, tqt_path; bit_width=bit_width, dim=dim, lossless=true)

    println("\n--- Lossless Decompress ---")
    decompress_file(tqt_path, out_path)
    verify_roundtrip(input_path, out_path)

    # Clean up
    rm(tqt_path, force=true)
    rm(out_path, force=true)

    # ── Bit-width comparison: lossless mode ──
    orig_size = filesize(input_path)

    println("\n\n=== Lossless Mode: Bit-width Comparison ===")
    println("Bits | Compressed | Ratio  | TQT part | Residual part | Exact?")
    println("-----|------------|--------|----------|---------------|-------")
    for b in [1, 2, 3, 4, 6, 8]
        tqt_tmp = base * ".tmp_b$(b).tqt"
        out_tmp = base * ".tmp_b$(b).out"
        try
            compress_file(input_path, tqt_tmp; bit_width=b, dim=dim, lossless=true)
            decompress_file(tqt_tmp, out_tmp)

            orig = read(input_path)
            decomp = read(out_tmp)
            exact = orig == decomp ? "YES" : "NO"

            cs = filesize(tqt_tmp)
            ratio = round(orig_size / cs, digits=2)

            # Estimate TQT base vs residual size
            # Header + codebook + per-vector data (Float32 norm + packed indices)
            n_vec = (length(orig) + dim - 1) ÷ dim
            n_centroids = 1 << b
            packed_per_vec = cld(dim * b, 8)
            tqt_base = 38 + 4 + n_centroids * 8 + n_vec * (4 + packed_per_vec)
            residual_part = cs - tqt_base

            println("  $b   | $(lpad(cs, 10)) | $(lpad(ratio, 5))× | $(lpad(tqt_base, 8)) | $(lpad(residual_part, 13)) | $exact")
        catch e
            println("  $b   | error: $e")
        finally
            rm(tqt_tmp, force=true)
            rm(out_tmp, force=true)
        end
    end

    # ── Lossy comparison ──
    println("\n=== Lossy Mode: Bit-width Comparison ===")
    println("Bits | Compressed | Ratio  | Exact Match %")
    println("-----|------------|--------|-------------")
    for b in [1, 2, 3, 4, 6, 8]
        tqt_tmp = base * ".tmp_b$(b).tqt"
        out_tmp = base * ".tmp_b$(b).out"
        try
            compress_file(input_path, tqt_tmp; bit_width=b, dim=dim, lossless=false)
            decompress_file(tqt_tmp, out_tmp)

            orig = read(input_path)
            decomp = read(out_tmp)
            exact = sum(orig .== decomp)
            pct = round(100.0 * exact / length(orig), digits=1)
            cs = filesize(tqt_tmp)
            ratio = round(orig_size / cs, digits=2)

            println("  $b   | $(lpad(cs, 10)) | $(lpad(ratio, 5))× | $(pct)%")
        catch e
            println("  $b   | error: $e")
        finally
            rm(tqt_tmp, force=true)
            rm(out_tmp, force=true)
        end
    end

    println("\nDone!")
end

main()
