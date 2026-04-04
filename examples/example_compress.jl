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

    # Compress
    println("\n--- Compressing ---")
    compress_file(input_path, tqt_path; bit_width=bit_width, dim=dim)

    # Decompress
    println("\n--- Decompressing ---")
    decompress_file(tqt_path, out_path)

    # Verify
    verify_roundtrip(input_path, out_path)

    # Show file sizes
    println("\n=== File Sizes ===")
    orig_size = filesize(input_path)
    comp_size = filesize(tqt_path)
    rest_size = filesize(out_path)
    println("Original:     $orig_size bytes")
    println("Compressed:   $comp_size bytes  ($(round(comp_size/orig_size*100, digits=1))% of original)")
    println("Restored:     $rest_size bytes")

    # Try different bit-widths for comparison
    println("\n=== Bit-width Comparison ===")
    println("Bits | Compressed Size | Ratio  | Exact Match %")
    println("-----|-----------------|--------|-------------")
    for b in [1, 2, 3, 4, 6, 8]
        tqt_tmp = base * ".tmp_b$(b).tqt"
        out_tmp = base * ".tmp_b$(b).out"
        try
            compress_file(input_path, tqt_tmp; bit_width=b, dim=dim)
            decompress_file(tqt_tmp, out_tmp)

            orig = read(input_path)
            decomp = read(out_tmp)
            exact = sum(orig .== decomp)
            pct = round(100.0 * exact / length(orig), digits=1)
            cs = filesize(tqt_tmp)
            ratio = round(orig_size / cs, digits=2)

            println("  $b   | $(lpad(cs, 15)) | $(lpad(ratio, 5))× | $(pct)%")
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
