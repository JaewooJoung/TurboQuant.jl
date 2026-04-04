#!/usr/bin/env julia
#
# Standalone CLI tool for .tqt compression/decompression
#
# Usage:
#   julia examples/tqt_tool.jl compress   input.txt  output.tqt  [bit_width] [dim]
#   julia examples/tqt_tool.jl decompress input.tqt  output.txt
#   julia examples/tqt_tool.jl info       file.tqt

include(joinpath(@__DIR__, "tqt_file_io.jl"))

function show_tqt_info(path::String)
    open(path, "r") do io
        magic = read(io, 4)
        if magic != TQT_MAGIC
            println("Error: Not a valid .tqt file")
            return
        end

        original_size = read(io, UInt64)
        dim           = Int(read(io, UInt32))
        bit_width     = Int(read(io, UInt8))
        n_vectors     = Int(read(io, UInt64))
        padding_bytes = Int(read(io, UInt32))
        seed          = read(io, UInt64)
        n_centroids   = Int(read(io, UInt32))

        compressed_size = filesize(path)
        ratio = round(original_size / compressed_size, digits=2)

        println("TQT File Info: $path")
        println("  Original size:  $original_size bytes")
        println("  Compressed size: $compressed_size bytes")
        println("  Ratio:          $(ratio)×")
        println("  Vector dim:     $dim")
        println("  Bit-width:      $bit_width")
        println("  Vectors:        $n_vectors")
        println("  Padding:        $padding_bytes bytes")
        println("  Seed:           $seed")
        println("  Centroids:      $n_centroids")

        centroids = [read(io, Float64) for _ in 1:n_centroids]
        println("  Centroid values: $(round.(centroids, digits=6))")

        # Estimate effective bits per byte of original data
        bits_per_byte = compressed_size * 8.0 / original_size
        println("  Bits/byte:      $(round(bits_per_byte, digits=2)) (vs 8.0 uncompressed)")
    end
end

function main()
    if length(ARGS) < 1
        println("""
        TurboQuant File Tool (.tqt)

        Usage:
          julia tqt_tool.jl compress   <input> <output.tqt> [bit_width=4] [dim=64]
          julia tqt_tool.jl decompress <input.tqt> <output>
          julia tqt_tool.jl info       <file.tqt>

        Examples:
          julia tqt_tool.jl compress   photo.png photo.tqt 4 128
          julia tqt_tool.jl decompress photo.tqt photo_restored.png
          julia tqt_tool.jl info       photo.tqt

        Note: TurboQuant is a LOSSY quantizer. Reconstructed files will have
        small byte-level errors. Higher bit-width = better quality.
        Recommended: bit_width=4-8 for text, 2-4 for binary/media data.
        """)
        return
    end

    cmd = lowercase(ARGS[1])

    if cmd == "compress"
        if length(ARGS) < 3
            println("Usage: compress <input> <output.tqt> [bit_width=4] [dim=64]")
            return
        end
        input_path = ARGS[2]
        output_path = ARGS[3]
        bit_width = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 4
        dim = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 64

        compress_file(input_path, output_path; bit_width=bit_width, dim=dim)

    elseif cmd == "decompress"
        if length(ARGS) < 3
            println("Usage: decompress <input.tqt> <output>")
            return
        end
        tqt_path = ARGS[2]
        output_path = ARGS[3]

        decompress_file(tqt_path, output_path)

    elseif cmd == "info"
        if length(ARGS) < 2
            println("Usage: info <file.tqt>")
            return
        end
        show_tqt_info(ARGS[2])

    else
        println("Unknown command: $cmd")
        println("Use: compress, decompress, or info")
    end
end

main()
