module TurboQuant

using LinearAlgebra
using Random
using Statistics

export TurboQuantMSE, TurboQuantProd, KVCache
export setup, quantize, dequantize, inner_product
export LloydMaxCodebook, solve_codebook
export RandomRotation, HadamardRotation
export compress_kv!, attention_with_quantized_kv

include("codebook.jl")
include("rotation.jl")
include("mse_quantizer.jl")
include("prod_quantizer.jl")
include("kv_cache.jl")
include("nn_search.jl")

end # module
