module TurboQuant

using LinearAlgebra
using Random
using Statistics

export TurboQuantMSE, TurboQuantProd, KVCache
export setup, quantize, dequantize, inner_product
export LloydMaxCodebook, solve_codebook, quantize_scalar, dequantize_scalar, precompute_codebooks
export RandomRotation, HadamardRotation, rotate, rotate_back
export compress_kv!, attention_with_quantized_kv
export CompressedVectorMSE, CompressedVectorProd
export compression_ratio, mse_distortion
export memory_usage, fp16_memory_usage
export build_index, search, batch_search, recall_at_k
export TurboQuantIndex, inner_product_bias

include("codebook.jl")
include("rotation.jl")
include("mse_quantizer.jl")
include("prod_quantizer.jl")
include("kv_cache.jl")
include("nn_search.jl")

end # module
