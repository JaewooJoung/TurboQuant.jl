[English](README.md) | [Español](README.es.md) | [한국어](README.ko-KR.md) | [Svenska](README.sv-SE.md) | [简体中文](README.zh-CN.md)

# TurboQuant.jl

[![JuliaHub](https://juliahub.com/docs/General/TurboQuant/stable/version.svg)](https://juliahub.com/ui/Packages/General/TurboQuant)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-blueviolet?logo=julia)](https://julialang.org/)
[![pkgeval](https://juliahub.com/docs/General/TurboQuant/stable/pkgeval.svg)](https://juliahub.com/ui/Packages/General/TurboQuant)
[![deps](https://juliahub.com/docs/General/TurboQuant/stable/deps.svg)](https://juliahub.com/ui/Packages/General/TurboQuant)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JaewooJoung.github.io/TurboQuant.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JaewooJoung.github.io/TurboQuant.jl/dev)
[![Build Status](https://github.com/JaewooJoung/TurboQuant.jl/workflows/CI/badge.svg)](https://github.com/JaewooJoung/TurboQuant.jl/actions)
[![Coverage](https://codecov.io/gh/JaewooJoung/TurboQuant.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JaewooJoung/TurboQuant.jl)
[![License](https://img.shields.io/github/license/JaewooJoung/TurboQuant.jl)](https://github.com/JaewooJoung/TurboQuant.jl/blob/main/LICENSE)
[![Sponsor](https://img.shields.io/badge/Sponsor-PayPal-003087?logo=paypal&logoColor=white)](https://paypal.me/jaewoojoung)


基于 [TurboQuant (arXiv:2504.19874v1)](https://arxiv.org/abs/2504.19874v1) 的高级向量量化 Julia 实现。无需任何校准数据或码本学习，即可实现接近信息论最优的压缩。

## 主要特性

- **在线 / 数据无关** — 无需预处理、校准或码本学习
- **近最优失真** — 在 Shannon 下界的 2.72 倍以内，O(1/4^b) 速率
- **无偏内积** — 对注意力机制和最近邻搜索至关重要
- **完全可向量化** — 专为 GPU/TPU 友好执行而设计

## 安装

```julia
using Pkg
Pkg.add("TurboQuant")
```

或从仓库直接安装：

```julia
using Pkg
Pkg.add(url="https://github.com/JaewooJoung/TurboQuant.jl")
```

## 快速开始

### 在代码中使用模块

```julia
using TurboQuant
using LinearAlgebra

# --- MSE 量化器 ---
d = 128    # 向量维度
b = 4      # 每坐标比特数
tq = setup(TurboQuantMSE, d, b)

x = randn(d)                          # 原始向量
comp = quantize(tq, x)                # 压缩
x_hat = dequantize(tq, comp)          # 解压缩
println("MSE: ", sum((x .- x_hat[:,1]).^2) / d)

# --- 批量量化 ---
X = randn(d, 1000)                    # 1000 个向量
comp = quantize(tq, X)
X_hat = dequantize(tq, comp)

# --- 无偏内积量化器 ---
tq_prod = setup(TurboQuantProd, d, 4)
comp = quantize(tq_prod, X)
y = randn(d)
approx_ips = inner_product(tq_prod, comp, y)   # E[结果] = X' * y（无偏）
```

### 文件压缩（.tqt 格式）

将任意文件压缩为 `.tqt` 格式并解压还原：

```julia
include("examples/tqt_file_io.jl")

# 无损压缩（默认）— 保证精确重建
compress_file("myfile.txt", "myfile.tqt"; bit_width=4, dim=64, lossless=true)
decompress_file("myfile.tqt", "myfile_restored.txt")
verify_roundtrip("myfile.txt", "myfile_restored.txt")
# → 结果：LOSSLESS（完美重建）

# 有损压缩 — 更小的输出，近似重建
compress_file("myfile.txt", "myfile.tqt"; bit_width=4, dim=64, lossless=false)
```

**无损模式工作原理**：TurboQuant 使用有损向量量化压缩数据，然后计算字节级
残差（原始 - 重建），并以 RLE 编码存储。解压时加回残差校正，实现精确重建。
`bit_width` 越高，残差越小，整体压缩效果越好。

## 示例

### 运行演示

演示展示了不同比特宽度下的无损和有损对比：

```bash
# 使用内置示例文本
julia examples/example_compress.jl

# 压缩您自己的文件
julia examples/example_compress.jl path/to/myfile.txt

# 自定义比特宽度（6）和向量维度（128）
julia examples/example_compress.jl myfile.txt 6 128
```

输出示例：

```
=== 无损模式：比特宽度对比 ===
比特 | 压缩大小    | 比率   | TQT 部分 | 残差部分       | 精确？
-----|------------|--------|----------|---------------|-------
  1  |        ... |  ...×  |      ... |           ... | YES
  4  |        ... |  ...×  |      ... |           ... | YES
  8  |        ... |  ...×  |      ... |           ... | YES

=== 有损模式：比特宽度对比 ===
比特 | 压缩大小    | 比率   | 精确匹配 %
-----|------------|--------|----------
  1  |        ... |  ...×  | ~0.7%
  4  |        ... |  ...×  | ~3.5%
  8  |        ... |  ...×  | ~43.5%
```

### CLI 工具

用于压缩 / 解压 / 检查的独立命令行工具：

```bash
# 无损压缩（默认）
julia examples/tqt_tool.jl compress input.png output.tqt 4 128

# 有损压缩
julia examples/tqt_tool.jl compress input.png output.tqt 2 64 --lossy

# 解压（从头部自动检测无损/有损）
julia examples/tqt_tool.jl decompress output.tqt restored.png

# 检查 .tqt 文件头
julia examples/tqt_tool.jl info output.tqt
```

`info` 输出显示模式（无损/有损）、原始大小、向量维度、比特宽度、码本质心、残差大小和有效每字节比特数。

## 模块组件

### 码本（`src/codebook.jl`）

通过随机旋转投影单位向量时产生的 Beta 分布的 Lloyd-Max 最优标量量化器。

```julia
cb = solve_codebook(128, 4)       # 维度=128，4比特
# cb.centroids  — 16 个最优质心值
# cb.boundaries — 15 个决策边界
```

### 旋转（`src/rotation.jl`）

两种旋转模式：

| 模式 | 复杂度 | 用途 |
|---|---|---|
| `RandomRotation(d, seed)` | O(d²) | 精确正交，最佳质量 |
| `HadamardRotation(d, seed)` | O(d log d) | 结构化，大 d 时更快 |

```julia
R = RandomRotation(128, UInt64(42))
y = rotate(R, x)          # 正向
x_back = rotate_back(R, y) # 逆向
```

### MSE 量化器（`src/mse_quantizer.jl`）

MSE 最优向量量化器：随机旋转 + 逐坐标 Lloyd-Max。

```julia
tq = setup(TurboQuantMSE, d, b; use_hadamard=false)
comp = quantize(tq, X)         # 返回 CompressedVectorMSE
X_hat = dequantize(tq, comp)
ratio = compression_ratio(comp)
```

### 内积量化器（`src/prod_quantizer.jl`）

保证无偏内积的两阶段流水线（MSE + QJL）。

```julia
tq = setup(TurboQuantProd, d, b)   # b >= 2 必需
comp = quantize(tq, X)
X_hat = dequantize(tq, comp)

# 无需完整解压即可计算内积
scores = inner_product(tq, comp, query_vector)
```

### KV 缓存（`src/kv_cache.jl`）

Transformer 注意力的即插即用量化 KV 缓存：

```julia
cache = KVCache(n_heads, head_dim, bit_width; max_seq_len=131072)

# 流式处理 token
for token in tokens
    compress_kv!(cache, K_token, V_token)
end

# 使用量化缓存计算注意力
output = attention_with_quantized_kv(cache, Q)

# 内存对比
println("量化：  ", memory_usage(cache), " bytes")
println("FP16：  ", fp16_memory_usage(cache), " bytes")
```

- Key 使用 `TurboQuantProd`（注意力分数的无偏内积）
- Value 使用 `TurboQuantMSE`（softmax 加权后 MSE 即可）
- 自动检测异常头并分配额外比特
- 在 `max_seq_len` 处 FIFO 驱逐

### 最近邻搜索（`src/nn_search.jl`）

零训练时间的近似最近邻搜索：

```julia
# 构建索引（即时 — 无需 k-means）
index = build_index(database_vectors, 4; use_hadamard=true)

# 查询
indices, scores = search(index, query, k)

# 批量查询
idx_mat, score_mat = batch_search(index, queries, k)

# 评估召回率
recall = recall_at_k(index, queries, ground_truth, k)
```

## .tqt 文件格式（v3）

带比特打包索引的二进制布局：

```
[文件头]  38 字节
  魔术字：        "TQT\x03"          （4 字节）
  标志位：        UInt8               （1 字节，比特 0 = 无损）
  原始大小：      UInt64              （8 字节）
  维度：          UInt32              （4 字节）
  比特宽度：      UInt8               （1 字节）
  向量数：        UInt64              （8 字节）
  填充：          UInt32              （4 字节）
  种子：          UInt64              （8 字节）

[码本]
  质心数：        UInt32              （4 字节）
  质心：          Float64 × 2^b      （8 × 2^b 字节）

[数据] × n_vectors
  范数：          Float32             （4 字节）
  索引：          ceil(dim×b/8) 字节  （比特打包，每索引 b 比特）

[残差 — 仅无损模式]
  残差大小：      UInt32              （4 字节）
  残差数据：      可变                （RLE 编码的 zigzag 残差）
```

**存储计算**：当 `bit_width=2, dim=64` 时，每个向量占用 4（范数）+ 16（打包索引）= 20 字节，而非 v2 中的 72 字节（8 + 64）。当 `bit_width=1` 时，每个向量仅需 4 + 8 = 12 字节。

## 失真性能

| 比特宽度 | MSE 失真 | 与 Shannon 界的比率 |
|---|---|---|
| 1 | 0.36 | ~1.45× |
| 2 | 0.117 | ~1.87× |
| 3 | 0.030 | ~1.92× |
| 4 | 0.009 | ~2.30× |
| 一般 b | (sqrt(3π)/2) · 4^(−b) | ≤ 2.72× |

## 运行测试

```julia
using Pkg
Pkg.test("TurboQuant")
```

## 参数指南

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `bit_width` | 质量优先：3-4，激进压缩：1-2 | 越高 = 质量越好，压缩率越低 |
| `dim` | 64-128 | 必须能整除数据；越大统计特性越好 |
| `use_hadamard` | d > 256 时用 `true` | O(d log d) vs O(d²) 旋转 |
| `seed` | 任意 UInt64 | 相同种子 = 可复现的压缩 |

## 参考文献

- Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization for Near-Optimal Quantized Retrieval, KV Cache Quantization, and Beyond.* arXiv:2504.19874v1, April 2025.

## 许可证

MIT
