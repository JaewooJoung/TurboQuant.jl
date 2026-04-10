[English](README.md) | [한국어](README.ko-KR.md) | [Svenska](README.sv-SE.md) | [简体中文](README.zh-CN.md)

# TurboQuant.jl

[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://github.com/JaewooJoung/TurboQuant.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JaewooJoung.github.io/TurboQuant.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JaewooJoung.github.io/TurboQuant.jl/dev)
[![Build Status](https://github.com/JaewooJoung/TurboQuant.jl/workflows/CI/badge.svg)](https://github.com/JaewooJoung/TurboQuant.jl/actions)
[![Coverage](https://codecov.io/gh/JaewooJoung/TurboQuant.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JaewooJoung/TurboQuant.jl)
[![License](https://img.shields.io/github/license/JaewooJoung/TurboQuant.jl)](https://github.com/JaewooJoung/TurboQuant.jl/blob/main/LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-blueviolet?logo=julia)](https://julialang.org/)


En Julia-implementation av avancerad vektorkvantisering baserad på [TurboQuant (arXiv:2504.19874v1)](https://arxiv.org/abs/2504.19874v1). Uppnår nära informationsteoretiskt optimal komprimering utan kalibrerings­data eller kodboksinlärning.

## Huvudfunktioner

- **Online / dataoberoende** — ingen förbehandling, kalibrering eller kodboksinlärning krävs
- **Nära optimal distorsion** — inom 2,72× av Shannons undre gräns, O(1/4^b)-takt
- **Väntevärdesriktiga inre produkter** — avgörande för attention-mekanismer och närmaste-granne-sökning
- **Fullt vektoriserbar** — utformad för GPU/TPU-vänlig körning

## Installation

```julia
using Pkg
Pkg.add("TurboQuant")
```

Eller installera direkt från repositoriet:

```julia
using Pkg
Pkg.add(url="https://github.com/JaewooJoung/TurboQuant.jl")
```

## Snabbstart

### Använda modulen i kod

```julia
using TurboQuant
using LinearAlgebra

# --- MSE-kvantiserare ---
d = 128    # vektordimension
b = 4      # bitar per koordinat
tq = setup(TurboQuantMSE, d, b)

x = randn(d)                          # originalvektor
comp = quantize(tq, x)                # komprimera
x_hat = dequantize(tq, comp)          # dekomprimera
println("MSE: ", sum((x .- x_hat[:,1]).^2) / d)

# --- Batchkvantisering ---
X = randn(d, 1000)                    # 1000 vektorer
comp = quantize(tq, X)
X_hat = dequantize(tq, comp)

# --- Väntevärdesriktig inre-produkt-kvantiserare ---
tq_prod = setup(TurboQuantProd, d, 4)
comp = quantize(tq_prod, X)
y = randn(d)
approx_ips = inner_product(tq_prod, comp, y)   # E[resultat] = X' * y (väntevärdesriktig)
```

### Filkomprimering (.tqt-format)

Komprimera valfri fil till `.tqt`-format och dekomprimera tillbaka:

```julia
include("examples/tqt_file_io.jl")

# Förlustfri komprimering (standard) — exakt rekonstruktion garanterad
compress_file("myfile.txt", "myfile.tqt"; bit_width=4, dim=64, lossless=true)
decompress_file("myfile.tqt", "myfile_restored.txt")
verify_roundtrip("myfile.txt", "myfile_restored.txt")
# → Resultat: LOSSLESS (perfekt rekonstruktion)

# Förlustgivande komprimering — mindre utdata, approximativ rekonstruktion
compress_file("myfile.txt", "myfile.tqt"; bit_width=4, dim=64, lossless=false)
```

**Så fungerar förlustfritt läge**: TurboQuant komprimerar data med förlustgivande
vektorkvantisering och beräknar sedan residualen på bytenivå (original − rekonstruerad)
som lagras med RLE-kodning. Vid dekomprimering läggs residualkorrigeringen till
för exakt rekonstruktion. Högre `bit_width` ger mindre residualer och därmed
bättre total komprimering.

## Exempel

### Kör demon

Demon visar både förlustfria och förlustgivande jämförelser över bitbredder:

```bash
# Använd inbyggd exempeltext
julia examples/example_compress.jl

# Komprimera din egen fil
julia examples/example_compress.jl path/to/myfile.txt

# Anpassad bitbredd (6) och vektordimension (128)
julia examples/example_compress.jl myfile.txt 6 128
```

Exempelutdata:

```
=== Förlustfritt läge: Bitbredds­jämförelse ===
Bitar | Komprimerad | Kvot   | TQT-del  | Residualdel   | Exakt?
------|-------------|--------|----------|---------------|-------
  1   |         ... |  ...×  |      ... |           ... | YES
  4   |         ... |  ...×  |      ... |           ... | YES
  8   |         ... |  ...×  |      ... |           ... | YES

=== Förlustgivande läge: Bitbredds­jämförelse ===
Bitar | Komprimerad | Kvot   | Exakt matchning %
------|-------------|--------|-------------------
  1   |         ... |  ...×  | ~0,7 %
  4   |         ... |  ...×  | ~3,5 %
  8   |         ... |  ...×  | ~43,5 %
```

### CLI-verktyg

Ett fristående kommandoradsverktyg för komprimering / dekomprimering / inspektion:

```bash
# Förlustfri komprimering (standard)
julia examples/tqt_tool.jl compress input.png output.tqt 4 128

# Förlustgivande komprimering
julia examples/tqt_tool.jl compress input.png output.tqt 2 64 --lossy

# Dekomprimera (auto-detekterar förlustfri/förlustgivande från header)
julia examples/tqt_tool.jl decompress output.tqt restored.png

# Inspektera en .tqt-filheader
julia examples/tqt_tool.jl info output.tqt
```

`info`-utdata visar läge (förlustfri/förlustgivande), originalstorlek, vektordimensioner, bitbredd, kodbokscentroider, residualstorlek och effektiva bitar per byte.

## Modulkomponenter

### Kodbok (`src/codebook.jl`)

Lloyd-Max optimal skalarkvantiserare för betafördelningen som uppstår vid projektion av enhetsvektorer via slumpmässig rotation.

```julia
cb = solve_codebook(128, 4)       # dimension=128, 4-bitars
# cb.centroids  — 16 optimala centroidvärden
# cb.boundaries — 15 beslutsgränser
```

### Rotation (`src/rotation.jl`)

Två rotationslägen:

| Läge | Komplexitet | Användning |
|---|---|---|
| `RandomRotation(d, seed)` | O(d²) | Exakt ortogonal, bäst kvalitet |
| `HadamardRotation(d, seed)` | O(d log d) | Strukturerad, snabbare för stort d |

```julia
R = RandomRotation(128, UInt64(42))
y = rotate(R, x)          # framåt
x_back = rotate_back(R, y) # invers
```

### MSE-kvantiserare (`src/mse_quantizer.jl`)

MSE-optimal vektorkvantiserare: slumpmässig rotation + koordinatvis Lloyd-Max.

```julia
tq = setup(TurboQuantMSE, d, b; use_hadamard=false)
comp = quantize(tq, X)         # returnerar CompressedVectorMSE
X_hat = dequantize(tq, comp)
ratio = compression_ratio(comp)
```

### Inre-produkt-kvantiserare (`src/prod_quantizer.jl`)

Tvåstegspipeline (MSE + QJL) som garanterar väntevärdesriktiga inre produkter.

```julia
tq = setup(TurboQuantProd, d, b)   # b >= 2 krävs
comp = quantize(tq, X)
X_hat = dequantize(tq, comp)

# Beräkna inre produkter utan full dekomprimering
scores = inner_product(tq, comp, query_vector)
```

### KV-cache (`src/kv_cache.jl`)

Drop-in-kvantiserad KV-cache för transformer-attention:

```julia
cache = KVCache(n_heads, head_dim, bit_width; max_seq_len=131072)

# Strömma tokens
for token in tokens
    compress_kv!(cache, K_token, V_token)
end

# Beräkna attention med kvantiserad cache
output = attention_with_quantized_kv(cache, Q)

# Minnesjämförelse
println("Kvantiserad: ", memory_usage(cache), " bytes")
println("FP16:        ", fp16_memory_usage(cache), " bytes")
```

- Nycklar använder `TurboQuantProd` (väntevärdesriktiga inre produkter för attention-poäng)
- Värden använder `TurboQuantMSE` (MSE räcker efter softmax-viktning)
- Automatisk detektering av outlier-huvuden med extra bitallokering
- FIFO-utkastning vid `max_seq_len`

### Närmaste-granne-sökning (`src/nn_search.jl`)

Approximativ närmaste-granne-sökning utan träningstid:

```julia
# Bygg index (omedelbart — inget k-means)
index = build_index(database_vectors, 4; use_hadamard=true)

# Fråga
indices, scores = search(index, query, k)

# Batchfråga
idx_mat, score_mat = batch_search(index, queries, k)

# Utvärdera recall
recall = recall_at_k(index, queries, ground_truth, k)
```

## .tqt-filformat (v3)

Binär layout med bitpackade index:

```
[Header]  38 byte
  Magic:          "TQT\x03"          (4 byte)
  Flaggor:        UInt8               (1 byte, bit 0 = förlustfri)
  Originalstorlek: UInt64             (8 byte)
  Dimension:      UInt32              (4 byte)
  Bitbredd:       UInt8               (1 byte)
  Antal vektorer: UInt64              (8 byte)
  Utfyllnad:      UInt32              (4 byte)
  Seed:           UInt64              (8 byte)

[Kodbok]
  Antal centroider: UInt32            (4 byte)
  Centroider:     Float64 × 2^b      (8 × 2^b byte)

[Data] × n_vectors
  Norm:           Float32             (4 byte)
  Index:          ceil(dim×b/8) byte  (bitpackade, b bitar per index)

[Residual — endast förlustfritt läge]
  Residualstorlek: UInt32             (4 byte)
  Residualdata:   variabel            (RLE-kodade zigzag-residualer)
```

**Lagringsberäkning**: Vid `bit_width=2, dim=64` tar varje vektor 4 (norm) + 16 (packade index) = 20 byte istället för 72 byte (8 + 64) i v2. För `bit_width=1` är det bara 4 + 8 = 12 byte per vektor.

## Distorsionsprestanda

| Bitbredd | MSE-distorsion | Kvot till Shannon-gräns |
|---|---|---|
| 1 | 0,36 | ~1,45× |
| 2 | 0,117 | ~1,87× |
| 3 | 0,030 | ~1,92× |
| 4 | 0,009 | ~2,30× |
| Allmänt b | (sqrt(3π)/2) · 4^(−b) | ≤ 2,72× |

## Köra tester

```julia
using Pkg
Pkg.test("TurboQuant")
```

## Parameterguide

| Parameter | Rekommenderat | Anmärkningar |
|---|---|---|
| `bit_width` | 3–4 för kvalitet, 1–2 för aggressiv komprimering | Högre = bättre kvalitet, mindre komprimering |
| `dim` | 64–128 | Måste dela data jämnt; större = bättre statistiska egenskaper |
| `use_hadamard` | `true` för d > 256 | O(d log d) vs O(d²) rotation |
| `seed` | valfritt UInt64 | Samma seed = reproducerbar komprimering |

## Referenser

- Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization for Near-Optimal Quantized Retrieval, KV Cache Quantization, and Beyond.* arXiv:2504.19874v1, April 2025.

## Licens

MIT
