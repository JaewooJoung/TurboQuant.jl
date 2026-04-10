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

Una implementación en Julia de cuantización vectorial avanzada basada en [TurboQuant (arXiv:2504.19874v1)](https://arxiv.org/abs/2504.19874v1). Logra una compresión cercana al óptimo teórico-informacional sin necesidad de datos de calibración ni aprendizaje de libro de códigos.

## Características principales

- **En línea / independiente de los datos** — no requiere preprocesamiento, calibración ni aprendizaje de libro de códigos
- **Distorsión casi óptima** — dentro de 2,72× del límite inferior de Shannon, tasa O(1/4^b)
- **Productos internos insesgados** — fundamental para mecanismos de atención y búsqueda de vecinos más cercanos
- **Completamente vectorizable** — diseñado para ejecución compatible con GPU/TPU

## Instalación

```julia
using Pkg
Pkg.add("TurboQuant")
```

O instalar directamente desde el repositorio:

```julia
using Pkg
Pkg.add(url="https://github.com/JaewooJoung/TurboQuant.jl")
```

## Inicio rápido

### Uso del módulo en código

```julia
using TurboQuant
using LinearAlgebra

# --- Cuantizador MSE ---
d = 128    # dimensión del vector
b = 4      # bits por coordenada
tq = setup(TurboQuantMSE, d, b)

x = randn(d)                          # vector original
comp = quantize(tq, x)                # comprimir
x_hat = dequantize(tq, comp)          # descomprimir
println("MSE: ", sum((x .- x_hat[:,1]).^2) / d)

# --- Cuantización por lotes ---
X = randn(d, 1000)                    # 1000 vectores
comp = quantize(tq, X)
X_hat = dequantize(tq, comp)

# --- Cuantizador de producto interno insesgado ---
tq_prod = setup(TurboQuantProd, d, 4)
comp = quantize(tq_prod, X)
y = randn(d)
approx_ips = inner_product(tq_prod, comp, y)   # E[resultado] = X' * y (insesgado)
```

### Compresión de archivos (formato .tqt)

Comprimir cualquier archivo al formato `.tqt` y descomprimirlo:

```julia
include("examples/tqt_file_io.jl")

# Compresión sin pérdida (predeterminado) — reconstrucción exacta garantizada
compress_file("myfile.txt", "myfile.tqt"; bit_width=4, dim=64, lossless=true)
decompress_file("myfile.tqt", "myfile_restored.txt")
verify_roundtrip("myfile.txt", "myfile_restored.txt")
# → Resultado: LOSSLESS (reconstrucción perfecta)

# Compresión con pérdida — salida más pequeña, reconstrucción aproximada
compress_file("myfile.txt", "myfile.tqt"; bit_width=4, dim=64, lossless=false)
```

**Cómo funciona el modo sin pérdida**: TurboQuant comprime los datos con
cuantización vectorial con pérdida, luego calcula el residuo a nivel de byte
(original − reconstruido) y lo almacena con codificación RLE. Al descomprimir,
se suma la corrección residual para lograr una reconstrucción exacta. Un
`bit_width` más alto significa residuos más pequeños y, por lo tanto, mejor
compresión general.

## Ejemplos

### Ejecutar la demostración

La demostración muestra comparaciones sin pérdida y con pérdida en distintos anchos de bit:

```bash
# Usar texto de ejemplo incorporado
julia examples/example_compress.jl

# Comprimir su propio archivo
julia examples/example_compress.jl path/to/myfile.txt

# Ancho de bit personalizado (6) y dimensión de vector (128)
julia examples/example_compress.jl myfile.txt 6 128
```

Ejemplo de salida:

```
=== Modo sin pérdida: Comparación de anchos de bit ===
Bits | Comprimido | Ratio  | Parte TQT | Parte residual | ¿Exacto?
-----|------------|--------|-----------|----------------|--------
  1  |        ... |  ...×  |       ... |            ... | SÍ
  4  |        ... |  ...×  |       ... |            ... | SÍ
  8  |        ... |  ...×  |       ... |            ... | SÍ

=== Modo con pérdida: Comparación de anchos de bit ===
Bits | Comprimido | Ratio  | Coincidencia exacta %
-----|------------|--------|---------------------
  1  |        ... |  ...×  | ~0,7%
  4  |        ... |  ...×  | ~3,5%
  8  |        ... |  ...×  | ~43,5%
```

### Herramienta CLI

Una herramienta de línea de comandos independiente para comprimir / descomprimir / inspeccionar:

```bash
# Compresión sin pérdida (predeterminado)
julia examples/tqt_tool.jl compress input.png output.tqt 4 128

# Compresión con pérdida
julia examples/tqt_tool.jl compress input.png output.tqt 2 64 --lossy

# Descomprimir (detecta automáticamente sin pérdida/con pérdida desde el encabezado)
julia examples/tqt_tool.jl decompress output.tqt restored.png

# Inspeccionar el encabezado de un archivo .tqt
julia examples/tqt_tool.jl info output.tqt
```

La salida de `info` muestra el modo (sin pérdida/con pérdida), tamaño original, dimensiones del vector, ancho de bit, centroides del libro de códigos, tamaño del residuo y bits efectivos por byte.

## Componentes del módulo

### Libro de códigos (`src/codebook.jl`)

Cuantizador escalar óptimo de Lloyd-Max para la distribución Beta que surge al proyectar vectores unitarios mediante rotación aleatoria.

```julia
cb = solve_codebook(128, 4)       # dimensión=128, 4 bits
# cb.centroids  — 16 valores de centroide óptimos
# cb.boundaries — 15 límites de decisión
```

### Rotación (`src/rotation.jl`)

Dos modos de rotación:

| Modo | Complejidad | Uso |
|---|---|---|
| `RandomRotation(d, seed)` | O(d²) | Ortogonal exacta, mejor calidad |
| `HadamardRotation(d, seed)` | O(d log d) | Estructurada, más rápida para d grande |

```julia
R = RandomRotation(128, UInt64(42))
y = rotate(R, x)          # directa
x_back = rotate_back(R, y) # inversa
```

### Cuantizador MSE (`src/mse_quantizer.jl`)

Cuantizador vectorial óptimo en MSE: rotación aleatoria + Lloyd-Max por coordenada.

```julia
tq = setup(TurboQuantMSE, d, b; use_hadamard=false)
comp = quantize(tq, X)         # devuelve CompressedVectorMSE
X_hat = dequantize(tq, comp)
ratio = compression_ratio(comp)
```

### Cuantizador de producto interno (`src/prod_quantizer.jl`)

Pipeline de dos etapas (MSE + QJL) que garantiza productos internos insesgados.

```julia
tq = setup(TurboQuantProd, d, b)   # b >= 2 requerido
comp = quantize(tq, X)
X_hat = dequantize(tq, comp)

# Calcular productos internos sin descompresión completa
scores = inner_product(tq, comp, query_vector)
```

### Caché KV (`src/kv_cache.jl`)

Caché KV cuantizada de reemplazo directo para atención de transformadores:

```julia
cache = KVCache(n_heads, head_dim, bit_width; max_seq_len=131072)

# Transmitir tokens
for token in tokens
    compress_kv!(cache, K_token, V_token)
end

# Calcular atención con caché cuantizada
output = attention_with_quantized_kv(cache, Q)

# Comparación de memoria
println("Cuantizada: ", memory_usage(cache), " bytes")
println("FP16:       ", fp16_memory_usage(cache), " bytes")
```

- Las claves usan `TurboQuantProd` (productos internos insesgados para puntuaciones de atención)
- Los valores usan `TurboQuantMSE` (MSE suficiente después de la ponderación softmax)
- Detección automática de cabezas atípicas con asignación de bits adicionales
- Desalojo FIFO en `max_seq_len`

### Búsqueda de vecinos más cercanos (`src/nn_search.jl`)

Búsqueda aproximada de vecinos más cercanos sin tiempo de entrenamiento:

```julia
# Construir índice (instantáneo — sin k-means)
index = build_index(database_vectors, 4; use_hadamard=true)

# Consulta
indices, scores = search(index, query, k)

# Consulta por lotes
idx_mat, score_mat = batch_search(index, queries, k)

# Evaluar recall
recall = recall_at_k(index, queries, ground_truth, k)
```

## Formato de archivo .tqt (v3)

Diseño binario con índices empaquetados en bits:

```
[Encabezado]  38 bytes
  Número mágico:    "TQT\x03"          (4 bytes)
  Banderas:         UInt8               (1 byte, bit 0 = sin pérdida)
  Tamaño original:  UInt64              (8 bytes)
  Dimensión:        UInt32              (4 bytes)
  Ancho de bit:     UInt8               (1 byte)
  Núm. vectores:    UInt64              (8 bytes)
  Relleno:          UInt32              (4 bytes)
  Semilla:          UInt64              (8 bytes)

[Libro de códigos]
  Núm. centroides:  UInt32              (4 bytes)
  Centroides:       Float64 × 2^b      (8 × 2^b bytes)

[Datos] × n_vectors
  Norma:            Float32             (4 bytes)
  Índices:          ceil(dim×b/8) bytes (empaquetados, b bits por índice)

[Residuo — solo modo sin pérdida]
  Tamaño residuo:   UInt32              (4 bytes)
  Datos residuales: variable            (residuos zigzag codificados con RLE)
```

**Cálculo de almacenamiento**: Con `bit_width=2, dim=64`, cada vector ocupa 4 (norma) + 16 (índices empaquetados) = 20 bytes en lugar de los 72 bytes (8 + 64) en v2. Para `bit_width=1`, son solo 4 + 8 = 12 bytes por vector.

## Rendimiento de distorsión

| Ancho de bit | Distorsión MSE | Ratio respecto al límite de Shannon |
|---|---|---|
| 1 | 0,36 | ~1,45× |
| 2 | 0,117 | ~1,87× |
| 3 | 0,030 | ~1,92× |
| 4 | 0,009 | ~2,30× |
| General b | (sqrt(3π)/2) · 4^(−b) | ≤ 2,72× |

## Ejecutar pruebas

```julia
using Pkg
Pkg.test("TurboQuant")
```

## Guía de parámetros

| Parámetro | Recomendado | Notas |
|---|---|---|
| `bit_width` | 3–4 para calidad, 1–2 para compresión agresiva | Mayor = mejor calidad, menor compresión |
| `dim` | 64–128 | Debe dividir los datos uniformemente; mayor = mejores propiedades estadísticas |
| `use_hadamard` | `true` para d > 256 | O(d log d) vs O(d²) rotación |
| `seed` | cualquier UInt64 | Misma semilla = compresión reproducible |

## Referencias

- Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization for Near-Optimal Quantized Retrieval, KV Cache Quantization, and Beyond.* arXiv:2504.19874v1, April 2025.

## Licencia

MIT
