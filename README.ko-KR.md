[English](README.md) | [한국어](README.ko-KR.md) | [Svenska](README.sv-SE.md) | [简体中文](README.zh-CN.md)

# TurboQuant.jl

[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://github.com/JaewooJoung/TurboQuant.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JaewooJoung.github.io/TurboQuant.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JaewooJoung.github.io/TurboQuant.jl/dev)
[![Build Status](https://github.com/JaewooJoung/TurboQuant.jl/workflows/CI/badge.svg)](https://github.com/JaewooJoung/TurboQuant.jl/actions)
[![Coverage](https://codecov.io/gh/JaewooJoung/TurboQuant.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JaewooJoung/TurboQuant.jl)
[![License](https://img.shields.io/github/license/JaewooJoung/TurboQuant.jl)](https://github.com/JaewooJoung/TurboQuant.jl/blob/main/LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-blueviolet?logo=julia)](https://julialang.org/)


[TurboQuant (arXiv:2504.19874v1)](https://arxiv.org/abs/2504.19874v1) 기반의 고급 벡터 양자화 Julia 구현체입니다. 보정 데이터나 코드북 학습 없이 정보 이론적 최적에 근접한 압축을 달성합니다.

## 주요 특징

- **온라인 / 데이터 비의존적** — 전처리, 보정, 코드북 학습이 필요 없음
- **준최적 왜곡** — Shannon 하한의 2.72배 이내, O(1/4^b) 비율
- **비편향 내적** — 어텐션 메커니즘 및 최근접 이웃 탐색에 핵심적
- **완전 벡터화 가능** — GPU/TPU 친화적 실행을 위해 설계

## 설치

```julia
using Pkg
Pkg.add("TurboQuant")
```

또는 저장소에서 직접 설치:

```julia
using Pkg
Pkg.add(url="https://github.com/JaewooJoung/TurboQuant.jl")
```

## 빠른 시작

### 코드에서 모듈 사용하기

```julia
using TurboQuant
using LinearAlgebra

# --- MSE 양자화기 ---
d = 128    # 벡터 차원
b = 4      # 좌표당 비트 수
tq = setup(TurboQuantMSE, d, b)

x = randn(d)                          # 원본 벡터
comp = quantize(tq, x)                # 압축
x_hat = dequantize(tq, comp)          # 복원
println("MSE: ", sum((x .- x_hat[:,1]).^2) / d)

# --- 배치 양자화 ---
X = randn(d, 1000)                    # 1000개 벡터
comp = quantize(tq, X)
X_hat = dequantize(tq, comp)

# --- 비편향 내적 양자화기 ---
tq_prod = setup(TurboQuantProd, d, 4)
comp = quantize(tq_prod, X)
y = randn(d)
approx_ips = inner_product(tq_prod, comp, y)   # E[result] = X' * y (비편향)
```

### 파일 압축 (.tqt 형식)

모든 파일을 `.tqt` 형식으로 압축하고 다시 복원할 수 있습니다:

```julia
include("examples/tqt_file_io.jl")

# 무손실 압축 (기본값) — 정확한 복원 보장
compress_file("myfile.txt", "myfile.tqt"; bit_width=4, dim=64, lossless=true)
decompress_file("myfile.tqt", "myfile_restored.txt")
verify_roundtrip("myfile.txt", "myfile_restored.txt")
# → 결과: LOSSLESS (완벽한 복원)

# 손실 압축 — 더 작은 출력, 근사 복원
compress_file("myfile.txt", "myfile.tqt"; bit_width=4, dim=64, lossless=false)
```

**무손실 모드 동작 원리**: TurboQuant는 손실 벡터 양자화로 데이터를 압축한 후
바이트 수준의 잔차(원본 - 복원)를 계산하여 RLE 인코딩으로 저장합니다. 복원 시
잔차 보정을 더하여 정확한 원본을 재구성합니다. `bit_width`가 높을수록 잔차가
작아져 전체 압축률이 향상됩니다.

## 예제

### 데모 실행

데모는 다양한 비트 폭에 대해 무손실 및 손실 비교를 보여줍니다:

```bash
# 내장 샘플 텍스트 사용
julia examples/example_compress.jl

# 자신의 파일 압축
julia examples/example_compress.jl path/to/myfile.txt

# 사용자 정의 비트 폭(6) 및 벡터 차원(128)
julia examples/example_compress.jl myfile.txt 6 128
```

출력 예시:

```
=== 무손실 모드: 비트 폭 비교 ===
비트 | 압축 크기   | 비율   | TQT 부분 | 잔차 부분      | 정확?
-----|------------|--------|----------|---------------|-------
  1  |        ... |  ...×  |      ... |           ... | YES
  4  |        ... |  ...×  |      ... |           ... | YES
  8  |        ... |  ...×  |      ... |           ... | YES

=== 손실 모드: 비트 폭 비교 ===
비트 | 압축 크기   | 비율   | 정확 일치율 %
-----|------------|--------|-------------
  1  |        ... |  ...×  | ~0.7%
  4  |        ... |  ...×  | ~3.5%
  8  |        ... |  ...×  | ~43.5%
```

### CLI 도구

압축 / 복원 / 검사를 위한 독립형 커맨드라인 도구:

```bash
# 무손실 압축 (기본값)
julia examples/tqt_tool.jl compress input.png output.tqt 4 128

# 손실 압축
julia examples/tqt_tool.jl compress input.png output.tqt 2 64 --lossy

# 복원 (헤더에서 무손실/손실 자동 감지)
julia examples/tqt_tool.jl decompress output.tqt restored.png

# .tqt 파일 헤더 검사
julia examples/tqt_tool.jl info output.tqt
```

`info` 출력은 모드(무손실/손실), 원본 크기, 벡터 차원, 비트 폭, 코드북 중심값, 잔차 크기, 유효 바이트당 비트를 보여줍니다.

## 모듈 구성요소

### 코드북 (`src/codebook.jl`)

단위 벡터를 랜덤 회전으로 투영할 때 발생하는 베타 분포에 대한 Lloyd-Max 최적 스칼라 양자화기입니다.

```julia
cb = solve_codebook(128, 4)       # 차원=128, 4비트
# cb.centroids  — 16개 최적 중심값
# cb.boundaries — 15개 결정 경계
```

### 회전 (`src/rotation.jl`)

두 가지 회전 모드:

| 모드 | 복잡도 | 용도 |
|---|---|---|
| `RandomRotation(d, seed)` | O(d²) | 정확한 직교, 최상 품질 |
| `HadamardRotation(d, seed)` | O(d log d) | 구조적, 큰 d에 더 빠름 |

```julia
R = RandomRotation(128, UInt64(42))
y = rotate(R, x)          # 순방향
x_back = rotate_back(R, y) # 역방향
```

### MSE 양자화기 (`src/mse_quantizer.jl`)

MSE 최적 벡터 양자화기: 랜덤 회전 + 좌표별 Lloyd-Max.

```julia
tq = setup(TurboQuantMSE, d, b; use_hadamard=false)
comp = quantize(tq, X)         # CompressedVectorMSE 반환
X_hat = dequantize(tq, comp)
ratio = compression_ratio(comp)
```

### 내적 양자화기 (`src/prod_quantizer.jl`)

비편향 내적을 보장하는 2단계 파이프라인 (MSE + QJL).

```julia
tq = setup(TurboQuantProd, d, b)   # b >= 2 필수
comp = quantize(tq, X)
X_hat = dequantize(tq, comp)

# 전체 복원 없이 내적 계산
scores = inner_product(tq, comp, query_vector)
```

### KV 캐시 (`src/kv_cache.jl`)

트랜스포머 어텐션을 위한 드롭인 양자화 KV 캐시:

```julia
cache = KVCache(n_heads, head_dim, bit_width; max_seq_len=131072)

# 토큰 스트리밍
for token in tokens
    compress_kv!(cache, K_token, V_token)
end

# 양자화된 캐시로 어텐션 계산
output = attention_with_quantized_kv(cache, Q)

# 메모리 비교
println("양자화: ", memory_usage(cache), " bytes")
println("FP16:   ", fp16_memory_usage(cache), " bytes")
```

- Key는 `TurboQuantProd` 사용 (어텐션 점수를 위한 비편향 내적)
- Value는 `TurboQuantMSE` 사용 (소프트맥스 가중 후 MSE로 충분)
- 추가 비트 할당을 통한 자동 아웃라이어 헤드 감지
- `max_seq_len`에서 FIFO 축출

### 최근접 이웃 탐색 (`src/nn_search.jl`)

학습 시간 제로의 근사 최근접 이웃 탐색:

```julia
# 인덱스 구축 (즉시 — k-means 불필요)
index = build_index(database_vectors, 4; use_hadamard=true)

# 쿼리
indices, scores = search(index, query, k)

# 배치 쿼리
idx_mat, score_mat = batch_search(index, queries, k)

# 재현율 평가
recall = recall_at_k(index, queries, ground_truth, k)
```

## .tqt 파일 형식 (v3)

비트 패킹 인덱스가 포함된 바이너리 레이아웃:

```
[헤더]  38바이트
  매직 넘버:      "TQT\x03"          (4바이트)
  플래그:         UInt8               (1바이트, 비트 0 = 무손실)
  원본 크기:      UInt64              (8바이트)
  차원:           UInt32              (4바이트)
  비트 폭:        UInt8               (1바이트)
  벡터 수:        UInt64              (8바이트)
  패딩:           UInt32              (4바이트)
  시드:           UInt64              (8바이트)

[코드북]
  중심값 수:      UInt32              (4바이트)
  중심값:         Float64 × 2^b      (8 × 2^b 바이트)

[데이터] × n_vectors
  노름:           Float32             (4바이트)
  인덱스:         ceil(dim×b/8) 바이트 (비트 패킹, 인덱스당 b비트)

[잔차 — 무손실 모드 전용]
  잔차 크기:      UInt32              (4바이트)
  잔차 데이터:    가변               (RLE 인코딩된 지그재그 잔차)
```

**저장 공간 계산**: `bit_width=2, dim=64`일 때 각 벡터는 4 (노름) + 16 (패킹된 인덱스) = 20바이트로, v2의 72바이트(8 + 64) 대신 사용됩니다. `bit_width=1`의 경우 벡터당 4 + 8 = 12바이트에 불과합니다.

## 왜곡 성능

| 비트 폭 | MSE 왜곡 | Shannon 하한 대비 비율 |
|---|---|---|
| 1 | 0.36 | ~1.45x |
| 2 | 0.117 | ~1.87x |
| 3 | 0.030 | ~1.92x |
| 4 | 0.009 | ~2.30x |
| 일반 b | (sqrt(3pi)/2) · 4^(-b) | ≤ 2.72x |

## 테스트 실행

```julia
using Pkg
Pkg.test("TurboQuant")
```

## 매개변수 가이드

| 매개변수 | 권장값 | 비고 |
|---|---|---|
| `bit_width` | 품질: 3-4, 공격적 압축: 1-2 | 높을수록 품질 향상, 압축률 감소 |
| `dim` | 64-128 | 데이터에 균등 분할 필수; 클수록 통계적 특성 향상 |
| `use_hadamard` | d > 256일 때 `true` | O(d log d) vs O(d²) 회전 |
| `seed` | 임의의 UInt64 | 동일 시드 = 재현 가능한 압축 |

## 참고문헌

- Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization for Near-Optimal Quantized Retrieval, KV Cache Quantization, and Beyond.* arXiv:2504.19874v1, April 2025.

## 라이선스

MIT
