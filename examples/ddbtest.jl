#!/usr/bin/env julia
#
# DuckDB + TurboQuant 벡터 양자화 비교 테스트
#
# NOAA 오로라 예측 데이터를 다운로드하여:
#   1) DuckDB에 원본(Float64) 그대로 저장
#   2) DuckDB에 TurboQuant 양자화 후 저장
#   3) 파일 크기, 쿼리 속도, 정확도 비교
#
# 필요 패키지:
#   using Pkg
#   Pkg.add(["DuckDB", "JSON3", "Downloads"])
#
# 실행:
#   julia examples/ddbtest.jl

using Downloads
using JSON3
using DuckDB
using LinearAlgebra
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "src", "TurboQuant.jl"))
using .TurboQuant

# ─────────────────────────────────────────────────────────────────────────────
# 1. NOAA 오로라 데이터 다운로드
# ─────────────────────────────────────────────────────────────────────────────

const AURORA_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"

function fetch_aurora_data()
    println("=== NOAA 오로라 데이터 다운로드 ===")
    println("URL: $AURORA_URL")

    tmpfile = Downloads.download(AURORA_URL)
    raw_json = read(tmpfile, String)
    rm(tmpfile, force=true)

    data = JSON3.read(raw_json)
    println("다운로드 완료: $(length(data))개 레코드")
    println("JSON 크기: $(sizeof(raw_json)) bytes")

    # 데이터 구조: [{Longitude, Latitude, Aurora}, ...]
    # Observation Time이 첫 항목에 있을 수 있음
    records = []
    for item in data
        if haskey(item, :Longitude) && haskey(item, :Latitude) && haskey(item, :Aurora)
            push!(records, (
                lon = Float64(item[:Longitude]),
                lat = Float64(item[:Latitude]),
                aurora = Float64(item[:Aurora])
            ))
        end
    end

    println("유효 레코드: $(length(records))개")

    # 위도별로 그룹화 → 각 위도가 하나의 벡터 (경도 방향 360개 값)
    lat_groups = Dict{Float64, Vector{Float64}}()
    for r in records
        if !haskey(lat_groups, r.lat)
            lat_groups[r.lat] = Float64[]
        end
        push!(lat_groups[r.lat], r.aurora)
    end

    println("위도 그룹: $(length(lat_groups))개")
    sample_len = length(first(values(lat_groups)))
    println("그룹당 벡터 길이: $(sample_len) (경도 방향)")

    return records, lat_groups
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. DuckDB에 원본 데이터 저장
# ─────────────────────────────────────────────────────────────────────────────

function create_raw_db(records, db_path::String)
    println("\n=== 원본 DuckDB 생성: $db_path ===")

    rm(db_path, force=true)
    db = DuckDB.DB(db_path)
    con = DuckDB.connect(db)

    # 테이블 생성
    DuckDB.execute(con, """
        CREATE TABLE aurora (
            longitude DOUBLE,
            latitude  DOUBLE,
            aurora    DOUBLE
        )
    """)

    # 데이터 삽입
    appender = DuckDB.Appender(con, "aurora")
    for r in records
        DuckDB.append(appender, r.lon)
        DuckDB.append(appender, r.lat)
        DuckDB.append(appender, r.aurora)
        DuckDB.end_row(appender)
    end
    DuckDB.close(appender)

    # 위도별 벡터 테이블 (DOUBLE[] 배열로 저장)
    DuckDB.execute(con, """
        CREATE TABLE aurora_vectors AS
        SELECT latitude,
               list(aurora ORDER BY longitude) AS aurora_vec,
               count(*) AS vec_len
        FROM aurora
        GROUP BY latitude
        ORDER BY latitude
    """)

    # 통계
    result = DuckDB.execute(con, "SELECT count(*) FROM aurora")
    row_count = DuckDB.toDataFrame(result).count_star[1] isa Missing ? 0 : DuckDB.toDataFrame(result).count_star[1]
    println("행 수: $row_count")

    result = DuckDB.execute(con, "SELECT count(*) FROM aurora_vectors")
    vec_count = DuckDB.toDataFrame(result).count_star[1]
    println("벡터 수: $vec_count")

    DuckDB.disconnect(con)
    DuckDB.close(db)

    fsize = filesize(db_path)
    println("DB 파일 크기: $(fsize) bytes ($(round(fsize/1024, digits=1)) KB)")
    return fsize
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. DuckDB에 TurboQuant 양자화 데이터 저장
# ─────────────────────────────────────────────────────────────────────────────

function create_quantized_db(lat_groups, db_path::String; bit_width::Int=2)
    println("\n=== TurboQuant $(bit_width)-bit 양자화 DuckDB 생성: $db_path ===")

    # 벡터 차원 결정 (모든 그룹이 같은 길이라고 가정)
    lats = sort(collect(keys(lat_groups)))
    dim = length(lat_groups[lats[1]])

    println("벡터 차원: $dim, 벡터 수: $(length(lats)), bit_width: $bit_width")

    # TurboQuant 설정
    tq = setup(TurboQuantMSE, dim, bit_width; seed=UInt64(42))

    rm(db_path, force=true)
    db = DuckDB.DB(db_path)
    con = DuckDB.connect(db)

    # 양자화된 데이터 테이블
    # indices를 BLOB으로, norm을 FLOAT으로 저장
    DuckDB.execute(con, """
        CREATE TABLE aurora_quantized (
            latitude     DOUBLE,
            norm         FLOAT,
            indices_blob BLOB,
            vec_len      INTEGER,
            bit_width    INTEGER
        )
    """)

    # 코드북 저장 (한 번만)
    DuckDB.execute(con, """
        CREATE TABLE codebook (
            bit_width   INTEGER,
            dim         INTEGER,
            seed        BIGINT,
            centroids   BLOB
        )
    """)

    # 코드북 삽입
    centroid_bytes = reinterpret(UInt8, Float64.(tq.codebook.centroids))
    appender_cb = DuckDB.Appender(con, "codebook")
    DuckDB.append(appender_cb, Int32(bit_width))
    DuckDB.append(appender_cb, Int32(dim))
    DuckDB.append(appender_cb, Int64(42))
    DuckDB.append(appender_cb, centroid_bytes)
    DuckDB.end_row(appender_cb)
    DuckDB.close(appender_cb)

    # 각 위도 벡터를 양자화하여 저장
    total_raw_bytes = 0
    total_quant_bytes = 0

    appender = DuckDB.Appender(con, "aurora_quantized")
    for lat in lats
        vec = lat_groups[lat]
        total_raw_bytes += length(vec) * 8  # Float64 = 8 bytes

        # 양자화
        comp = quantize(tq, vec)

        # 비트 패킹
        packed = bitpack(comp.indices[:, 1], bit_width)
        norm_val = Float32(comp.norms[1])

        total_quant_bytes += length(packed) + 4  # packed + Float32 norm

        DuckDB.append(appender, lat)
        DuckDB.append(appender, norm_val)
        DuckDB.append(appender, packed)
        DuckDB.append(appender, Int32(length(vec)))
        DuckDB.append(appender, Int32(bit_width))
        DuckDB.end_row(appender)
    end
    DuckDB.close(appender)

    DuckDB.disconnect(con)
    DuckDB.close(db)

    fsize = filesize(db_path)
    println("원본 벡터 데이터: $(total_raw_bytes) bytes ($(round(total_raw_bytes/1024, digits=1)) KB)")
    println("양자화 데이터:    $(total_quant_bytes) bytes ($(round(total_quant_bytes/1024, digits=1)) KB)")
    println("데이터 압축률:    $(round(total_raw_bytes / total_quant_bytes, digits=1))×")
    println("DB 파일 크기:     $(fsize) bytes ($(round(fsize/1024, digits=1)) KB)")
    return fsize, tq
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. 양자화 데이터를 읽어서 복원 & 정확도 측정
# ─────────────────────────────────────────────────────────────────────────────

function verify_quantized_db(lat_groups, db_path::String, tq)
    println("\n=== 양자화 데이터 복원 검증 ===")

    db = DuckDB.DB(db_path)
    con = DuckDB.connect(db)

    result = DuckDB.execute(con, "SELECT latitude, norm, indices_blob, vec_len, bit_width FROM aurora_quantized ORDER BY latitude")
    df = DuckDB.toDataFrame(result)

    lats = sort(collect(keys(lat_groups)))
    dim = length(lat_groups[lats[1]])

    total_mse = 0.0
    total_max_err = 0.0
    total_vals = 0
    cosine_sims = Float64[]

    for i in 1:size(df, 1)
        lat = df.latitude[i]
        norm_val = Float64(df.norm[i])
        packed = Vector{UInt8}(df.indices_blob[i])
        bit_width = Int(df.bit_width[i])

        # 복원
        indices = bitunpack(packed, dim, bit_width)
        indices_mat = reshape(indices, dim, 1)
        norms = [norm_val]
        comp = TurboQuant.CompressedVectorMSE(indices_mat, norms, dim, 1, bit_width)
        restored = dequantize(tq, comp)[:, 1]

        # 원본과 비교
        original = lat_groups[lat]
        diff = original .- restored
        mse = mean(diff .^ 2)
        max_err = maximum(abs.(diff))
        total_mse += mse
        total_max_err = max(total_max_err, max_err)
        total_vals += length(original)

        # 코사인 유사도
        if norm(original) > 1e-10 && norm(restored) > 1e-10
            cs = dot(original, restored) / (norm(original) * norm(restored))
            push!(cosine_sims, cs)
        end
    end

    avg_mse = total_mse / size(df, 1)
    avg_cosine = mean(cosine_sims)

    println("평균 MSE:        $(round(avg_mse, digits=4))")
    println("최대 오차:       $(round(total_max_err, digits=4))")
    println("평균 코사인 유사도: $(round(avg_cosine, digits=6))")
    println("벡터 수:         $(size(df, 1))")

    DuckDB.disconnect(con)
    DuckDB.close(db)

    return avg_mse, avg_cosine
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. 유사도 검색 비교 (원본 vs 양자화)
# ─────────────────────────────────────────────────────────────────────────────

function similarity_search_demo(lat_groups, tq, bit_width::Int)
    println("\n=== 유사도 검색 데모 ===")

    lats = sort(collect(keys(lat_groups)))
    dim = length(lat_groups[lats[1]])

    # 모든 벡터를 행렬로
    n = length(lats)
    X = Matrix{Float64}(undef, dim, n)
    for (j, lat) in enumerate(lats)
        X[:, j] .= lat_groups[lat]
    end

    # 쿼리: 위도 65도 근처의 오로라 패턴 (오로라 활동 활발 지역)
    query_lat_idx = argmin(abs.(lats .- 65.0))
    query = X[:, query_lat_idx]
    println("쿼리 벡터: 위도 $(lats[query_lat_idx])°의 오로라 패턴")

    # 원본으로 내적 검색
    raw_scores = X' * query
    raw_top5 = sortperm(raw_scores, rev=true)[1:5]

    println("\n[원본 데이터 Top-5 유사 위도]")
    for (rank, idx) in enumerate(raw_top5)
        @printf("  %d. 위도 %6.1f° | 내적: %10.2f\n", rank, lats[idx], raw_scores[idx])
    end

    # TurboQuant 양자화 후 검색
    tq_prod = setup(TurboQuantProd, dim, max(bit_width, 2);
                    seed=UInt64(42))
    comp = quantize(tq_prod, X)
    quant_scores = inner_product(tq_prod, comp, query)
    quant_top5 = sortperm(quant_scores, rev=true)[1:5]

    println("\n[TurboQuant $(bit_width)-bit 양자화 Top-5 유사 위도]")
    for (rank, idx) in enumerate(quant_top5)
        @printf("  %d. 위도 %6.1f° | 내적: %10.2f (원본: %10.2f)\n",
                rank, lats[idx], quant_scores[idx], raw_scores[idx])
    end

    # Top-5 일치율
    overlap = length(intersect(Set(raw_top5), Set(quant_top5)))
    println("\nTop-5 일치율: $overlap / 5 ($(overlap * 20)%)")
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main()
    println("=" ^ 65)
    println("  DuckDB + TurboQuant 벡터 양자화 비교 테스트")
    println("  데이터: NOAA Ovation Aurora 예측 (실시간)")
    println("=" ^ 65)

    # 1. 데이터 다운로드
    records, lat_groups = fetch_aurora_data()

    # 경로 설정
    dir = @__DIR__
    raw_db_path = joinpath(dir, "aurora_raw.duckdb")
    quant_db_paths = Dict{Int, String}()

    # 2. 원본 DB 생성
    raw_size = create_raw_db(records, raw_db_path)

    # 3. 여러 bit-width로 양자화 DB 생성 & 비교
    println("\n" * "=" ^ 65)
    println("  bit-width별 양자화 비교")
    println("=" ^ 65)

    results = []
    local last_tq  # 마지막 tq를 유사도 검색에 사용

    for bw in [1, 2, 3, 4]
        qpath = joinpath(dir, "aurora_quant_$(bw)bit.duckdb")
        quant_db_paths[bw] = qpath

        qsize, tq = create_quantized_db(lat_groups, qpath; bit_width=bw)
        mse, cosine = verify_quantized_db(lat_groups, qpath, tq)

        push!(results, (bw=bw, raw_size=raw_size, quant_size=qsize,
                        mse=mse, cosine=cosine))
        last_tq = tq
    end

    # 4. 요약 테이블
    println("\n" * "=" ^ 65)
    println("  최종 비교 요약")
    println("=" ^ 65)
    println()
    @printf("%-6s | %12s | %12s | %8s | %10s | %12s\n",
            "Bits", "원본 DB", "양자화 DB", "DB 비율", "평균 MSE", "코사인 유사도")
    println("-" ^ 72)
    for r in results
        @printf("  %d    | %8.1f KB | %8.1f KB | %6.1f%% | %10.4f | %12.6f\n",
                r.bw,
                r.raw_size / 1024,
                r.quant_size / 1024,
                r.quant_size / r.raw_size * 100,
                r.mse,
                r.cosine)
    end

    println()
    println("참고: 원본 DB는 모든 행(lon, lat, aurora) + 위도별 배열 테이블 포함")
    println("      양자화 DB는 비트 패킹된 인덱스 + Float32 norm + 코드북만 포함")

    # 5. 유사도 검색 데모
    similarity_search_demo(lat_groups, last_tq, 2)

    # 6. 정리
    println("\n=== 생성된 파일 ===")
    println("  원본 DB:    $raw_db_path ($(round(filesize(raw_db_path)/1024, digits=1)) KB)")
    for (bw, path) in sort(collect(quant_db_paths))
        println("  $(bw)-bit DB:   $path ($(round(filesize(path)/1024, digits=1)) KB)")
    end

    println("\n완료!")
end

main()
