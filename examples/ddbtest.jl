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
#   Pkg.add(["DuckDB", "JSON3"])
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

# ── Bit-packing (self-contained, no external dependency) ────────────────────

function _bitpack(indices::AbstractVector{UInt8}, bit_width::Int)
    n = length(indices)
    n_bytes = cld(n * bit_width, 8)
    packed = zeros(UInt8, n_bytes)
    bit_pos = 0
    for i in 1:n
        val = UInt32(indices[i] - 1)
        for b in 0:(bit_width - 1)
            if (val >> b) & 1 == 1
                packed[(bit_pos >> 3) + 1] |= UInt8(1) << (bit_pos & 7)
            end
            bit_pos += 1
        end
    end
    return packed
end

function _bitunpack(packed::AbstractVector{UInt8}, n::Int, bit_width::Int)
    indices = Vector{UInt8}(undef, n)
    bit_pos = 0
    for i in 1:n
        val = UInt32(0)
        for b in 0:(bit_width - 1)
            if (packed[(bit_pos >> 3) + 1] >> (bit_pos & 7)) & 1 == 1
                val |= UInt32(1) << b
            end
            bit_pos += 1
        end
        indices[i] = UInt8(val + 1)
    end
    return indices
end

# DuckDB 쿼리 결과를 Dict 배열로 변환하는 헬퍼
# DuckDB.jl 결과는 columnar table — Tables.jl 인터페이스 사용
function query_to_dicts(con, sql::String)
    result = DuckDB.execute(con, sql)
    # DuckDB.execute returns a materialized result that supports columnar access
    # Convert via iteration — each row is a NamedTuple in DuckDB.jl
    return collect(result)
end

function query_scalar(con, sql::String)
    rows = query_to_dicts(con, sql)
    # 첫 번째 행의 첫 번째 값 반환
    row = first(rows)
    return row[1]
end

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
    println("JSON 크기: $(sizeof(raw_json)) bytes")

    # JSON 구조 탐색: 최상위가 객체(dict)인지 배열인지에 따라 분기
    # NOAA 오로라 데이터는 보통:
    #   { "Observation Time": "...", "Forecast Time": "...",
    #     "Data Format": "[Longitude, Latitude, Aurora]",
    #     "coordinates": [[lon, lat, aurora], ...] }
    # 또는 플랫 배열: [{Longitude, Latitude, Aurora}, ...]

    records = []

    if data isa AbstractDict || data isa JSON3.Object
        # 객체 형태 — "coordinates" 키에서 좌표 배열 추출
        println("JSON 형식: 객체 (키: $(join(keys(data), ", ")))")

        if haskey(data, :coordinates)
            coords = data[:coordinates]
        elseif haskey(data, Symbol("coordinates"))
            coords = data[Symbol("coordinates")]
        else
            # 키 이름 탐색
            coord_key = nothing
            for k in keys(data)
                val = data[k]
                if val isa AbstractVector && length(val) > 100
                    coord_key = k
                    break
                end
            end
            if coord_key !== nothing
                coords = data[coord_key]
                println("좌표 데이터 키: $coord_key")
            else
                error("좌표 데이터를 찾을 수 없습니다. 키 목록: $(collect(keys(data)))")
            end
        end

        println("좌표 레코드 수: $(length(coords))")

        for item in coords
            if item isa AbstractVector || item isa JSON3.Array
                # [lon, lat, aurora] 형태
                if length(item) >= 3
                    push!(records, (
                        lon = Float64(item[1]),
                        lat = Float64(item[2]),
                        aurora = Float64(item[3])
                    ))
                end
            elseif item isa AbstractDict || item isa JSON3.Object
                # {Longitude, Latitude, Aurora} 형태
                lon_val = get(item, :Longitude, get(item, :longitude, nothing))
                lat_val = get(item, :Latitude, get(item, :latitude, nothing))
                aur_val = get(item, :Aurora, get(item, :aurora, nothing))
                if lon_val !== nothing && lat_val !== nothing && aur_val !== nothing
                    push!(records, (
                        lon = Float64(lon_val),
                        lat = Float64(lat_val),
                        aurora = Float64(aur_val)
                    ))
                end
            end
        end

    elseif data isa AbstractVector
        # 배열 형태
        println("JSON 형식: 배열 ($(length(data))개 항목)")

        for item in data
            if item isa AbstractVector || item isa JSON3.Array
                if length(item) >= 3 && all(x -> x isa Number, item)
                    push!(records, (
                        lon = Float64(item[1]),
                        lat = Float64(item[2]),
                        aurora = Float64(item[3])
                    ))
                end
            elseif item isa AbstractDict || item isa JSON3.Object
                lon_val = get(item, :Longitude, get(item, :longitude, nothing))
                lat_val = get(item, :Latitude, get(item, :latitude, nothing))
                aur_val = get(item, :Aurora, get(item, :aurora, nothing))
                if lon_val !== nothing && lat_val !== nothing && aur_val !== nothing
                    push!(records, (
                        lon = Float64(lon_val),
                        lat = Float64(lat_val),
                        aurora = Float64(aur_val)
                    ))
                end
            end
        end
    else
        error("예상치 못한 JSON 최상위 타입: $(typeof(data))")
    end

    println("유효 레코드: $(length(records))개")

    if isempty(records)
        # 디버그: JSON 일부 출력
        preview = raw_json[1:min(500, length(raw_json))]
        println("JSON 미리보기: $preview")
        error("유효한 좌표 데이터를 파싱할 수 없습니다")
    end

    # 위도별로 그룹화 → 각 위도가 하나의 벡터 (경도 방향 값들)
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
    row_count = query_scalar(con, "SELECT count(*) FROM aurora")
    println("행 수: $row_count")

    vec_count = query_scalar(con, "SELECT count(*) FROM aurora_vectors")
    println("벡터 수: $vec_count")

    # 원본 벡터 데이터 크기 (Float64 × 360 × 181)
    raw_data_bytes = length(records) * 8  # 65160 × 8 bytes = 521,280

    DuckDB.execute(con, "CHECKPOINT")
    DuckDB.disconnect(con)
    DuckDB.close(db)

    fsize = filesize(db_path)
    println("순수 벡터 데이터: $(raw_data_bytes) bytes ($(round(raw_data_bytes/1024, digits=1)) KB)")
    println("DB 파일 크기:     $(fsize) bytes ($(round(fsize/1024, digits=1)) KB)")
    println("  (DuckDB 최소 블록 할당으로 파일이 데이터보다 큼)")

    # 원본 벡터를 바이너리 파일로도 저장 (공정한 크기 비교용)
    raw_bin_path = replace(db_path, ".duckdb" => ".raw.bin")
    open(raw_bin_path, "w") do io
        for r in records
            write(io, Float64(r.aurora))
        end
    end
    println("원본 바이너리:    $(filesize(raw_bin_path)) bytes ($(round(filesize(raw_bin_path)/1024, digits=1)) KB)")

    return raw_data_bytes
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

    # 코드북 삽입 (hex-encoded BLOB via SQL)
    centroid_bytes = Vector{UInt8}(reinterpret(UInt8, collect(Float64.(tq.codebook.centroids))))
    hex_cb = bytes2hex(centroid_bytes)
    DuckDB.execute(con, """
        INSERT INTO codebook VALUES ($bit_width, $dim, 42, '\\x$hex_cb'::BLOB)
    """)

    # 각 위도 벡터를 양자화하여 저장
    total_raw_bytes = 0
    total_quant_bytes = 0

    for lat in lats
        vec = lat_groups[lat]
        total_raw_bytes += length(vec) * 8  # Float64 = 8 bytes

        # 양자화
        comp = quantize(tq, vec)

        # 비트 패킹
        packed = _bitpack(comp.indices[:, 1], bit_width)
        norm_val = Float32(comp.norms[1])

        total_quant_bytes += length(packed) + 4  # packed + Float32 norm

        # BLOB은 hex 리터럴로 삽입
        hex_packed = bytes2hex(packed)
        vlen = length(vec)
        DuckDB.execute(con, """
            INSERT INTO aurora_quantized VALUES
            ($lat, $norm_val, '\\x$hex_packed'::BLOB, $vlen, $bit_width)
        """)
    end

    DuckDB.execute(con, "CHECKPOINT")
    DuckDB.disconnect(con)
    DuckDB.close(db)

    fsize = filesize(db_path)

    # 양자화 바이너리 파일로도 저장 (공정한 크기 비교용)
    quant_bin_path = replace(db_path, ".duckdb" => ".bin")
    open(quant_bin_path, "w") do io
        for lat in lats
            vec = lat_groups[lat]
            comp = quantize(tq, vec)
            packed = _bitpack(comp.indices[:, 1], bit_width)
            write(io, Float32(comp.norms[1]))  # 4 bytes
            write(io, packed)                   # ceil(dim*bw/8) bytes
        end
    end
    quant_bin_size = filesize(quant_bin_path)

    println("원본 벡터 데이터:   $(total_raw_bytes) bytes ($(round(total_raw_bytes/1024, digits=1)) KB)")
    println("양자화 바이너리:    $(quant_bin_size) bytes ($(round(quant_bin_size/1024, digits=1)) KB)")
    println("순수 데이터 압축률: $(round(total_raw_bytes / quant_bin_size, digits=1))×")
    println("DB 파일 크기:       $(fsize) bytes ($(round(fsize/1024, digits=1)) KB)")

    return total_raw_bytes, quant_bin_size, tq
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. 양자화 데이터를 읽어서 복원 & 정확도 측정
# ─────────────────────────────────────────────────────────────────────────────

function verify_quantized_db(lat_groups, db_path::String, tq)
    println("\n=== 양자화 데이터 복원 검증 ===")

    db = DuckDB.DB(db_path)
    con = DuckDB.connect(db)

    rows = query_to_dicts(con, "SELECT latitude, norm, indices_blob, vec_len, bit_width FROM aurora_quantized ORDER BY latitude")

    lats = sort(collect(keys(lat_groups)))
    dim = length(lat_groups[lats[1]])

    total_mse = 0.0
    total_max_err = 0.0
    total_vals = 0
    cosine_sims = Float64[]

    for i in 1:length(rows)
        row = rows[i]
        lat = Float64(row.latitude)
        norm_val = Float64(row.norm)
        blob = row.indices_blob
        packed = blob isa Vector{UInt8} ? blob : Vector{UInt8}(blob)
        bit_width = Int(row.bit_width)

        # 복원
        indices = _bitunpack(packed, dim, bit_width)
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

    avg_mse = total_mse / length(rows)
    avg_cosine = mean(cosine_sims)

    println("평균 MSE:        $(round(avg_mse, digits=4))")
    println("최대 오차:       $(round(total_max_err, digits=4))")
    println("평균 코사인 유사도: $(round(avg_cosine, digits=6))")
    println("벡터 수:         $(length(rows))")

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

    # 데이터를 88번 복제하여 대규모 데이터셋 시뮬레이션
    REPEAT = 88
    println("\n데이터 $(REPEAT)× 복제 중...")
    orig_lats = sort(collect(keys(lat_groups)))
    expanded_groups = Dict{Float64, Vector{Float64}}()
    expanded_records = []
    for rep in 1:REPEAT
        offset = (rep - 1) * 1000.0  # 위도 offset으로 고유 키 생성
        for lat in orig_lats
            new_lat = lat + offset
            expanded_groups[new_lat] = copy(lat_groups[lat])
            for (i, val) in enumerate(lat_groups[lat])
                push!(expanded_records, (lon=Float64(i), lat=new_lat, aurora=val))
            end
        end
    end
    records = expanded_records
    lat_groups = expanded_groups
    n_vecs = length(lat_groups)
    n_recs = length(records)
    println("복제 완료: $(n_recs)개 레코드, $(n_vecs)개 벡터 ($(REPEAT)× 원본)")

    # 경로 설정
    dir = @__DIR__
    raw_db_path = joinpath(dir, "aurora_raw.duckdb")
    quant_db_paths = Dict{Int, String}()

    # 2. 원본 DB 생성
    raw_data_bytes = create_raw_db(records, raw_db_path)

    # 3. 여러 bit-width로 양자화 DB 생성 & 비교
    println("\n" * "=" ^ 65)
    println("  bit-width별 양자화 비교")
    println("=" ^ 65)

    results = []
    local last_tq  # 마지막 tq를 유사도 검색에 사용

    for bw in [1, 2, 3, 4]
        qpath = joinpath(dir, "aurora_quant_$(bw)bit.duckdb")
        quant_db_paths[bw] = qpath

        raw_bytes, quant_bytes, tq = create_quantized_db(lat_groups, qpath; bit_width=bw)
        mse, cosine = verify_quantized_db(lat_groups, qpath, tq)

        push!(results, (bw=bw, raw_bytes=raw_bytes, quant_bytes=quant_bytes,
                        mse=mse, cosine=cosine))
        last_tq = tq
    end

    # 4. 요약 테이블
    println("\n" * "=" ^ 65)
    println("  최종 비교 요약 (순수 데이터 크기 기준)")
    println("=" ^ 65)
    println()
    @printf("%-6s | %10s | %10s | %8s | %10s | %12s\n",
            "Bits", "원본 데이터", "양자화", "압축률", "평균 MSE", "코사인 유사도")
    println("-" ^ 72)
    for r in results
        @printf("  %d    | %7.1f KB | %7.1f KB | %6.1f× | %10.4f | %12.6f\n",
                r.bw,
                r.raw_bytes / 1024,
                r.quant_bytes / 1024,
                r.raw_bytes / r.quant_bytes,
                r.mse,
                r.cosine)
    end

    println()
    println("참고: '원본 데이터' = Float64 × 65,160개 오로라 값 = $(round(raw_data_bytes/1024, digits=1)) KB")
    println("      '양자화' = 비트 패킹 인덱스 + Float32 norm (바이너리 파일 기준)")
    println("      DuckDB 파일 크기는 최소 블록 할당(~780KB)으로 차이가 안 보임")

    # 5. 유사도 검색 데모
    similarity_search_demo(lat_groups, last_tq, 2)

    # 6. 파일 크기 비교 (바이너리 파일 — DuckDB 오버헤드 없이 순수 비교)
    raw_bin_path = replace(raw_db_path, ".duckdb" => ".raw.bin")
    println("\n=== 파일 크기 비교 (바이너리) ===")
    if isfile(raw_bin_path)
        @printf("  원본 (Float64): %8.1f KB  ← %s\n", filesize(raw_bin_path)/1024, raw_bin_path)
    end
    for (bw, path) in sort(collect(quant_db_paths))
        bin_path = replace(path, ".duckdb" => ".bin")
        if isfile(bin_path)
            ratio = filesize(raw_bin_path) / filesize(bin_path)
            @printf("  %d-bit 양자화:  %8.1f KB  (%.1f× 압축)  ← %s\n",
                    bw, filesize(bin_path)/1024, ratio, bin_path)
        end
    end

    println("\n=== DuckDB 파일 ===")
    println("  원본 DB:    $(round(filesize(raw_db_path)/1024, digits=1)) KB")
    for (bw, path) in sort(collect(quant_db_paths))
        println("  $(bw)-bit DB:   $(round(filesize(path)/1024, digits=1)) KB")
    end
    println("  (DuckDB 최소 블록 할당으로 소규모 데이터에서는 파일 크기 동일)")

    println("\n완료!")
end

main()
