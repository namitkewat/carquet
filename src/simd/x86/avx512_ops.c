/**
 * @file avx512_ops.c
 * @brief AVX-512 optimized operations for x86-64 processors
 *
 * Provides SIMD-accelerated implementations using 512-bit vectors:
 * - Bit unpacking for various bit widths
 * - Byte stream split/merge (for BYTE_STREAM_SPLIT encoding)
 * - Delta decoding (prefix sums)
 * - Dictionary gather operations (using AVX-512 scatter/gather)
 * - Boolean packing/unpacking
 * - Masked operations for predicated processing
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
/* Check for AVX-512 support */
#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(__AVX512F__))

#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <immintrin.h>

/* Portable count trailing zeros */
static inline int portable_ctz(unsigned int v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(v);
#elif defined(_MSC_VER)
    unsigned long index;
    _BitScanForward(&index, v);
    return (int)index;
#else
    int n = 0;
    if (!(v & 0xFFFF)) { n += 16; v >>= 16; }
    if (!(v & 0xFF)) { n += 8; v >>= 8; }
    if (!(v & 0xF)) { n += 4; v >>= 4; }
    if (!(v & 0x3)) { n += 2; v >>= 2; }
    if (!(v & 0x1)) { n += 1; }
    return n;
#endif
}

/* Portable 64-bit count trailing zeros (needed for 64-byte mask operations) */
static inline int portable_ctz64(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(v);
#elif defined(_MSC_VER)
    unsigned long index;
    _BitScanForward64(&index, (unsigned __int64)v);
    return (int)index;
#else
    if ((uint32_t)v) return portable_ctz((unsigned int)v);
    return 32 + portable_ctz((unsigned int)(v >> 32));
#endif
}

/* ============================================================================
 * Bit Unpacking - AVX-512 Optimized
 * ============================================================================
 */

void carquet_avx512_bitunpack32_8bit(const uint8_t* input, uint32_t* values);

/**
 * Unpack 8 8-bit values to 32-bit using AVX-512.
 */
void carquet_avx512_bitunpack8_8bit(const uint8_t* input, uint32_t* values) {
    __m128i bytes = _mm_loadl_epi64((const __m128i*)input);
    __m512i expanded = _mm512_cvtepu8_epi32(bytes);
    __m256i result = _mm512_castsi512_si256(expanded);
    _mm256_storeu_si256((__m256i*)values, result);
}

/**
 * Unpack 8 16-bit values to 32-bit using AVX-512.
 */
void carquet_avx512_bitunpack8_16bit(const uint8_t* input, uint32_t* values) {
    __m128i words = _mm_loadu_si128((const __m128i*)input);
    __m256i result = _mm256_cvtepu16_epi32(words);
    _mm256_storeu_si256((__m256i*)values, result);
}

/**
 * Unpack 8 4-bit values to 32-bit using AVX-512.
 */
void carquet_avx512_bitunpack8_4bit(const uint8_t* input, uint32_t* values) {
    /* 8 x 4-bit values = 4 input bytes, expand nibbles to bytes then widen */
    uint8_t expanded[8];
    for (int i = 0; i < 4; i++) {
        uint8_t byte = input[i];
        expanded[i * 2] = (uint8_t)(byte & 0x0F);
        expanded[i * 2 + 1] = (uint8_t)(byte >> 4);
    }
    __m128i bytes = _mm_loadl_epi64((const __m128i*)expanded);
    __m256i result = _mm256_cvtepu8_epi32(bytes);
    _mm256_storeu_si256((__m256i*)values, result);
}

/**
 * Unpack 32 8-bit values to 32-bit using AVX-512.
 */
void carquet_avx512_bitunpack32_8bit(const uint8_t* input, uint32_t* values) {
    /* Load 32 bytes as two 128-bit halves */
    __m128i bytes_lo = _mm_loadu_si128((const __m128i*)input);
    __m128i bytes_hi = _mm_loadu_si128((const __m128i*)(input + 16));

    /* Expand each half to 32-bit using AVX-512 (16 x 8-bit -> 16 x 32-bit) */
    __m512i result_lo = _mm512_cvtepu8_epi32(bytes_lo);
    __m512i result_hi = _mm512_cvtepu8_epi32(bytes_hi);

    _mm512_storeu_si512((__m512i*)values, result_lo);
    _mm512_storeu_si512((__m512i*)(values + 16), result_hi);
}

/**
 * Unpack 16 16-bit values to 32-bit using AVX-512.
 */
void carquet_avx512_bitunpack16_16bit(const uint8_t* input, uint32_t* values) {
    __m256i words = _mm256_loadu_si256((const __m256i*)input);
    __m512i result = _mm512_cvtepu16_epi32(words);
    _mm512_storeu_si512((__m512i*)values, result);
}

/**
 * Unpack 32 4-bit values to 32-bit using AVX-512.
 */
void carquet_avx512_bitunpack32_4bit(const uint8_t* input, uint32_t* values) {
    /* Load 16 bytes containing 32 x 4-bit values */
    __m128i bytes = _mm_loadu_si128((const __m128i*)input);

    /* Split nibbles */
    __m128i lo_nibbles = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
    __m128i hi_nibbles = _mm_srli_epi16(bytes, 4);
    hi_nibbles = _mm_and_si128(hi_nibbles, _mm_set1_epi8(0x0F));

    /* Interleave to get correct order - produces two 128-bit results */
    __m128i interleaved_lo = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
    __m128i interleaved_hi = _mm_unpackhi_epi8(lo_nibbles, hi_nibbles);

    /* Expand each half to 32-bit using AVX-512 (16 x 8-bit -> 16 x 32-bit) */
    __m512i result_lo = _mm512_cvtepu8_epi32(interleaved_lo);
    __m512i result_hi = _mm512_cvtepu8_epi32(interleaved_hi);

    _mm512_storeu_si512((__m512i*)values, result_lo);
    _mm512_storeu_si512((__m512i*)(values + 16), result_hi);
}

/* ============================================================================
 * Byte Stream Split - AVX-512 Optimized
 * ============================================================================
 */

/**
 * Encode floats using byte stream split with AVX-512.
 * Processes 16 floats (64 bytes) at a time using VBMI byte permutation.
 */
void carquet_avx512_byte_stream_split_encode_float(
    const float* values,
    int64_t count,
    uint8_t* output) {

    const uint8_t* src = (const uint8_t*)values;
    int64_t i = 0;

#ifdef __AVX512VBMI__
    /* Single permutation that places all 4 byte streams in the 4 128-bit lanes:
     * Lane 0 (bits 0-127):   byte 0 from each of 16 floats
     * Lane 1 (bits 128-255): byte 1 from each of 16 floats
     * Lane 2 (bits 256-383): byte 2 from each of 16 floats
     * Lane 3 (bits 384-511): byte 3 from each of 16 floats
     */
    /* Use _mm512_set_epi32 instead of _mm512_set_epi8 for GCC 8 compatibility */
    const __m512i perm_all = _mm512_set_epi32(
        0x3F3B3733, 0x2F2B2723, 0x1F1B1713, 0x0F0B0703,  /* byte 3s */
        0x3E3A3632, 0x2E2A2622, 0x1E1A1612, 0x0E0A0602,  /* byte 2s */
        0x3D393531, 0x2D292521, 0x1D191511, 0x0D090501,   /* byte 1s */
        0x3C383430, 0x2C282420, 0x1C181410, 0x0C080400);  /* byte 0s */

    for (; i + 16 <= count; i += 16) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(src + i * 4));

        /* Single permutation gathers all 4 streams */
        __m512i transposed = _mm512_permutexvar_epi8(perm_all, v);

        /* Extract and store each 128-bit lane to its stream */
        _mm_storeu_si128((__m128i*)(output + 0 * count + i), _mm512_castsi512_si128(transposed));
        _mm_storeu_si128((__m128i*)(output + 1 * count + i), _mm512_extracti32x4_epi32(transposed, 1));
        _mm_storeu_si128((__m128i*)(output + 2 * count + i), _mm512_extracti32x4_epi32(transposed, 2));
        _mm_storeu_si128((__m128i*)(output + 3 * count + i), _mm512_extracti32x4_epi32(transposed, 3));
    }
#else
    /* Fallback without VBMI: use shuffle + permutexvar approach
     * Step 1: shuffle_epi8 transposes within each 128-bit lane (4 floats -> 4 bytes per stream)
     * Step 2: permutexvar_epi32 rearranges dwords to group all byte 0s, byte 1s, etc.
     */
    /* Use _mm512_set_epi32 instead of _mm512_set_epi8 for GCC 8 compatibility */
    const __m512i intra_lane_shuf = _mm512_set_epi32(
        0x0F0B0703, 0x0E0A0602, 0x0D090501, 0x0C080400,
        0x0F0B0703, 0x0E0A0602, 0x0D090501, 0x0C080400,
        0x0F0B0703, 0x0E0A0602, 0x0D090501, 0x0C080400,
        0x0F0B0703, 0x0E0A0602, 0x0D090501, 0x0C080400);
    const __m512i cross_lane_perm = _mm512_set_epi32(
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);

    for (; i + 16 <= count; i += 16) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(src + i * 4));

        /* Transpose within each 128-bit lane */
        __m512i shuffled = _mm512_shuffle_epi8(v, intra_lane_shuf);

        /* Rearrange dwords across lanes to group streams */
        __m512i transposed = _mm512_permutexvar_epi32(cross_lane_perm, shuffled);

        /* Extract and store each 128-bit lane to its stream */
        _mm_storeu_si128((__m128i*)(output + 0 * count + i), _mm512_castsi512_si128(transposed));
        _mm_storeu_si128((__m128i*)(output + 1 * count + i), _mm512_extracti32x4_epi32(transposed, 1));
        _mm_storeu_si128((__m128i*)(output + 2 * count + i), _mm512_extracti32x4_epi32(transposed, 2));
        _mm_storeu_si128((__m128i*)(output + 3 * count + i), _mm512_extracti32x4_epi32(transposed, 3));
    }
#endif

    /* Handle remaining values */
    for (; i < count; i++) {
        for (int b = 0; b < 4; b++) {
            output[b * count + i] = src[i * 4 + b];
        }
    }
}

/**
 * Decode byte stream split floats using AVX-512.
 */
void carquet_avx512_byte_stream_split_decode_float(
    const uint8_t* data,
    int64_t count,
    float* values) {

    uint8_t* dst = (uint8_t*)values;
    int64_t i = 0;

    /* Process 16 floats at a time */
    for (; i + 16 <= count; i += 16) {
        /* Load 16 bytes from each of the 4 streams */
        __m128i b0 = _mm_loadu_si128((const __m128i*)(data + 0 * count + i));
        __m128i b1 = _mm_loadu_si128((const __m128i*)(data + 1 * count + i));
        __m128i b2 = _mm_loadu_si128((const __m128i*)(data + 2 * count + i));
        __m128i b3 = _mm_loadu_si128((const __m128i*)(data + 3 * count + i));

        /* Interleave to reconstruct floats */
        __m128i lo01_lo = _mm_unpacklo_epi8(b0, b1);
        __m128i lo01_hi = _mm_unpackhi_epi8(b0, b1);
        __m128i lo23_lo = _mm_unpacklo_epi8(b2, b3);
        __m128i lo23_hi = _mm_unpackhi_epi8(b2, b3);

        __m128i result0 = _mm_unpacklo_epi16(lo01_lo, lo23_lo);
        __m128i result1 = _mm_unpackhi_epi16(lo01_lo, lo23_lo);
        __m128i result2 = _mm_unpacklo_epi16(lo01_hi, lo23_hi);
        __m128i result3 = _mm_unpackhi_epi16(lo01_hi, lo23_hi);

        _mm_storeu_si128((__m128i*)(dst + i * 4 + 0), result0);
        _mm_storeu_si128((__m128i*)(dst + i * 4 + 16), result1);
        _mm_storeu_si128((__m128i*)(dst + i * 4 + 32), result2);
        _mm_storeu_si128((__m128i*)(dst + i * 4 + 48), result3);
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        for (int b = 0; b < 4; b++) {
            dst[i * 4 + b] = data[b * count + i];
        }
    }
}

/**
 * Encode doubles using byte stream split with AVX-512.
 * Processes 8 doubles (64 bytes) at a time using two 256-bit halves.
 */
void carquet_avx512_byte_stream_split_encode_double(
    const double* values,
    int64_t count,
    uint8_t* output) {

    const uint8_t* src = (const uint8_t*)values;
    int64_t i = 0;
    const __m256i s0 = _mm256_setr_epi8(
        0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i s1 = _mm256_setr_epi8(
        1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i s2 = _mm256_setr_epi8(
        2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i s3 = _mm256_setr_epi8(
        3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i s4 = _mm256_setr_epi8(
        4, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        4, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i s5 = _mm256_setr_epi8(
        5, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        5, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i s6 = _mm256_setr_epi8(
        6, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        6, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i s7 = _mm256_setr_epi8(
        7, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        7, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    for (; i + 8 <= count; i += 8) {
        __m256i lo = _mm256_loadu_si256((const __m256i*)(src + i * 8));
        __m256i hi = _mm256_loadu_si256((const __m256i*)(src + i * 8 + 32));

        __m256i out0_lo = _mm256_shuffle_epi8(lo, s0);
        __m256i out1_lo = _mm256_shuffle_epi8(lo, s1);
        __m256i out2_lo = _mm256_shuffle_epi8(lo, s2);
        __m256i out3_lo = _mm256_shuffle_epi8(lo, s3);
        __m256i out4_lo = _mm256_shuffle_epi8(lo, s4);
        __m256i out5_lo = _mm256_shuffle_epi8(lo, s5);
        __m256i out6_lo = _mm256_shuffle_epi8(lo, s6);
        __m256i out7_lo = _mm256_shuffle_epi8(lo, s7);
        __m256i out0_hi = _mm256_shuffle_epi8(hi, s0);
        __m256i out1_hi = _mm256_shuffle_epi8(hi, s1);
        __m256i out2_hi = _mm256_shuffle_epi8(hi, s2);
        __m256i out3_hi = _mm256_shuffle_epi8(hi, s3);
        __m256i out4_hi = _mm256_shuffle_epi8(hi, s4);
        __m256i out5_hi = _mm256_shuffle_epi8(hi, s5);
        __m256i out6_hi = _mm256_shuffle_epi8(hi, s6);
        __m256i out7_hi = _mm256_shuffle_epi8(hi, s7);

        uint64_t t0 = (uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out0_lo), 0) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out0_lo, 1), 0) << 16) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out0_hi), 0) << 32) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out0_hi, 1), 0) << 48);
        uint64_t t1 = (uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out1_lo), 0) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out1_lo, 1), 0) << 16) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out1_hi), 0) << 32) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out1_hi, 1), 0) << 48);
        uint64_t t2 = (uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out2_lo), 0) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out2_lo, 1), 0) << 16) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out2_hi), 0) << 32) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out2_hi, 1), 0) << 48);
        uint64_t t3 = (uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out3_lo), 0) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out3_lo, 1), 0) << 16) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out3_hi), 0) << 32) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out3_hi, 1), 0) << 48);
        uint64_t t4 = (uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out4_lo), 0) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out4_lo, 1), 0) << 16) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out4_hi), 0) << 32) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out4_hi, 1), 0) << 48);
        uint64_t t5 = (uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out5_lo), 0) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out5_lo, 1), 0) << 16) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out5_hi), 0) << 32) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out5_hi, 1), 0) << 48);
        uint64_t t6 = (uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out6_lo), 0) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out6_lo, 1), 0) << 16) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out6_hi), 0) << 32) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out6_hi, 1), 0) << 48);
        uint64_t t7 = (uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out7_lo), 0) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out7_lo, 1), 0) << 16) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_castsi256_si128(out7_hi), 0) << 32) |
                      ((uint64_t)(uint16_t)_mm_extract_epi16(_mm256_extracti128_si256(out7_hi, 1), 0) << 48);

        memcpy(output + 0 * count + i, &t0, sizeof(t0));
        memcpy(output + 1 * count + i, &t1, sizeof(t1));
        memcpy(output + 2 * count + i, &t2, sizeof(t2));
        memcpy(output + 3 * count + i, &t3, sizeof(t3));
        memcpy(output + 4 * count + i, &t4, sizeof(t4));
        memcpy(output + 5 * count + i, &t5, sizeof(t5));
        memcpy(output + 6 * count + i, &t6, sizeof(t6));
        memcpy(output + 7 * count + i, &t7, sizeof(t7));
    }

    for (; i < count; i++) {
        for (int b = 0; b < 8; b++) {
            output[b * count + i] = src[i * 8 + b];
        }
    }
}

/**
 * Decode byte stream split doubles using AVX-512.
 * Processes 8 doubles (64 bytes) at a time via 128-bit interleave stages.
 */
void carquet_avx512_byte_stream_split_decode_double(
    const uint8_t* data,
    int64_t count,
    double* values) {

    uint8_t* dst = (uint8_t*)values;
    int64_t i = 0;

    for (; i + 8 <= count; i += 8) {
        uint64_t b0, b1, b2, b3, b4, b5, b6, b7;
        memcpy(&b0, data + 0 * count + i, sizeof(b0));
        memcpy(&b1, data + 1 * count + i, sizeof(b1));
        memcpy(&b2, data + 2 * count + i, sizeof(b2));
        memcpy(&b3, data + 3 * count + i, sizeof(b3));
        memcpy(&b4, data + 4 * count + i, sizeof(b4));
        memcpy(&b5, data + 5 * count + i, sizeof(b5));
        memcpy(&b6, data + 6 * count + i, sizeof(b6));
        memcpy(&b7, data + 7 * count + i, sizeof(b7));

        __m128i s0 = _mm_cvtsi64_si128((long long)b0);
        __m128i s1 = _mm_cvtsi64_si128((long long)b1);
        __m128i s2 = _mm_cvtsi64_si128((long long)b2);
        __m128i s3 = _mm_cvtsi64_si128((long long)b3);
        __m128i s4 = _mm_cvtsi64_si128((long long)b4);
        __m128i s5 = _mm_cvtsi64_si128((long long)b5);
        __m128i s6 = _mm_cvtsi64_si128((long long)b6);
        __m128i s7 = _mm_cvtsi64_si128((long long)b7);

        __m128i u01_lo = _mm_unpacklo_epi8(s0, s1);
        __m128i u01_hi = _mm_unpackhi_epi8(s0, s1);
        __m128i u23_lo = _mm_unpacklo_epi8(s2, s3);
        __m128i u23_hi = _mm_unpackhi_epi8(s2, s3);
        __m128i u45_lo = _mm_unpacklo_epi8(s4, s5);
        __m128i u45_hi = _mm_unpackhi_epi8(s4, s5);
        __m128i u67_lo = _mm_unpacklo_epi8(s6, s7);
        __m128i u67_hi = _mm_unpackhi_epi8(s6, s7);

        __m128i r0 = _mm_unpacklo_epi16(u01_lo, u23_lo);
        __m128i r1 = _mm_unpackhi_epi16(u01_lo, u23_lo);
        __m128i r2 = _mm_unpacklo_epi16(u01_hi, u23_hi);
        __m128i r3 = _mm_unpackhi_epi16(u01_hi, u23_hi);
        __m128i r4 = _mm_unpacklo_epi16(u45_lo, u67_lo);
        __m128i r5 = _mm_unpackhi_epi16(u45_lo, u67_lo);
        __m128i r6 = _mm_unpacklo_epi16(u45_hi, u67_hi);
        __m128i r7 = _mm_unpackhi_epi16(u45_hi, u67_hi);

        __m128i d0 = _mm_unpacklo_epi32(r0, r4);
        __m128i d1 = _mm_unpackhi_epi32(r0, r4);
        __m128i d2 = _mm_unpacklo_epi32(r1, r5);
        __m128i d3 = _mm_unpackhi_epi32(r1, r5);
        __m128i d4 = _mm_unpacklo_epi32(r2, r6);
        __m128i d5 = _mm_unpackhi_epi32(r2, r6);
        __m128i d6 = _mm_unpacklo_epi32(r3, r7);
        __m128i d7 = _mm_unpackhi_epi32(r3, r7);

        _mm_storeu_si128((__m128i*)(dst + i * 8 +  0), d0);
        _mm_storeu_si128((__m128i*)(dst + i * 8 + 16), d1);
        _mm_storeu_si128((__m128i*)(dst + i * 8 + 32), d2);
        _mm_storeu_si128((__m128i*)(dst + i * 8 + 48), d3);
        _mm_storeu_si128((__m128i*)(dst + i * 8 + 64), d4);
        _mm_storeu_si128((__m128i*)(dst + i * 8 + 80), d5);
        _mm_storeu_si128((__m128i*)(dst + i * 8 + 96), d6);
        _mm_storeu_si128((__m128i*)(dst + i * 8 + 112), d7);
    }

    for (; i < count; i++) {
        for (int b = 0; b < 8; b++) {
            dst[i * 8 + b] = data[b * count + i];
        }
    }
}

/* ============================================================================
 * Delta Decoding - AVX-512 Optimized (Prefix Sum)
 * ============================================================================
 */

/**
 * Apply prefix sum (cumulative sum) to int32 array using AVX-512.
 */
void carquet_avx512_prefix_sum_i32(int32_t* values, int64_t count, int32_t initial) {
    int32_t sum = initial;
    int64_t i = 0;

    /* AVX-512 prefix sum for 16 elements at a time */
    for (; i + 16 <= count; i += 16) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(values + i));

        /* Multi-step prefix sum within vector */
        /* Step 1: Add adjacent pairs */
        __m512i shifted1 = _mm512_maskz_alignr_epi32(0xFFFE, v, _mm512_setzero_si512(), 15);
        v = _mm512_add_epi32(v, shifted1);

        /* Step 2: Add elements 2 apart */
        __m512i shifted2 = _mm512_maskz_alignr_epi32(0xFFFC, v, _mm512_setzero_si512(), 14);
        v = _mm512_add_epi32(v, shifted2);

        /* Step 3: Add elements 4 apart */
        __m512i shifted4 = _mm512_maskz_alignr_epi32(0xFFF0, v, _mm512_setzero_si512(), 12);
        v = _mm512_add_epi32(v, shifted4);

        /* Step 4: Add elements 8 apart */
        __m512i shifted8 = _mm512_maskz_alignr_epi32(0xFF00, v, _mm512_setzero_si512(), 8);
        v = _mm512_add_epi32(v, shifted8);

        /* Add running sum */
        __m512i sums = _mm512_set1_epi32(sum);
        v = _mm512_add_epi32(v, sums);
        _mm512_storeu_si512((__m512i*)(values + i), v);

        /* Update running sum to last element */
        sum = values[i + 15];
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        sum += values[i];
        values[i] = sum;
    }
}

/**
 * Apply prefix sum to int64 array using AVX-512.
 */
void carquet_avx512_prefix_sum_i64(int64_t* values, int64_t count, int64_t initial) {
    int64_t sum = initial;
    int64_t i = 0;

    /* AVX-512 prefix sum for 8 elements at a time */
    for (; i + 8 <= count; i += 8) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(values + i));

        /* Multi-step prefix sum */
        __m512i shifted1 = _mm512_maskz_alignr_epi64(0xFE, v, _mm512_setzero_si512(), 7);
        v = _mm512_add_epi64(v, shifted1);

        __m512i shifted2 = _mm512_maskz_alignr_epi64(0xFC, v, _mm512_setzero_si512(), 6);
        v = _mm512_add_epi64(v, shifted2);

        __m512i shifted4 = _mm512_maskz_alignr_epi64(0xF0, v, _mm512_setzero_si512(), 4);
        v = _mm512_add_epi64(v, shifted4);

        /* Add running sum */
        __m512i sums = _mm512_set1_epi64(sum);
        v = _mm512_add_epi64(v, sums);
        _mm512_storeu_si512((__m512i*)(values + i), v);

        /* Update running sum */
        sum = values[i + 7];
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        sum += values[i];
        values[i] = sum;
    }
}

/* ============================================================================
 * Dictionary Gather - AVX-512 Optimized
 * ============================================================================
 */

/**
 * Gather int32 values from dictionary using AVX-512 gather instructions.
 */
void carquet_avx512_gather_i32(const int32_t* dict, const uint32_t* indices,
                               int64_t count, int32_t* output) {
    int64_t i = 0;

    /* Process 16 at a time using AVX-512 gather */
    for (; i + 16 <= count; i += 16) {
        __m512i idx = _mm512_loadu_si512((const __m512i*)(indices + i));
        __m512i result = _mm512_i32gather_epi32(idx, dict, 4);
        _mm512_storeu_si512((__m512i*)(output + i), result);
    }

    /* Handle remaining with AVX2 */
    for (; i + 8 <= count; i += 8) {
        __m256i idx = _mm256_loadu_si256((const __m256i*)(indices + i));
        __m256i result = _mm256_i32gather_epi32(dict, idx, 4);
        _mm256_storeu_si256((__m256i*)(output + i), result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/**
 * Gather int64 values from dictionary using AVX-512 gather instructions.
 */
void carquet_avx512_gather_i64(const int64_t* dict, const uint32_t* indices,
                               int64_t count, int64_t* output) {
    int64_t i = 0;

    /* Process 8 at a time using AVX-512 gather */
    for (; i + 8 <= count; i += 8) {
        __m256i idx = _mm256_loadu_si256((const __m256i*)(indices + i));
        __m512i result = _mm512_i32gather_epi64(idx, dict, 8);
        _mm512_storeu_si512((__m512i*)(output + i), result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/**
 * Gather float values from dictionary using AVX-512 gather instructions.
 * Note: float and int32 are both 4 bytes, so we reuse gather_i32 via cast.
 */
void carquet_avx512_gather_float(const float* dict, const uint32_t* indices,
                                  int64_t count, float* output) {
    /* Data movement doesn't care about type - reuse int32 implementation */
    carquet_avx512_gather_i32((const int32_t*)dict, indices, count, (int32_t*)output);
}

/**
 * Gather double values from dictionary using AVX-512 gather instructions.
 * Note: double and int64 are both 8 bytes, so we reuse gather_i64 via cast.
 */
void carquet_avx512_gather_double(const double* dict, const uint32_t* indices,
                                   int64_t count, double* output) {
    /* Data movement doesn't care about type - reuse int64 implementation */
    carquet_avx512_gather_i64((const int64_t*)dict, indices, count, (int64_t*)output);
}

bool carquet_avx512_checked_gather_i32(const int32_t* dict, int32_t dict_count,
                                        const uint32_t* indices, int64_t count,
                                        int32_t* output) {
    int64_t i = 0;
    __m512i limit = _mm512_set1_epi32(dict_count);

    for (; i + 16 <= count; i += 16) {
        __m512i idx = _mm512_loadu_si512((const void*)(indices + i));
        __mmask16 valid = _mm512_cmp_epu32_mask(idx, limit, _MM_CMPINT_LT);
        if (valid != 0xFFFFu) {
            return false;
        }
        __m512i result = _mm512_i32gather_epi32(idx, dict, 4);
        _mm512_storeu_si512((void*)(output + i), result);
    }

    for (; i < count; i++) {
        uint32_t idx = indices[i];
        if (idx >= (uint32_t)dict_count) {
            return false;
        }
        output[i] = dict[idx];
    }

    return true;
}

bool carquet_avx512_checked_gather_i64(const int64_t* dict, int32_t dict_count,
                                        const uint32_t* indices, int64_t count,
                                        int64_t* output) {
    int64_t i = 0;
    __m256i limit = _mm256_set1_epi32(dict_count);

    for (; i + 8 <= count; i += 8) {
        __m256i idx = _mm256_loadu_si256((const __m256i*)(indices + i));
        __mmask8 valid = _mm256_cmp_epu32_mask(idx, limit, _MM_CMPINT_LT);
        if (valid != 0xFFu) {
            return false;
        }
        __m512i result = _mm512_i32gather_epi64(idx, dict, 8);
        _mm512_storeu_si512((void*)(output + i), result);
    }

    for (; i < count; i++) {
        uint32_t idx = indices[i];
        if (idx >= (uint32_t)dict_count) {
            return false;
        }
        output[i] = dict[idx];
    }

    return true;
}

bool carquet_avx512_checked_gather_float(const float* dict, int32_t dict_count,
                                          const uint32_t* indices, int64_t count,
                                          float* output) {
    return carquet_avx512_checked_gather_i32(
        (const int32_t*)dict, dict_count, indices, count, (int32_t*)output);
}

bool carquet_avx512_checked_gather_double(const double* dict, int32_t dict_count,
                                           const uint32_t* indices, int64_t count,
                                           double* output) {
    return carquet_avx512_checked_gather_i64(
        (const int64_t*)dict, dict_count, indices, count, (int64_t*)output);
}

/* ============================================================================
 * Memcpy/Memset - AVX-512 Optimized
 * ============================================================================
 */

/**
 * Fast memset for large buffers using AVX-512.
 */
void carquet_avx512_memset(void* dest, uint8_t value, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    __m512i v = _mm512_set1_epi8((char)value);

    while (n >= 256) {
        _mm512_storeu_si512((__m512i*)(d + 0), v);
        _mm512_storeu_si512((__m512i*)(d + 64), v);
        _mm512_storeu_si512((__m512i*)(d + 128), v);
        _mm512_storeu_si512((__m512i*)(d + 192), v);
        d += 256;
        n -= 256;
    }

    while (n >= 64) {
        _mm512_storeu_si512((__m512i*)d, v);
        d += 64;
        n -= 64;
    }

    /* Handle tail with AVX2/SSE */
    __m256i v256 = _mm256_set1_epi8((char)value);
    while (n >= 32) {
        _mm256_storeu_si256((__m256i*)d, v256);
        d += 32;
        n -= 32;
    }

    __m128i v128 = _mm_set1_epi8((char)value);
    while (n >= 16) {
        _mm_storeu_si128((__m128i*)d, v128);
        d += 16;
        n -= 16;
    }

    while (n > 0) {
        *d++ = value;
        n--;
    }
}

/**
 * Fast memcpy for large buffers using AVX-512.
 */
void carquet_avx512_memcpy(void* dest, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    const uint8_t* s = (const uint8_t*)src;

    while (n >= 256) {
        __m512i v0 = _mm512_loadu_si512((const __m512i*)(s + 0));
        __m512i v1 = _mm512_loadu_si512((const __m512i*)(s + 64));
        __m512i v2 = _mm512_loadu_si512((const __m512i*)(s + 128));
        __m512i v3 = _mm512_loadu_si512((const __m512i*)(s + 192));
        _mm512_storeu_si512((__m512i*)(d + 0), v0);
        _mm512_storeu_si512((__m512i*)(d + 64), v1);
        _mm512_storeu_si512((__m512i*)(d + 128), v2);
        _mm512_storeu_si512((__m512i*)(d + 192), v3);
        d += 256;
        s += 256;
        n -= 256;
    }

    while (n >= 64) {
        _mm512_storeu_si512((__m512i*)d, _mm512_loadu_si512((const __m512i*)s));
        d += 64;
        s += 64;
        n -= 64;
    }

    while (n >= 32) {
        _mm256_storeu_si256((__m256i*)d, _mm256_loadu_si256((const __m256i*)s));
        d += 32;
        s += 32;
        n -= 32;
    }

    while (n >= 16) {
        _mm_storeu_si128((__m128i*)d, _mm_loadu_si128((const __m128i*)s));
        d += 16;
        s += 16;
        n -= 16;
    }

    while (n > 0) {
        *d++ = *s++;
        n--;
    }
}

uint32_t carquet_avx512_crc32c(uint32_t crc, const uint8_t* data, size_t len) {
    size_t i = 0;

#ifdef __x86_64__
    for (; i + 8 <= len; i += 8) {
        uint64_t val;
        memcpy(&val, data + i, 8);
        crc = (uint32_t)_mm_crc32_u64(crc, val);
    }
#endif

    for (; i + 4 <= len; i += 4) {
        uint32_t val;
        memcpy(&val, data + i, 4);
        crc = _mm_crc32_u32(crc, val);
    }

    if (i + 2 <= len) {
        uint16_t val;
        memcpy(&val, data + i, 2);
        crc = _mm_crc32_u16(crc, val);
        i += 2;
    }

    if (i < len) {
        crc = _mm_crc32_u8(crc, data[i]);
    }

    return crc;
}

void carquet_avx512_match_copy(uint8_t* dst, const uint8_t* src, size_t len, size_t offset) {
    if (offset >= 64) {
        while (len >= 64) {
            _mm512_storeu_si512((void*)dst, _mm512_loadu_si512((const void*)src));
            dst += 64;
            src += 64;
            len -= 64;
        }
    } else if (offset == 1) {
        __m512i v = _mm512_set1_epi8((char)*src);
        while (len >= 64) {
            _mm512_storeu_si512((void*)dst, v);
            dst += 64;
            len -= 64;
        }
    } else if (offset == 4) {
        uint32_t pattern;
        memcpy(&pattern, src, sizeof(pattern));
        __m512i v = _mm512_set1_epi32((int32_t)pattern);
        while (len >= 64) {
            _mm512_storeu_si512((void*)dst, v);
            dst += 64;
            len -= 64;
        }
    } else if (offset == 8) {
        uint64_t pattern;
        memcpy(&pattern, src, sizeof(pattern));
        __m512i v = _mm512_set1_epi64((long long)pattern);
        while (len >= 64) {
            _mm512_storeu_si512((void*)dst, v);
            dst += 64;
            len -= 64;
        }
    }

    while (len > 0) {
        *dst++ = *src++;
        len--;
    }
}

size_t carquet_avx512_match_length(const uint8_t* p, const uint8_t* match, const uint8_t* limit) {
    const uint8_t* start = p;

    while (p + 64 <= limit) {
        __m512i a = _mm512_loadu_si512((const void*)p);
        __m512i b = _mm512_loadu_si512((const void*)match);
        __mmask64 mask = _mm512_cmpeq_epi8_mask(a, b);
        if (mask != ~0ULL) {
            return (size_t)(p - start) + (size_t)portable_ctz64(~mask);
        }
        p += 64;
        match += 64;
    }

    while (p < limit && *p == *match) {
        p++;
        match++;
    }

    return (size_t)(p - start);
}

/* ============================================================================
 * Boolean Operations - AVX-512 Optimized
 * ============================================================================
 */

/**
 * Unpack boolean values from packed bits to byte array using AVX-512.
 */
void carquet_avx512_unpack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    int64_t i = 0;

    /* Process 64 bools (8 bytes) at a time using AVX-512 mask */
    for (; i + 64 <= count; i += 64) {
        int byte_idx = (int)(i / 8);
        uint64_t packed;
        memcpy(&packed, input + byte_idx, 8);

        /* Convert to mask and create result with maskz_set1 (1 where set, 0 otherwise) */
        __m512i result = _mm512_maskz_set1_epi8((__mmask64)packed, 1);

        _mm512_storeu_si512((__m512i*)(output + i), result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        int byte_idx = (int)(i / 8);
        int bit_idx = (int)(i % 8);
        output[i] = (input[byte_idx] >> bit_idx) & 1;
    }
}

/**
 * Pack boolean values from byte array to packed bits using AVX-512.
 */
void carquet_avx512_pack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    int64_t i = 0;

    /* Process 64 bools at a time */
    for (; i + 64 <= count; i += 64) {
        __m512i bools = _mm512_loadu_si512((const __m512i*)(input + i));

        /* Use test_epi8_mask: bit is set if (a & b) != 0, i.e., if bool is non-zero */
        __mmask64 mask = _mm512_test_epi8_mask(bools, bools);

        /* Store mask as 8 bytes */
        uint64_t packed = (uint64_t)mask;
        memcpy(output + i / 8, &packed, 8);
    }

    /* Handle remaining elements with masked load */
    if (i < count) {
        int64_t remaining = count - i;
        /* Create mask for remaining elements: set bits 0..(remaining-1) */
        __mmask64 load_mask = (remaining >= 64) ? ~0ULL : ((1ULL << remaining) - 1);

        /* Masked load zeros out elements beyond the mask */
        __m512i bools = _mm512_maskz_loadu_epi8(load_mask, input + i);

        /* Test for non-zero values */
        __mmask64 result_mask = _mm512_test_epi8_mask(bools, bools);

        /* Write only the bytes we need */
        int64_t bytes_to_write = (remaining + 7) / 8;
        uint64_t packed = (uint64_t)result_mask;
        memcpy(output + i / 8, &packed, (size_t)bytes_to_write);
    }
}

/* ============================================================================
 * Run Detection - AVX-512 Optimized
 * ============================================================================
 */

/**
 * Find the length of a run of repeated int32 values.
 */
int64_t carquet_avx512_find_run_length_i32(const int32_t* values, int64_t count) {
    if (count == 0) return 0;

    int32_t first = values[0];
    __m512i target = _mm512_set1_epi32(first);
    int64_t i = 0;

    /* Check 16 at a time */
    for (; i + 16 <= count; i += 16) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(values + i));
        __mmask16 cmp = _mm512_cmpeq_epi32_mask(v, target);

        if (cmp != 0xFFFF) {  /* Not all equal */
            /* Find first mismatch using trailing zeros */
            int tz = portable_ctz(~cmp);
            return i + tz;
        }
    }

    /* Handle remaining */
    for (; i < count; i++) {
        if (values[i] != first) {
            return i;
        }
    }

    return count;
}

int64_t carquet_avx512_count_non_nulls(const int16_t* def_levels, int64_t count, int16_t max_def_level) {
    int64_t non_null_count = 0;
    int64_t i = 0;
    __m512i max_vec = _mm512_set1_epi16(max_def_level);

    for (; i + 32 <= count; i += 32) {
        __m512i levels = _mm512_loadu_si512((const void*)(def_levels + i));
        __mmask32 mask = _mm512_cmpeq_epi16_mask(levels, max_vec);
        non_null_count += _mm_popcnt_u32((unsigned int)mask);
    }

    for (; i < count; i++) {
        if (def_levels[i] == max_def_level) {
            non_null_count++;
        }
    }

    return non_null_count;
}

void carquet_avx512_build_null_bitmap(const int16_t* def_levels, int64_t count,
                                       int16_t max_def_level, uint8_t* null_bitmap) {
    int64_t i = 0;
    int64_t byte_index = 0;
    __m512i max_vec = _mm512_set1_epi16(max_def_level);

    for (; i + 32 <= count; i += 32, byte_index += 4) {
        __m512i levels = _mm512_loadu_si512((const void*)(def_levels + i));
        __mmask32 mask = _mm512_cmp_epi16_mask(levels, max_vec, _MM_CMPINT_LT);
        uint32_t bits = (uint32_t)mask;
        memcpy(null_bitmap + byte_index, &bits, sizeof(bits));
    }

    for (; i < count; byte_index++) {
        uint8_t bits = 0;
        for (int j = 0; j < 8 && i < count; j++, i++) {
            if (def_levels[i] < max_def_level) {
                bits |= (uint8_t)(1u << j);
            }
        }
        null_bitmap[byte_index] = bits;
    }
}

void carquet_avx512_fill_def_levels(int16_t* def_levels, int64_t count, int16_t value) {
    int64_t i = 0;
    __m512i val_vec = _mm512_set1_epi16(value);

    for (; i + 32 <= count; i += 32) {
        _mm512_storeu_si512((void*)(def_levels + i), val_vec);
    }
    for (; i < count; i++) {
        def_levels[i] = value;
    }
}

void carquet_avx512_minmax_i32(const int32_t* values, int64_t count,
                                int32_t* min_value, int32_t* max_value) {
    int32_t min_v = values[0];
    int32_t max_v = values[0];
    __m512i min_vec = _mm512_set1_epi32(min_v);
    __m512i max_vec = _mm512_set1_epi32(max_v);
    int64_t i = 1;

    for (; i + 16 <= count; i += 16) {
        __m512i v = _mm512_loadu_si512((const void*)(values + i));
        min_vec = _mm512_min_epi32(min_vec, v);
        max_vec = _mm512_max_epi32(max_vec, v);
    }

    int32_t tmp_min[16];
    int32_t tmp_max[16];
    _mm512_storeu_si512((void*)tmp_min, min_vec);
    _mm512_storeu_si512((void*)tmp_max, max_vec);
    for (int j = 0; j < 16; j++) {
        if (tmp_min[j] < min_v) min_v = tmp_min[j];
        if (tmp_max[j] > max_v) max_v = tmp_max[j];
    }
    for (; i < count; i++) {
        if (values[i] < min_v) min_v = values[i];
        if (values[i] > max_v) max_v = values[i];
    }
    *min_value = min_v;
    *max_value = max_v;
}

void carquet_avx512_minmax_i64(const int64_t* values, int64_t count,
                                int64_t* min_value, int64_t* max_value) {
    int64_t min_v = values[0];
    int64_t max_v = values[0];
    __m512i min_vec = _mm512_set1_epi64(min_v);
    __m512i max_vec = _mm512_set1_epi64(max_v);
    int64_t i = 1;

    for (; i + 8 <= count; i += 8) {
        __m512i v = _mm512_loadu_si512((const void*)(values + i));
        __mmask8 lt = _mm512_cmpgt_epi64_mask(min_vec, v);
        __mmask8 gt = _mm512_cmpgt_epi64_mask(v, max_vec);
        min_vec = _mm512_mask_mov_epi64(min_vec, lt, v);
        max_vec = _mm512_mask_mov_epi64(max_vec, gt, v);
    }

    int64_t tmp_min[8];
    int64_t tmp_max[8];
    _mm512_storeu_si512((void*)tmp_min, min_vec);
    _mm512_storeu_si512((void*)tmp_max, max_vec);
    for (int j = 0; j < 8; j++) {
        if (tmp_min[j] < min_v) min_v = tmp_min[j];
        if (tmp_max[j] > max_v) max_v = tmp_max[j];
    }
    for (; i < count; i++) {
        if (values[i] < min_v) min_v = values[i];
        if (values[i] > max_v) max_v = values[i];
    }
    *min_value = min_v;
    *max_value = max_v;
}

void carquet_avx512_minmax_float(const float* values, int64_t count,
                                  float* min_value, float* max_value) {
    float min_v = values[0];
    float max_v = values[0];
    __m512 min_vec = _mm512_set1_ps(min_v);
    __m512 max_vec = _mm512_set1_ps(max_v);
    int64_t i = 1;

    for (; i + 16 <= count; i += 16) {
        __m512 v = _mm512_loadu_ps(values + i);
        __mmask16 lt = _mm512_cmplt_ps_mask(v, min_vec);
        __mmask16 gt = _mm512_cmp_ps_mask(v, max_vec, _CMP_GT_OQ);
        min_vec = _mm512_mask_mov_ps(min_vec, lt, v);
        max_vec = _mm512_mask_mov_ps(max_vec, gt, v);
    }

    float tmp_min[16];
    float tmp_max[16];
    _mm512_storeu_ps(tmp_min, min_vec);
    _mm512_storeu_ps(tmp_max, max_vec);
    for (int j = 0; j < 16; j++) {
        if (tmp_min[j] < min_v) min_v = tmp_min[j];
        if (tmp_max[j] > max_v) max_v = tmp_max[j];
    }
    for (; i < count; i++) {
        if (values[i] < min_v) min_v = values[i];
        if (values[i] > max_v) max_v = values[i];
    }
    *min_value = min_v;
    *max_value = max_v;
}

void carquet_avx512_minmax_double(const double* values, int64_t count,
                                   double* min_value, double* max_value) {
    double min_v = values[0];
    double max_v = values[0];
    __m512d min_vec = _mm512_set1_pd(min_v);
    __m512d max_vec = _mm512_set1_pd(max_v);
    int64_t i = 1;

    for (; i + 8 <= count; i += 8) {
        __m512d v = _mm512_loadu_pd(values + i);
        __mmask8 lt = _mm512_cmplt_pd_mask(v, min_vec);
        __mmask8 gt = _mm512_cmp_pd_mask(v, max_vec, _CMP_GT_OQ);
        min_vec = _mm512_mask_mov_pd(min_vec, lt, v);
        max_vec = _mm512_mask_mov_pd(max_vec, gt, v);
    }

    double tmp_min[8];
    double tmp_max[8];
    _mm512_storeu_pd(tmp_min, min_vec);
    _mm512_storeu_pd(tmp_max, max_vec);
    for (int j = 0; j < 8; j++) {
        if (tmp_min[j] < min_v) min_v = tmp_min[j];
        if (tmp_max[j] > max_v) max_v = tmp_max[j];
    }
    for (; i < count; i++) {
        if (values[i] < min_v) min_v = values[i];
        if (values[i] > max_v) max_v = values[i];
    }
    *min_value = min_v;
    *max_value = max_v;
}

/* ============================================================================
 * Conflict Detection - AVX-512 Specific
 * ============================================================================
 */

#ifdef __AVX512CD__

/**
 * Detect conflicts in indices for scatter operations.
 * Returns a mask where bit i is set if indices[i] conflicts with any earlier index.
 */
__mmask16 carquet_avx512_detect_conflicts_i32(const uint32_t* indices) {
    __m512i idx = _mm512_loadu_si512((const __m512i*)indices);
    __m512i conflicts = _mm512_conflict_epi32(idx);

    /* Non-zero conflict value means there's a conflict */
    return _mm512_cmpneq_epi32_mask(conflicts, _mm512_setzero_si512());
}

#endif /* __AVX512CD__ */

#endif /* __AVX512F__ */
#endif /* x86_64 */
