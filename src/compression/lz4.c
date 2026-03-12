/**
 * @file lz4.c
 * @brief LZ4 compression/decompression wrapper using the official lz4 library
 *
 * Implements LZ4 block format (LZ4_RAW) as used by Apache Parquet.
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <lz4.h>

/* ============================================================================
 * LZ4 Decompression
 * ============================================================================
 */

carquet_status_t carquet_lz4_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (src_size == 0) {
        *dst_size = 0;
        return CARQUET_OK;
    }

    if (!src) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (src_size > (size_t)INT_MAX || dst_capacity > (size_t)INT_MAX) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    int result = LZ4_decompress_safe(
        (const char*)src, (char*)dst,
        (int)src_size, (int)dst_capacity);

    if (result < 0) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    *dst_size = (size_t)result;
    return CARQUET_OK;
}

/* ============================================================================
 * LZ4 Compression
 * ============================================================================
 */

carquet_status_t carquet_lz4_compress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (src_size == 0) {
        *dst_size = 0;
        return CARQUET_OK;
    }

    if (!src) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (src_size > (size_t)INT_MAX || dst_capacity > (size_t)INT_MAX) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    int result = LZ4_compress_default(
        (const char*)src, (char*)dst,
        (int)src_size, (int)dst_capacity);

    if (result <= 0) {
        return CARQUET_ERROR_COMPRESSION;
    }

    *dst_size = (size_t)result;
    return CARQUET_OK;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================
 */

size_t carquet_lz4_compress_bound(size_t src_size) {
    if (src_size > (size_t)INT_MAX) return 0;
    return (size_t)LZ4_compressBound((int)src_size);
}
