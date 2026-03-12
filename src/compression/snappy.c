/**
 * @file snappy.c
 * @brief Pure C Snappy compression/decompression
 *
 * Reference: https://github.com/google/snappy/blob/main/format_description.txt
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

extern size_t carquet_dispatch_match_length(const uint8_t* p, const uint8_t* match, const uint8_t* limit);

/* ============================================================================
 * Constants
 * ============================================================================
 */

#define SNAPPY_LITERAL         0
#define SNAPPY_COPY_1          1
#define SNAPPY_COPY_2          2
#define SNAPPY_COPY_4          3

#define SNAPPY_HASH_LOG        14
#define SNAPPY_HASH_SIZE       (1 << SNAPPY_HASH_LOG)
#define SNAPPY_MAX_OFFSET      65535
#define SNAPPY_BLOCK_SIZE      (1 << 16)

/* Forward declaration */
size_t carquet_snappy_compress_bound(size_t src_size);

/* ============================================================================
 * Varint Encoding
 * ============================================================================
 */

static size_t snappy_read_varint(const uint8_t* p, const uint8_t* end, uint32_t* value) {
    *value = 0;
    int shift = 0;
    const uint8_t* start = p;

    while (p < end) {
        uint8_t b = *p++;
        *value |= ((uint32_t)(b & 0x7F)) << shift;
        if ((b & 0x80) == 0) {
            return (size_t)(p - start);
        }
        shift += 7;
        if (shift >= 32) {
            return 0; /* Overflow */
        }
    }
    return 0; /* Truncated */
}

static size_t snappy_write_varint(uint8_t* p, uint32_t value) {
    uint8_t* start = p;
    while (value >= 0x80) {
        *p++ = (uint8_t)(value | 0x80);
        value >>= 7;
    }
    *p++ = (uint8_t)value;
    return (size_t)(p - start);
}

/* ============================================================================
 * Snappy Decompression
 * ============================================================================
 */

static inline void snappy_match_copy_fast(uint8_t* dst, size_t len, size_t offset) {
    const uint8_t* src = dst - offset;

    if (offset == 1) {
        memset(dst, *src, len);
        return;
    }

    if (offset == 2) {
        uint16_t pattern16;
        uint32_t pattern32;
        uint64_t pattern64;

        memcpy(&pattern16, src, sizeof(pattern16));
        pattern32 = (uint32_t)pattern16 | ((uint32_t)pattern16 << 16);
        pattern64 = (uint64_t)pattern32 | ((uint64_t)pattern32 << 32);

        while (len >= 8) {
            memcpy(dst, &pattern64, sizeof(pattern64));
            dst += 8;
            len -= 8;
        }
        while (len >= 2) {
            memcpy(dst, &pattern16, sizeof(pattern16));
            dst += 2;
            len -= 2;
        }
        if (len) {
            *dst = *(const uint8_t*)&pattern16;
        }
        return;
    }

    if (offset == 4) {
        uint32_t pattern32;
        uint64_t pattern64;
        memcpy(&pattern32, src, 4);
        pattern64 = (uint64_t)pattern32 | ((uint64_t)pattern32 << 32);

        while (len >= 8) {
            memcpy(dst, &pattern64, sizeof(pattern64));
            dst += 8;
            len -= 8;
        }
        while (len > 0) {
            *dst++ = *src++;
            len--;
        }
        return;
    }

    if (offset < 8) {
        /* For offsets 3, 5, 6, 7 the 8-byte tiling trick doesn't work
         * (8 % offset != 0 means the pattern drifts after each 8-byte copy).
         * Use byte-by-byte copy from the repeating source. */
        for (size_t i = 0; i < len; i++) {
            dst[i] = src[i % offset];
        }
        return;
    }

    if (offset < 16) {
        while (len >= 8) {
            uint64_t chunk;
            memcpy(&chunk, src, sizeof(chunk));
            memcpy(dst, &chunk, sizeof(chunk));
            dst += 8;
            src += 8;
            len -= 8;
        }
        while (len > 0) {
            *dst++ = *src++;
            len--;
        }
        return;
    }

    while (len >= 16) {
        uint64_t lo;
        uint64_t hi;
        memcpy(&lo, src, sizeof(lo));
        memcpy(&hi, src + 8, sizeof(hi));
        memcpy(dst, &lo, sizeof(lo));
        memcpy(dst + 8, &hi, sizeof(hi));
        dst += 16;
        src += 16;
        len -= 16;
    }
    if (len >= 8) {
        uint64_t chunk;
        memcpy(&chunk, src, sizeof(chunk));
        memcpy(dst, &chunk, sizeof(chunk));
        dst += 8;
        src += 8;
        len -= 8;
    }
    while (len > 0) {
        *dst++ = *src++;
        len--;
    }
}

carquet_status_t carquet_snappy_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!src || !dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    const uint8_t* ip = src;
    const uint8_t* const iend = src + src_size;

    /* Read uncompressed length */
    uint32_t uncompressed_len;
    size_t varint_len = snappy_read_varint(ip, iend, &uncompressed_len);
    if (varint_len == 0) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }
    ip += varint_len;

    if (uncompressed_len > dst_capacity) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    uint8_t* op = dst;
    uint8_t* const oend = dst + uncompressed_len;

    while (ip < iend && op < oend) {
        uint8_t tag = *ip++;
        uint8_t type = tag & 0x03;

        if (type == SNAPPY_LITERAL) {
            /* Literal */
            size_t len = (tag >> 2) + 1;

            if (len > 60) {
                /* Extended length */
                size_t extra_bytes = len - 60;
                if (ip + extra_bytes > iend) {
                    return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
                }
                len = 1;
                for (size_t i = 0; i < extra_bytes; i++) {
                    len += (size_t)ip[i] << (8 * i);
                }
                ip += extra_bytes;
            }

            if (ip + len > iend || op + len > oend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            memcpy(op, ip, len);
            ip += len;
            op += len;

        } else if (type == SNAPPY_COPY_1) {
            /* Copy with 1-byte offset */
            size_t len = ((tag >> 2) & 0x07) + 4;
            size_t offset = ((tag >> 5) << 8) | *ip++;

            if (offset == 0 || offset > (size_t)(op - dst)) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            if (op + len > oend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            snappy_match_copy_fast(op, len, offset);
            op += len;

        } else if (type == SNAPPY_COPY_2) {
            /* Copy with 2-byte offset */
            size_t len = ((tag >> 2) & 0x3F) + 1;
            if (ip + 2 > iend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            size_t offset = ip[0] | ((size_t)ip[1] << 8);
            ip += 2;

            if (offset == 0 || offset > (size_t)(op - dst)) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            if (op + len > oend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            snappy_match_copy_fast(op, len, offset);
            op += len;

        } else { /* SNAPPY_COPY_4 */
            /* Copy with 4-byte offset */
            size_t len = ((tag >> 2) & 0x3F) + 1;
            if (ip + 4 > iend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            size_t offset = ip[0] | ((size_t)ip[1] << 8) |
                           ((size_t)ip[2] << 16) | ((size_t)ip[3] << 24);
            ip += 4;

            if (offset == 0 || offset > (size_t)(op - dst)) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            if (op + len > oend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            snappy_match_copy_fast(op, len, offset);
            op += len;
        }
    }

    if ((size_t)(op - dst) != uncompressed_len) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    *dst_size = uncompressed_len;
    return CARQUET_OK;
}

/* ============================================================================
 * Snappy Compression
 * ============================================================================
 */

static inline uint32_t snappy_hash(uint32_t val) {
    return (val * 0x1e35a7bd) >> (32 - SNAPPY_HASH_LOG);
}

static inline uint32_t snappy_read32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

/* Prefetch hint - no-op on compilers without support */
#if defined(__GNUC__) || defined(__clang__)
#define SNAPPY_PREFETCH(addr) __builtin_prefetch((addr), 0, 1)
#elif defined(_MSC_VER)
#include <xmmintrin.h>
#define SNAPPY_PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T1)
#else
#define SNAPPY_PREFETCH(addr) ((void)0)
#endif

/* Fast inline match length using 64-bit XOR comparison.
 * Avoids SIMD dispatch overhead on the hot compression path.
 * Assumes little-endian byte order (x86/ARM). */
static inline size_t fast_match_length(const uint8_t* p, const uint8_t* match,
                                        const uint8_t* limit) {
    const uint8_t* start = p;

    while (p + 8 <= limit) {
        uint64_t a, b;
        memcpy(&a, p, 8);
        memcpy(&b, match, 8);
        uint64_t diff = a ^ b;
        if (diff) {
#if defined(__GNUC__) || defined(__clang__)
            return (size_t)(p - start) + ((size_t)__builtin_ctzll(diff) >> 3);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
            unsigned long idx;
            _BitScanForward64(&idx, diff);
            return (size_t)(p - start) + (idx >> 3);
#else
            for (size_t i = 0; i < 8; i++) {
                if (p[i] != match[i]) return (size_t)(p - start) + i;
            }
#endif
        }
        p += 8;
        match += 8;
    }

    while (p < limit && *p == *match) {
        p++;
        match++;
    }
    return (size_t)(p - start);
}

static uint8_t* snappy_emit_literal(uint8_t* op, const uint8_t* literal, size_t len) {
    if (len <= 60) {
        *op++ = (uint8_t)((len - 1) << 2);
    } else if (len <= 256) {
        *op++ = (60 << 2);
        *op++ = (uint8_t)(len - 1);
    } else if (len <= 65536) {
        *op++ = (61 << 2);
        *op++ = (uint8_t)((len - 1) & 0xFF);
        *op++ = (uint8_t)((len - 1) >> 8);
    } else if (len <= 16777216) {
        *op++ = (62 << 2);
        *op++ = (uint8_t)((len - 1) & 0xFF);
        *op++ = (uint8_t)(((len - 1) >> 8) & 0xFF);
        *op++ = (uint8_t)(((len - 1) >> 16) & 0xFF);
    } else {
        *op++ = (63 << 2);
        *op++ = (uint8_t)((len - 1) & 0xFF);
        *op++ = (uint8_t)(((len - 1) >> 8) & 0xFF);
        *op++ = (uint8_t)(((len - 1) >> 16) & 0xFF);
        *op++ = (uint8_t)(((len - 1) >> 24) & 0xFF);
    }
    memcpy(op, literal, len);
    return op + len;
}

static uint8_t* snappy_emit_copy(uint8_t* op, size_t offset, size_t len) {
    while (len >= 68) {
        /* Emit copy of 64 bytes */
        *op++ = (uint8_t)((63 << 2) | SNAPPY_COPY_2);
        *op++ = (uint8_t)(offset & 0xFF);
        *op++ = (uint8_t)(offset >> 8);
        len -= 64;
    }

    if (len > 64) {
        /* Emit copy of 60 bytes */
        *op++ = (uint8_t)((59 << 2) | SNAPPY_COPY_2);
        *op++ = (uint8_t)(offset & 0xFF);
        *op++ = (uint8_t)(offset >> 8);
        len -= 60;
    }

    if (len >= 12 || offset >= 2048) {
        /* Use 2-byte offset copy */
        *op++ = (uint8_t)(((len - 1) << 2) | SNAPPY_COPY_2);
        *op++ = (uint8_t)(offset & 0xFF);
        *op++ = (uint8_t)(offset >> 8);
    } else {
        /* Use 1-byte offset copy (len 4-11, offset < 2048) */
        *op++ = (uint8_t)(((offset >> 8) << 5) | ((len - 4) << 2) | SNAPPY_COPY_1);
        *op++ = (uint8_t)(offset & 0xFF);
    }

    return op;
}

static uint8_t* snappy_compress_block(
    const uint8_t* src,
    size_t src_size,
    uint8_t* op) {

    if (src_size == 0) return op;
    if (src_size < 15) return snappy_emit_literal(op, src, src_size);

    uint16_t hash_table[SNAPPY_HASH_SIZE];
    memset(hash_table, 0, sizeof(hash_table));

    const uint8_t* const iend = src + src_size;
    const uint8_t* const ilimit = iend - 15;
    const uint8_t* ip = src + 1;
    const uint8_t* anchor = src;
    const uint8_t* candidate;

    /* Pre-seed hash table for position 0 */
    hash_table[snappy_hash(snappy_read32(src))] = 0;
    uint32_t next_hash = snappy_hash(snappy_read32(ip));

    for (;;) {
        /* Accelerating skip loop: find next match candidate */
        size_t skip = 32;
        const uint8_t* next_ip = ip;

        do {
            ip = next_ip;
            uint32_t h = next_hash;
            size_t step = skip >> 5;
            skip++;
            next_ip = ip + step;

            if (next_ip > ilimit) goto emit_remainder;

            next_hash = snappy_hash(snappy_read32(next_ip));
            SNAPPY_PREFETCH(&hash_table[next_hash]);
            candidate = src + hash_table[h];
            hash_table[h] = (uint16_t)(ip - src);
        } while (snappy_read32(ip) != snappy_read32(candidate) ||
                 (size_t)(ip - candidate) > SNAPPY_MAX_OFFSET);

        /* Emit pending literal bytes */
        if (ip > anchor) {
            op = snappy_emit_literal(op, anchor, (size_t)(ip - anchor));
        }

        /* Emit match(es), chaining consecutive matches */
        do {
            size_t match_len = 4 + fast_match_length(ip + 4, candidate + 4, iend);
            size_t offset = (size_t)(ip - candidate);
            ip += match_len;
            op = snappy_emit_copy(op, offset, match_len);
            anchor = ip;

            if (ip >= ilimit) goto emit_remainder;

            /* Insert hash entries near match end for future matches */
            hash_table[snappy_hash(snappy_read32(ip - 2))] = (uint16_t)(ip - 2 - src);
            hash_table[snappy_hash(snappy_read32(ip - 1))] = (uint16_t)(ip - 1 - src);

            /* Try to chain: check for immediate match at current position */
            uint32_t h = snappy_hash(snappy_read32(ip));
            candidate = src + hash_table[h];
            hash_table[h] = (uint16_t)(ip - src);
        } while (snappy_read32(ip) == snappy_read32(candidate) &&
                 (size_t)(ip - candidate) <= SNAPPY_MAX_OFFSET);

        /* Prepare next hash for the skip loop */
        next_hash = snappy_hash(snappy_read32(++ip));
    }

emit_remainder:
    if (anchor < iend) {
        op = snappy_emit_literal(op, anchor, (size_t)(iend - anchor));
    }
    return op;
}

carquet_status_t carquet_snappy_compress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Check if output buffer is large enough for worst case */
    size_t max_output = carquet_snappy_compress_bound(src_size);
    if (dst_capacity < max_output) {
        return CARQUET_ERROR_COMPRESSION;
    }

    uint8_t* op = dst;

    /* Write uncompressed length */
    op += snappy_write_varint(op, (uint32_t)src_size);

    if (src_size == 0) {
        *dst_size = (size_t)(op - dst);
        return CARQUET_OK;
    }

    if (!src) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    size_t pos = 0;
    while (pos < src_size) {
        size_t block_size = src_size - pos;
        if (block_size > SNAPPY_BLOCK_SIZE) {
            block_size = SNAPPY_BLOCK_SIZE;
        }
        op = snappy_compress_block(src + pos, block_size, op);
        pos += block_size;
    }

    *dst_size = (size_t)(op - dst);
    return CARQUET_OK;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================
 */

size_t carquet_snappy_compress_bound(size_t src_size) {
    return 32 + src_size + src_size / 6;
}

carquet_status_t carquet_snappy_get_uncompressed_length(
    const uint8_t* src,
    size_t src_size,
    size_t* length) {

    if (!src || !length) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    uint32_t len;
    size_t varint_len = snappy_read_varint(src, src + src_size, &len);
    if (varint_len == 0) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    *length = len;
    return CARQUET_OK;
}
