/**
 * @file page_reader.c
 * @brief Page reading implementation
 *
 * Handles reading and decoding of Parquet data pages.
 */

#include <carquet/carquet.h>
#include "reader_internal.h"
#include "thrift/parquet_types.h"
#include "encoding/plain.h"
#include "encoding/rle.h"
#include "core/endian.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(CARQUET_ARCH_ARM) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#endif

/* CRC32 verification */
extern uint32_t carquet_crc32(const uint8_t* data, size_t length);

/* SIMD dispatch functions for dictionary gather */
extern void carquet_dispatch_gather_i32(const int32_t* dict, const uint32_t* indices,
                                         int64_t count, int32_t* output);
extern void carquet_dispatch_gather_i64(const int64_t* dict, const uint32_t* indices,
                                         int64_t count, int64_t* output);
extern void carquet_dispatch_gather_float(const float* dict, const uint32_t* indices,
                                           int64_t count, float* output);
extern void carquet_dispatch_gather_double(const double* dict, const uint32_t* indices,
                                            int64_t count, double* output);
extern bool carquet_dispatch_checked_gather_i32(const int32_t* dict, int32_t dict_count,
                                                 const uint32_t* indices, int64_t count,
                                                 int32_t* output);
extern bool carquet_dispatch_checked_gather_i64(const int64_t* dict, int32_t dict_count,
                                                 const uint32_t* indices, int64_t count,
                                                 int64_t* output);
extern bool carquet_dispatch_checked_gather_float(const float* dict, int32_t dict_count,
                                                   const uint32_t* indices, int64_t count,
                                                   float* output);
extern bool carquet_dispatch_checked_gather_double(const double* dict, int32_t dict_count,
                                                    const uint32_t* indices, int64_t count,
                                                    double* output);

/* SIMD dispatch functions for definition level processing */
extern int64_t carquet_dispatch_count_non_nulls(const int16_t* def_levels, int64_t count,
                                                  int16_t max_def_level);
extern void carquet_dispatch_fill_def_levels(int16_t* def_levels, int64_t count, int16_t value);

/* Forward declarations for compression functions */
extern carquet_status_t carquet_lz4_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);
extern carquet_status_t carquet_snappy_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);
extern int carquet_gzip_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);
extern int carquet_zstd_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);
extern carquet_status_t carquet_byte_stream_split_decode_float(
    const uint8_t* data,
    size_t data_size,
    float* values,
    int64_t count);
extern carquet_status_t carquet_byte_stream_split_decode_double(
    const uint8_t* data,
    size_t data_size,
    double* values,
    int64_t count);

/* ============================================================================
 * Pre-buffered I/O Helper
 * ============================================================================
 */

/**
 * Read data from a file offset, using the prebuffer cache if available.
 * Returns bytes read (0 on failure).
 */
static size_t prebuf_read_at(carquet_reader_t* file_reader,
                             int64_t offset, void* buf, size_t size) {
    /* Check prebuffer cache first */
    if (file_reader->prebuffer.data &&
        offset >= file_reader->prebuffer.file_offset &&
        offset + (int64_t)size <=
            file_reader->prebuffer.file_offset + (int64_t)file_reader->prebuffer.size) {
        memcpy(buf, file_reader->prebuffer.data +
               (offset - file_reader->prebuffer.file_offset), size);
        return size;
    }

    /* Fall back to fseek + fread */
    if (fseek(file_reader->file, (long)offset, SEEK_SET) != 0) return 0;
    return fread(buf, 1, size, file_reader->file);
}

/* ============================================================================
 * Decompression
 * ============================================================================
 */

static carquet_status_t decompress_page(
    carquet_compression_t codec,
    const uint8_t* compressed,
    size_t compressed_size,
    uint8_t* decompressed,
    size_t decompressed_capacity,
    size_t* decompressed_size) {

    switch (codec) {
        case CARQUET_COMPRESSION_UNCOMPRESSED:
            if (compressed_size > decompressed_capacity) {
                return CARQUET_ERROR_DECOMPRESSION;
            }
            memcpy(decompressed, compressed, compressed_size);
            *decompressed_size = compressed_size;
            return CARQUET_OK;

        case CARQUET_COMPRESSION_SNAPPY:
            return carquet_snappy_decompress(
                compressed, compressed_size,
                decompressed, decompressed_capacity, decompressed_size);

        case CARQUET_COMPRESSION_LZ4:
        case CARQUET_COMPRESSION_LZ4_RAW:
            return carquet_lz4_decompress(
                compressed, compressed_size,
                decompressed, decompressed_capacity, decompressed_size);

        case CARQUET_COMPRESSION_GZIP:
            return carquet_gzip_decompress(
                compressed, compressed_size,
                decompressed, decompressed_capacity, decompressed_size);

        case CARQUET_COMPRESSION_ZSTD:
            return carquet_zstd_decompress(
                compressed, compressed_size,
                decompressed, decompressed_capacity, decompressed_size);

        default:
            return CARQUET_ERROR_UNSUPPORTED_CODEC;
    }
}

/* ============================================================================
 * Utility Functions
 * ============================================================================
 */

static inline int bit_width_for_max(int max_val) {
    if (max_val == 0) return 0;
    int width = 0;
    while (max_val > 0) {
        width++;
        max_val >>= 1;
    }
    return width;
}

#if defined(CARQUET_ARCH_ARM) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
static bool gather_fixed_dictionary_values_neon(const uint8_t* dict_data,
                                                 int32_t dict_count,
                                                 const uint32_t* indices,
                                                 int32_t count,
                                                 size_t value_size,
                                                 uint8_t* output) {
    for (int32_t i = 0; i < count; i++) {
        uint32_t idx = indices[i];
        if (idx >= (uint32_t)dict_count) {
            return false;
        }

        const uint8_t* src = dict_data + (size_t)idx * value_size;
        uint8_t* dst = output + (size_t)i * value_size;

        switch (value_size) {
            case 12:
                vst1_u8(dst, vld1_u8(src));
                memcpy(dst + 8, src + 8, 4);
                break;
            case 16:
                vst1q_u8(dst, vld1q_u8(src));
                break;
            case 32:
                vst1q_u8(dst, vld1q_u8(src));
                vst1q_u8(dst + 16, vld1q_u8(src + 16));
                break;
            default:
                if ((value_size & 15U) == 0 && value_size <= 64) {
                    for (size_t off = 0; off < value_size; off += 16) {
                        vst1q_u8(dst + off, vld1q_u8(src + off));
                    }
                } else {
                    memcpy(dst, src, value_size);
                }
                break;
        }
    }

    return true;
}
#endif

static bool page_values_can_be_viewed_directly(
    const carquet_column_reader_t* reader,
    carquet_encoding_t encoding) {

#if !CARQUET_LITTLE_ENDIAN
    (void)reader;
    (void)encoding;
    return false;
#else
    if (encoding != CARQUET_ENCODING_PLAIN ||
        reader->max_def_level > 0 ||
        reader->max_rep_level > 0) {
        return false;
    }

    switch (reader->type) {
        case CARQUET_PHYSICAL_INT32:
        case CARQUET_PHYSICAL_INT64:
        case CARQUET_PHYSICAL_INT96:
        case CARQUET_PHYSICAL_FLOAT:
        case CARQUET_PHYSICAL_DOUBLE:
            return true;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            return reader->type_length > 0;
        case CARQUET_PHYSICAL_BOOLEAN:
        case CARQUET_PHYSICAL_BYTE_ARRAY:
        default:
            return false;
    }
#endif
}

static void release_decoded_level_buffers(carquet_column_reader_t* reader) {
    free(reader->decoded_def_levels);
    free(reader->decoded_rep_levels);
    reader->decoded_def_levels = NULL;
    reader->decoded_rep_levels = NULL;
}

static carquet_status_t ensure_decoded_page_buffers(
    carquet_column_reader_t* reader,
    int32_t num_values,
    size_t value_size,
    carquet_error_t* error) {

    bool need_def_levels = reader->max_def_level > 0;
    bool need_rep_levels = reader->max_rep_level > 0;
    size_t values_buffer_size = value_size * (size_t)num_values;

    if (reader->decoded_ownership == CARQUET_DATA_VIEW) {
        reader->decoded_values = NULL;
        reader->decoded_capacity = 0;
    }
    reader->decoded_ownership = CARQUET_DATA_OWNED;

    if (!need_def_levels && reader->decoded_def_levels) {
        free(reader->decoded_def_levels);
        reader->decoded_def_levels = NULL;
    }
    if (!need_rep_levels && reader->decoded_rep_levels) {
        free(reader->decoded_rep_levels);
        reader->decoded_rep_levels = NULL;
    }

    if ((size_t)num_values > reader->decoded_capacity) {
        free(reader->decoded_values);
        reader->decoded_values = NULL;
        release_decoded_level_buffers(reader);

        if (values_buffer_size > 0) {
            reader->decoded_values = calloc(1, values_buffer_size);
        }
        if (need_def_levels && num_values > 0) {
            reader->decoded_def_levels = malloc(sizeof(int16_t) * (size_t)num_values);
        }
        if (need_rep_levels && num_values > 0) {
            reader->decoded_rep_levels = malloc(sizeof(int16_t) * (size_t)num_values);
        }
        reader->decoded_capacity = (size_t)num_values;
    } else {
        if (need_def_levels && !reader->decoded_def_levels && num_values > 0) {
            reader->decoded_def_levels = malloc(sizeof(int16_t) * (size_t)num_values);
        }
        if (need_rep_levels && !reader->decoded_rep_levels && num_values > 0) {
            reader->decoded_rep_levels = malloc(sizeof(int16_t) * (size_t)num_values);
        }
    }

    if ((values_buffer_size > 0 && !reader->decoded_values) ||
        (need_def_levels && num_values > 0 && !reader->decoded_def_levels) ||
        (need_rep_levels && num_values > 0 && !reader->decoded_rep_levels)) {
        free(reader->decoded_values);
        reader->decoded_values = NULL;
        release_decoded_level_buffers(reader);
        reader->decoded_capacity = 0;
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate decode buffers");
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    return CARQUET_OK;
}

/* ============================================================================
 * Level Decoding
 * ============================================================================
 */

static carquet_status_t decode_levels_rle(
    const uint8_t* data,
    size_t data_size,
    int bit_width,
    int32_t num_values,
    int16_t* levels,
    size_t* bytes_consumed) {

    if (bit_width == 0) {
        /* All zeros */
        memset(levels, 0, num_values * sizeof(int16_t));
        *bytes_consumed = 0;
        return CARQUET_OK;
    }

    /* Use the convenience function for decoding levels */
    int64_t decoded = carquet_rle_decode_levels(
        data, data_size, bit_width, levels, num_values);

    if (decoded < 0) {
        return CARQUET_ERROR_DECODE;
    }

    if (decoded != num_values) {
        return CARQUET_ERROR_DECODE;
    }

    /* Estimate bytes consumed (not perfect, but good enough) */
    *bytes_consumed = data_size;
    return CARQUET_OK;
}

/* ============================================================================
 * Dictionary Page Reading
 * ============================================================================
 */

carquet_status_t carquet_read_dictionary_page(
    carquet_column_reader_t* reader,
    uint8_t* page_data,
    size_t page_size,
    const parquet_dictionary_page_header_t* header,
    carquet_data_ownership_t ownership,
    carquet_error_t* error) {

    if (!reader || !page_data || !header) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_ARGUMENT, "NULL argument");
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Allocate dictionary storage */
    size_t value_size = 0;
    switch (reader->type) {
        case CARQUET_PHYSICAL_INT32:
        case CARQUET_PHYSICAL_FLOAT:
            value_size = 4;
            break;
        case CARQUET_PHYSICAL_INT64:
        case CARQUET_PHYSICAL_DOUBLE:
            value_size = 8;
            break;
        case CARQUET_PHYSICAL_INT96:
            value_size = 12;
            break;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            value_size = reader->type_length;
            break;
        case CARQUET_PHYSICAL_BYTE_ARRAY:
            /* Variable length - will be handled differently */
            break;
        default:
            break;
    }

    reader->dictionary_count = header->num_values;
    reader->dictionary_ownership = ownership;

    if (reader->type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        /* For variable length, keep raw dictionary bytes and build offsets once */
        reader->dictionary_data = page_data;
        reader->dictionary_size = page_size;

        /* Build offset table for O(1) BYTE_ARRAY lookup */
        reader->dictionary_offsets = malloc((size_t)header->num_values * sizeof(uint32_t));
        if (!reader->dictionary_offsets) {
            if (ownership == CARQUET_DATA_OWNED) {
                free(reader->dictionary_data);
            }
            reader->dictionary_data = NULL;
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate offset table");
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }

        /* Scan dictionary once to build offset table */
        const uint8_t* dict_ptr = page_data;
        size_t dict_remaining = page_size;
        for (int32_t i = 0; i < header->num_values; i++) {
            if (dict_remaining < 4) {
                if (ownership == CARQUET_DATA_OWNED) {
                    free(reader->dictionary_data);
                }
                free(reader->dictionary_offsets);
                reader->dictionary_data = NULL;
                reader->dictionary_offsets = NULL;
                CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Truncated dictionary");
                return CARQUET_ERROR_DECODE;
            }
            reader->dictionary_offsets[i] = (uint32_t)(dict_ptr - page_data);
            uint32_t len = carquet_read_u32_le(dict_ptr);
            size_t entry_size = 4 + len;
            if (dict_remaining < entry_size) {
                if (ownership == CARQUET_DATA_OWNED) {
                    free(reader->dictionary_data);
                }
                free(reader->dictionary_offsets);
                reader->dictionary_data = NULL;
                reader->dictionary_offsets = NULL;
                CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Invalid dictionary entry");
                return CARQUET_ERROR_DECODE;
            }
            dict_ptr += entry_size;
            dict_remaining -= entry_size;
        }
    } else {
        /* Fixed size values */
        size_t dict_size = value_size * header->num_values;
        if (dict_size > page_size) {
            if (ownership == CARQUET_DATA_OWNED) {
                free(page_data);
            }
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Truncated dictionary");
            return CARQUET_ERROR_DECODE;
        }
        reader->dictionary_data = page_data;
        reader->dictionary_size = dict_size;
    }

    reader->has_dictionary = true;
    return CARQUET_OK;
}

/* ============================================================================
 * Data Page Reading
 * ============================================================================
 */

carquet_status_t carquet_read_data_page_v1(
    carquet_column_reader_t* reader,
    const uint8_t* page_data,
    size_t page_size,
    const parquet_data_page_header_t* header,
    void* values,
    int64_t max_values,
    int16_t* def_levels,
    int16_t* rep_levels,
    int64_t* values_read,
    carquet_error_t* error) {

    const uint8_t* ptr = page_data;
    size_t remaining = page_size;
    size_t bytes_consumed;

    int32_t num_values = header->num_values;
    if (num_values > max_values) {
        num_values = (int32_t)max_values;
    }

    /* Decode repetition levels if needed */
    if (reader->max_rep_level > 0 && rep_levels) {
        /* Read 4-byte length prefix */
        if (remaining < 4) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Truncated rep levels");
            return CARQUET_ERROR_DECODE;
        }
        uint32_t rep_size = carquet_read_u32_le(ptr);
        ptr += 4;
        remaining -= 4;

        if (rep_size > remaining) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Invalid rep level size");
            return CARQUET_ERROR_DECODE;
        }

        int bit_width = bit_width_for_max(reader->max_rep_level);
        carquet_status_t status = decode_levels_rle(
            ptr, rep_size, bit_width, num_values, rep_levels, &bytes_consumed);
        if (status != CARQUET_OK) {
            CARQUET_SET_ERROR(error, status, "Failed to decode rep levels");
            return status;
        }
        ptr += rep_size;
        remaining -= rep_size;
    } else if (rep_levels) {
        memset(rep_levels, 0, num_values * sizeof(int16_t));
    }

    /* Decode definition levels if needed */
    if (reader->max_def_level > 0 && def_levels) {
        /* Read 4-byte length prefix */
        if (remaining < 4) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Truncated def levels");
            return CARQUET_ERROR_DECODE;
        }
        uint32_t def_size = carquet_read_u32_le(ptr);
        ptr += 4;
        remaining -= 4;

        if (def_size > remaining) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Invalid def level size");
            return CARQUET_ERROR_DECODE;
        }

        int bit_width = bit_width_for_max(reader->max_def_level);
        carquet_status_t status = decode_levels_rle(
            ptr, def_size, bit_width, num_values, def_levels, &bytes_consumed);
        if (status != CARQUET_OK) {
            CARQUET_SET_ERROR(error, status, "Failed to decode def levels");
            return status;
        }
        ptr += def_size;
        remaining -= def_size;
    } else if (def_levels) {
        /* Set all to max level (all values present) - use SIMD dispatch */
        carquet_dispatch_fill_def_levels(def_levels, num_values, reader->max_def_level);
    }

    /* Count non-null values */
    int32_t non_null_count = num_values;
    if (def_levels && reader->max_def_level > 0) {
        non_null_count = (int32_t)carquet_dispatch_count_non_nulls(
            def_levels, num_values, reader->max_def_level);
    }

    /* Decode values based on encoding */
    carquet_status_t status = CARQUET_OK;

    switch (header->encoding) {
        case CARQUET_ENCODING_PLAIN:
            {
                /* For BYTE_ARRAY, carquet writes all num_values entries
                 * (including length-0 entries for nulls) so that values[i]
                 * aligns with def_levels[i].  For fixed-size types, only
                 * non_null_count values are stored on disk. */
                int32_t decode_count =
                    (reader->type == CARQUET_PHYSICAL_BYTE_ARRAY)
                        ? num_values : non_null_count;
                int64_t bytes = carquet_decode_plain(
                    ptr, remaining, reader->type, reader->type_length,
                    values, decode_count);
                if (bytes < 0) {
                    status = CARQUET_ERROR_DECODE;
                }
            }
            break;

        case CARQUET_ENCODING_BYTE_STREAM_SPLIT:
            switch (reader->type) {
                case CARQUET_PHYSICAL_FLOAT:
                    status = carquet_byte_stream_split_decode_float(
                        ptr, remaining, (float*)values, non_null_count);
                    break;
                case CARQUET_PHYSICAL_DOUBLE:
                    status = carquet_byte_stream_split_decode_double(
                        ptr, remaining, (double*)values, non_null_count);
                    break;
                default:
                    status = CARQUET_ERROR_INVALID_ENCODING;
                    break;
            }
            break;

        case CARQUET_ENCODING_RLE_DICTIONARY:
        case CARQUET_ENCODING_PLAIN_DICTIONARY:
            if (!reader->has_dictionary) {
                CARQUET_SET_ERROR(error, CARQUET_ERROR_DICTIONARY_NOT_FOUND,
                    "Dictionary encoding without dictionary");
                return CARQUET_ERROR_DICTIONARY_NOT_FOUND;
            }
            /* Decode dictionary indices using RLE */
            {
                /* Read bit width byte */
                if (remaining < 1) {
                    CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Missing bit width");
                    return CARQUET_ERROR_DECODE;
                }
                int bit_width = ptr[0];
                ptr++;
                remaining--;

                /* Decode num_values indices (not non_null_count) because
                 * carquet's writer emits an index for every logical row
                 * (including null positions).  This keeps values[i] aligned
                 * with def_levels[i] so the caller can index both arrays
                 * with the same offset. */

                /* Dictionary preservation: decode indices directly into output */
                if (reader->preserve_dictionary) {
                    uint32_t* out_indices = (uint32_t*)values;
                    int64_t decoded = carquet_rle_decode_all(
                        ptr, remaining, bit_width, out_indices, num_values);
                    if (decoded < 0) {
                        CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Failed to decode dictionary indices");
                        return CARQUET_ERROR_DECODE;
                    }
                    break;
                }

                /* Use reusable indices buffer to avoid per-page allocation */
                uint32_t* indices;
                if ((size_t)num_values <= reader->indices_capacity) {
                    indices = reader->indices_buffer;
                } else {
                    /* Need larger buffer - reallocate */
                    free(reader->indices_buffer);
                    reader->indices_buffer = malloc((size_t)num_values * sizeof(uint32_t));
                    if (!reader->indices_buffer) {
                        reader->indices_capacity = 0;
                        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate indices");
                        return CARQUET_ERROR_OUT_OF_MEMORY;
                    }
                    reader->indices_capacity = num_values;
                    indices = reader->indices_buffer;
                }

                int64_t decoded = carquet_rle_decode_all(
                    ptr, remaining, bit_width, indices, num_values);

                if (decoded < 0) {
                    CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Failed to decode dictionary indices");
                    return CARQUET_ERROR_DECODE;
                }

                /* Look up values from dictionary for all num_values positions */
                if (reader->type == CARQUET_PHYSICAL_BYTE_ARRAY) {
                    /* BYTE_ARRAY: dictionary is stored as length-prefixed values */
                    carquet_byte_array_t* out = (carquet_byte_array_t*)values;

                    /* Use O(1) offset table lookup (built when dictionary was read) */
                    if (reader->dictionary_offsets) {
                        for (int32_t i = 0; i < num_values; i++) {
                            int32_t idx = (int32_t)indices[i];
                            if (idx < 0 || idx >= reader->dictionary_count) {
                                status = CARQUET_ERROR_DECODE;
                                break;
                            }

                            /* Direct O(1) lookup using offset table */
                            uint32_t offset = reader->dictionary_offsets[idx];
                            const uint8_t* dict_ptr = reader->dictionary_data + offset;
                            uint32_t len = carquet_read_u32_le(dict_ptr);
                            out[i].data = (uint8_t*)(dict_ptr + 4);
                            out[i].length = (int32_t)len;
                        }
                    } else {
                        /* Fallback: scan each time (shouldn't happen for new readers) */
                        for (int32_t i = 0; i < num_values; i++) {
                            int32_t idx = (int32_t)indices[i];
                            if (idx < 0 || idx >= reader->dictionary_count) {
                                status = CARQUET_ERROR_DECODE;
                                break;
                            }

                            const uint8_t* dict_ptr = reader->dictionary_data;
                            for (int32_t j = 0; j < idx; j++) {
                                uint32_t len = carquet_read_u32_le(dict_ptr);
                                dict_ptr += 4 + len;
                            }
                            uint32_t len = carquet_read_u32_le(dict_ptr);
                            out[i].data = (uint8_t*)(dict_ptr + 4);
                            out[i].length = (int32_t)len;
                        }
                    }
                } else {
                    /* Use SIMD-optimized gather for common types */
                    switch (reader->type) {
                        case CARQUET_PHYSICAL_INT32:
                            if (!carquet_dispatch_checked_gather_i32(
                                    (const int32_t*)reader->dictionary_data,
                                    reader->dictionary_count,
                                    indices, num_values, (int32_t*)values)) {
                                CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE,
                                                  "Dictionary index out of bounds");
                                return CARQUET_ERROR_DECODE;
                            }
                            break;
                        case CARQUET_PHYSICAL_INT64:
                            if (!carquet_dispatch_checked_gather_i64(
                                    (const int64_t*)reader->dictionary_data,
                                    reader->dictionary_count,
                                    indices, num_values, (int64_t*)values)) {
                                CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE,
                                                  "Dictionary index out of bounds");
                                return CARQUET_ERROR_DECODE;
                            }
                            break;
                        case CARQUET_PHYSICAL_FLOAT:
                            if (!carquet_dispatch_checked_gather_float(
                                    (const float*)reader->dictionary_data,
                                    reader->dictionary_count,
                                    indices, num_values, (float*)values)) {
                                CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE,
                                                  "Dictionary index out of bounds");
                                return CARQUET_ERROR_DECODE;
                            }
                            break;
                        case CARQUET_PHYSICAL_DOUBLE:
                            if (!carquet_dispatch_checked_gather_double(
                                    (const double*)reader->dictionary_data,
                                    reader->dictionary_count,
                                    indices, num_values, (double*)values)) {
                                CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE,
                                                  "Dictionary index out of bounds");
                                return CARQUET_ERROR_DECODE;
                            }
                            break;
                        case CARQUET_PHYSICAL_INT96:
                        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
                            {
                                size_t value_size = (reader->type == CARQUET_PHYSICAL_INT96)
                                    ? 12 : (size_t)reader->type_length;
                                uint8_t* out = (uint8_t*)values;
                                bool ok;
#if defined(CARQUET_ARCH_ARM) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                                ok = gather_fixed_dictionary_values_neon(
                                    reader->dictionary_data,
                                    reader->dictionary_count,
                                    indices,
                                    num_values,
                                    value_size,
                                    out);
#else
                                ok = true;
                                for (int32_t i = 0; i < num_values; i++) {
                                    uint32_t idx = indices[i];
                                    if (idx >= (uint32_t)reader->dictionary_count) {
                                        ok = false;
                                        break;
                                    }
                                    memcpy(out + (size_t)i * value_size,
                                           reader->dictionary_data + (size_t)idx * value_size,
                                           value_size);
                                }
#endif
                                if (!ok) {
                                    CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE,
                                                      "Dictionary index out of bounds");
                                    return CARQUET_ERROR_DECODE;
                                }
                            }
                            break;
                        default:
                            break;
                    }
                }
                /* indices buffer is reused, don't free */
            }
            break;

        default:
            CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_ENCODING,
                "Unsupported encoding: %d", header->encoding);
            return CARQUET_ERROR_INVALID_ENCODING;
    }

    if (status != CARQUET_OK) {
        CARQUET_SET_ERROR(error, status, "Failed to decode values");
        return status;
    }

    *values_read = num_values;
    return CARQUET_OK;
}

/* ============================================================================
 * Data Page V2 Reading
 * ============================================================================
 *
 * V2 page layout: [rep_levels_bytes | def_levels_bytes | data_bytes]
 * - Rep/def levels are NOT compressed (stored before compressed data)
 * - No 4-byte length prefixes for levels (byte lengths come from header)
 * - Levels are RLE-encoded (same as V1, but without length prefix)
 * - Data portion may or may not be compressed (header.is_compressed)
 */

carquet_status_t carquet_read_data_page_v2(
    carquet_column_reader_t* reader,
    const uint8_t* page_data,
    size_t page_size,
    const parquet_data_page_header_v2_t* header,
    void* values,
    int64_t max_values,
    int16_t* def_levels,
    int16_t* rep_levels,
    int64_t* values_read,
    carquet_error_t* error) {

    const uint8_t* ptr = page_data;
    size_t remaining = page_size;
    size_t bytes_consumed;

    int32_t num_values = header->num_values;
    if (num_values > max_values) {
        num_values = (int32_t)max_values;
    }

    /* V2: Repetition levels come first, with known byte length (no length prefix) */
    if (reader->max_rep_level > 0 && rep_levels) {
        int32_t rep_bytes = header->repetition_levels_byte_length;
        if (rep_bytes < 0 || (size_t)rep_bytes > remaining) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Invalid V2 rep level size");
            return CARQUET_ERROR_DECODE;
        }

        if (rep_bytes > 0) {
            int bit_width = bit_width_for_max(reader->max_rep_level);
            carquet_status_t status = decode_levels_rle(
                ptr, (size_t)rep_bytes, bit_width, num_values, rep_levels, &bytes_consumed);
            if (status != CARQUET_OK) {
                CARQUET_SET_ERROR(error, status, "Failed to decode V2 rep levels");
                return status;
            }
        } else {
            memset(rep_levels, 0, num_values * sizeof(int16_t));
        }
        ptr += rep_bytes;
        remaining -= (size_t)rep_bytes;
    } else {
        /* Skip rep level bytes even if we don't need them */
        int32_t rep_bytes = header->repetition_levels_byte_length;
        if (rep_bytes > 0 && (size_t)rep_bytes <= remaining) {
            ptr += rep_bytes;
            remaining -= (size_t)rep_bytes;
        }
        if (rep_levels) {
            memset(rep_levels, 0, num_values * sizeof(int16_t));
        }
    }

    /* V2: Definition levels come second, with known byte length (no length prefix) */
    if (reader->max_def_level > 0 && def_levels) {
        int32_t def_bytes = header->definition_levels_byte_length;
        if (def_bytes < 0 || (size_t)def_bytes > remaining) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Invalid V2 def level size");
            return CARQUET_ERROR_DECODE;
        }

        if (def_bytes > 0) {
            int bit_width = bit_width_for_max(reader->max_def_level);
            carquet_status_t status = decode_levels_rle(
                ptr, (size_t)def_bytes, bit_width, num_values, def_levels, &bytes_consumed);
            if (status != CARQUET_OK) {
                CARQUET_SET_ERROR(error, status, "Failed to decode V2 def levels");
                return status;
            }
        } else {
            memset(def_levels, 0, num_values * sizeof(int16_t));
        }
        ptr += def_bytes;
        remaining -= (size_t)def_bytes;
    } else {
        /* Skip def level bytes even if we don't need them */
        int32_t def_bytes = header->definition_levels_byte_length;
        if (def_bytes > 0 && (size_t)def_bytes <= remaining) {
            ptr += def_bytes;
            remaining -= (size_t)def_bytes;
        }
        if (def_levels) {
            /* Set all to max level (all values present) - use SIMD dispatch */
            carquet_dispatch_fill_def_levels(def_levels, num_values, reader->max_def_level);
        }
    }

    /* V2: Remaining bytes are the data payload.
     * Note: For V2, decompression of the data portion is handled by the caller
     * (load_next_page_mmap/fread) BEFORE calling this function, since the caller
     * must decompress only the data portion while leaving levels uncompressed.
     * By the time we get here, ptr points to uncompressed data. */

    /* Count non-null values */
    int32_t non_null_count = num_values;
    if (def_levels && reader->max_def_level > 0) {
        non_null_count = (int32_t)carquet_dispatch_count_non_nulls(
            def_levels, num_values, reader->max_def_level);
    }

    /* Decode values based on encoding - reuse V1 value decoding logic */
    carquet_status_t status = CARQUET_OK;

    switch (header->encoding) {
        case CARQUET_ENCODING_PLAIN:
            {
                int32_t decode_count =
                    (reader->type == CARQUET_PHYSICAL_BYTE_ARRAY)
                        ? num_values : non_null_count;
                int64_t bytes = carquet_decode_plain(
                    ptr, remaining, reader->type, reader->type_length,
                    values, decode_count);
                if (bytes < 0) {
                    status = CARQUET_ERROR_DECODE;
                }
            }
            break;

        case CARQUET_ENCODING_BYTE_STREAM_SPLIT:
            switch (reader->type) {
                case CARQUET_PHYSICAL_FLOAT:
                    status = carquet_byte_stream_split_decode_float(
                        ptr, remaining, (float*)values, non_null_count);
                    break;
                case CARQUET_PHYSICAL_DOUBLE:
                    status = carquet_byte_stream_split_decode_double(
                        ptr, remaining, (double*)values, non_null_count);
                    break;
                default:
                    status = CARQUET_ERROR_INVALID_ENCODING;
                    break;
            }
            break;

        case CARQUET_ENCODING_RLE_DICTIONARY:
        case CARQUET_ENCODING_PLAIN_DICTIONARY:
            if (!reader->has_dictionary) {
                CARQUET_SET_ERROR(error, CARQUET_ERROR_DICTIONARY_NOT_FOUND,
                    "Dictionary encoding without dictionary");
                return CARQUET_ERROR_DICTIONARY_NOT_FOUND;
            }
            /* Decode dictionary indices using RLE */
            {
                /* Read bit width byte */
                if (remaining < 1) {
                    CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Missing bit width");
                    return CARQUET_ERROR_DECODE;
                }
                int bit_width = ptr[0];
                ptr++;
                remaining--;

                /* Dictionary preservation: decode indices directly into output */
                if (reader->preserve_dictionary) {
                    uint32_t* out_indices = (uint32_t*)values;
                    int64_t decoded = carquet_rle_decode_all(
                        ptr, remaining, bit_width, out_indices, num_values);
                    if (decoded < 0) {
                        CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Failed to decode dictionary indices");
                        return CARQUET_ERROR_DECODE;
                    }
                    break;
                }

                /* Use reusable indices buffer */
                uint32_t* indices;
                if ((size_t)num_values <= reader->indices_capacity) {
                    indices = reader->indices_buffer;
                } else {
                    free(reader->indices_buffer);
                    reader->indices_buffer = malloc((size_t)num_values * sizeof(uint32_t));
                    if (!reader->indices_buffer) {
                        reader->indices_capacity = 0;
                        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate indices");
                        return CARQUET_ERROR_OUT_OF_MEMORY;
                    }
                    reader->indices_capacity = num_values;
                    indices = reader->indices_buffer;
                }

                int64_t decoded = carquet_rle_decode_all(
                    ptr, remaining, bit_width, indices, num_values);

                if (decoded < 0) {
                    CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Failed to decode dictionary indices");
                    return CARQUET_ERROR_DECODE;
                }

                /* Look up values from dictionary — same logic as V1 */
                if (reader->type == CARQUET_PHYSICAL_BYTE_ARRAY) {
                    carquet_byte_array_t* out = (carquet_byte_array_t*)values;

                    if (reader->dictionary_offsets) {
                        for (int32_t i = 0; i < num_values; i++) {
                            int32_t idx = (int32_t)indices[i];
                            if (idx < 0 || idx >= reader->dictionary_count) {
                                status = CARQUET_ERROR_DECODE;
                                break;
                            }
                            uint32_t offset = reader->dictionary_offsets[idx];
                            const uint8_t* dict_ptr = reader->dictionary_data + offset;
                            uint32_t len = carquet_read_u32_le(dict_ptr);
                            out[i].data = (uint8_t*)(dict_ptr + 4);
                            out[i].length = (int32_t)len;
                        }
                    } else {
                        for (int32_t i = 0; i < num_values; i++) {
                            int32_t idx = (int32_t)indices[i];
                            if (idx < 0 || idx >= reader->dictionary_count) {
                                status = CARQUET_ERROR_DECODE;
                                break;
                            }
                            const uint8_t* dict_ptr = reader->dictionary_data;
                            for (int32_t j = 0; j < idx; j++) {
                                uint32_t len = carquet_read_u32_le(dict_ptr);
                                dict_ptr += 4 + len;
                            }
                            uint32_t len = carquet_read_u32_le(dict_ptr);
                            out[i].data = (uint8_t*)(dict_ptr + 4);
                            out[i].length = (int32_t)len;
                        }
                    }
                } else {
                    switch (reader->type) {
                        case CARQUET_PHYSICAL_INT32:
                            if (!carquet_dispatch_checked_gather_i32(
                                    (const int32_t*)reader->dictionary_data,
                                    reader->dictionary_count,
                                    indices, num_values, (int32_t*)values)) {
                                CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE,
                                                  "Dictionary index out of bounds");
                                return CARQUET_ERROR_DECODE;
                            }
                            break;
                        case CARQUET_PHYSICAL_INT64:
                            if (!carquet_dispatch_checked_gather_i64(
                                    (const int64_t*)reader->dictionary_data,
                                    reader->dictionary_count,
                                    indices, num_values, (int64_t*)values)) {
                                CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE,
                                                  "Dictionary index out of bounds");
                                return CARQUET_ERROR_DECODE;
                            }
                            break;
                        case CARQUET_PHYSICAL_FLOAT:
                            if (!carquet_dispatch_checked_gather_float(
                                    (const float*)reader->dictionary_data,
                                    reader->dictionary_count,
                                    indices, num_values, (float*)values)) {
                                CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE,
                                                  "Dictionary index out of bounds");
                                return CARQUET_ERROR_DECODE;
                            }
                            break;
                        case CARQUET_PHYSICAL_DOUBLE:
                            if (!carquet_dispatch_checked_gather_double(
                                    (const double*)reader->dictionary_data,
                                    reader->dictionary_count,
                                    indices, num_values, (double*)values)) {
                                CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE,
                                                  "Dictionary index out of bounds");
                                return CARQUET_ERROR_DECODE;
                            }
                            break;
                        case CARQUET_PHYSICAL_INT96:
                        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
                            {
                                size_t value_size = (reader->type == CARQUET_PHYSICAL_INT96)
                                    ? 12 : (size_t)reader->type_length;
                                uint8_t* out = (uint8_t*)values;
                                bool ok;
#if defined(CARQUET_ARCH_ARM) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                                ok = gather_fixed_dictionary_values_neon(
                                    reader->dictionary_data,
                                    reader->dictionary_count,
                                    indices, num_values,
                                    value_size, out);
#else
                                ok = true;
                                for (int32_t i = 0; i < num_values; i++) {
                                    uint32_t idx = indices[i];
                                    if (idx >= (uint32_t)reader->dictionary_count) {
                                        ok = false;
                                        break;
                                    }
                                    memcpy(out + (size_t)i * value_size,
                                           reader->dictionary_data + (size_t)idx * value_size,
                                           value_size);
                                }
#endif
                                if (!ok) {
                                    CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE,
                                                      "Dictionary index out of bounds");
                                    return CARQUET_ERROR_DECODE;
                                }
                            }
                            break;
                        default:
                            break;
                    }
                }
            }
            break;

        default:
            CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_ENCODING,
                "Unsupported encoding: %d", header->encoding);
            return CARQUET_ERROR_INVALID_ENCODING;
    }

    if (status != CARQUET_OK) {
        CARQUET_SET_ERROR(error, status, "Failed to decode values");
        return status;
    }

    *values_read = num_values;
    return CARQUET_OK;
}

/* ============================================================================
 * Helper: Get value size for a physical type
 * ============================================================================
 */

static size_t get_value_size(carquet_physical_type_t type, int32_t type_length) {
    switch (type) {
        case CARQUET_PHYSICAL_BOOLEAN:
            return 1;
        case CARQUET_PHYSICAL_INT32:
        case CARQUET_PHYSICAL_FLOAT:
            return 4;
        case CARQUET_PHYSICAL_INT64:
        case CARQUET_PHYSICAL_DOUBLE:
            return 8;
        case CARQUET_PHYSICAL_INT96:
            return 12;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            return type_length;
        case CARQUET_PHYSICAL_BYTE_ARRAY:
            return sizeof(carquet_byte_array_t);
        default:
            return 0;
    }
}

/* ============================================================================
 * Helper: Load dictionary page (mmap path)
 * ============================================================================
 */

static carquet_status_t load_dictionary_page_mmap(
    carquet_column_reader_t* reader,
    carquet_error_t* error) {

    carquet_reader_t* file_reader = reader->file_reader;
    const uint8_t* mmap_data = file_reader->mmap_data;
    const parquet_column_metadata_t* col_meta = reader->col_meta;

    /* Parse page header directly from mmap */
    int64_t dict_offset = col_meta->dictionary_page_offset;
    if (dict_offset < 0 || (size_t)dict_offset >= file_reader->file_size) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_PAGE, "Dictionary page offset out of range");
        return CARQUET_ERROR_INVALID_PAGE;
    }
    const uint8_t* header_ptr = mmap_data + dict_offset;
    size_t remaining = file_reader->file_size - (size_t)dict_offset;
    size_t max_header = remaining < 256 ? remaining : 256;

    parquet_page_header_t page_header;
    size_t header_size;
    carquet_status_t status = parquet_parse_page_header(
        header_ptr, max_header, &page_header, &header_size, error);
    if (status != CARQUET_OK) {
        return status;
    }

    if (page_header.type != CARQUET_PAGE_DICTIONARY) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_PAGE, "Expected dictionary page");
        return CARQUET_ERROR_INVALID_PAGE;
    }

    /* Get pointer to compressed data */
    const uint8_t* compressed = header_ptr + header_size;

    /* Verify CRC32 if present */
    if (page_header.has_crc && file_reader->options.verify_checksums) {
        uint32_t computed_crc = carquet_crc32(compressed, page_header.compressed_page_size);
        uint32_t expected_crc = (uint32_t)page_header.crc;
        if (computed_crc != expected_crc) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_CRC_MISMATCH,
                "Dictionary page CRC mismatch: expected 0x%08X, got 0x%08X",
                expected_crc, computed_crc);
            return CARQUET_ERROR_CRC_MISMATCH;
        }
    }

    /* Process dictionary data */
    const uint8_t* page_data;
    size_t page_size;
    uint8_t* decompressed = NULL;

    if (col_meta->codec == CARQUET_COMPRESSION_UNCOMPRESSED) {
        /* Zero-copy: point directly to mmap data */
        page_data = compressed;
        page_size = page_header.compressed_page_size;
    } else {
        /* Must decompress */
        decompressed = malloc(page_header.uncompressed_page_size);
        if (!decompressed) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate decompress buffer");
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }

        status = decompress_page(col_meta->codec,
            compressed, page_header.compressed_page_size,
            decompressed, page_header.uncompressed_page_size, &page_size);

        if (status != CARQUET_OK) {
            free(decompressed);
            CARQUET_SET_ERROR(error, status, "Failed to decompress dictionary");
            return status;
        }
        page_data = decompressed;
    }

    /* Parse dictionary */
    status = carquet_read_dictionary_page(
        reader, (uint8_t*)page_data, page_size,
        &page_header.dictionary_page_header,
        col_meta->codec == CARQUET_COMPRESSION_UNCOMPRESSED
            ? CARQUET_DATA_VIEW
            : CARQUET_DATA_OWNED,
        error);

    /* Compute actual first data page offset from dictionary page layout.
     * Some writers (e.g. DuckDB) set data_page_offset incorrectly for
     * dictionary-encoded columns. The reliable offset is always right
     * after the dictionary page: dict_offset + header + compressed data. */
    if (status == CARQUET_OK) {
        reader->data_start_offset = dict_offset + (int64_t)header_size +
                                    page_header.compressed_page_size;
    }

    if (col_meta->codec != CARQUET_COMPRESSION_UNCOMPRESSED && status != CARQUET_OK) {
        free(decompressed);
    }
    return status;
}

/* ============================================================================
 * Helper: Load dictionary page (fread path)
 * ============================================================================
 */

static carquet_status_t load_dictionary_page_fread(
    carquet_column_reader_t* reader,
    carquet_error_t* error) {

    carquet_reader_t* file_reader = reader->file_reader;
    const parquet_column_metadata_t* col_meta = reader->col_meta;
    int64_t dict_offset = col_meta->dictionary_page_offset;

    /* Read page header (from prebuffer cache or file) */
    uint8_t header_buf[256];
    size_t header_read = prebuf_read_at(file_reader, dict_offset,
                                         header_buf, sizeof(header_buf));
    if (header_read < 8) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_READ, "Failed to read dictionary header");
        return CARQUET_ERROR_FILE_READ;
    }

    parquet_page_header_t page_header;
    size_t header_size;
    carquet_status_t status = parquet_parse_page_header(
        header_buf, header_read, &page_header, &header_size, error);
    if (status != CARQUET_OK) {
        return status;
    }

    if (page_header.type != CARQUET_PAGE_DICTIONARY) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_PAGE, "Expected dictionary page");
        return CARQUET_ERROR_INVALID_PAGE;
    }

    /* Read compressed data (from prebuffer cache or file) */
    uint8_t* compressed = malloc(page_header.compressed_page_size);
    if (!compressed) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate compressed buffer");
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    int64_t dict_data_offset = dict_offset + (int64_t)header_size;
    if (prebuf_read_at(file_reader, dict_data_offset,
                       compressed, page_header.compressed_page_size) !=
        (size_t)page_header.compressed_page_size) {
        free(compressed);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_READ, "Failed to read dictionary data");
        return CARQUET_ERROR_FILE_READ;
    }

    /* Verify CRC32 if present */
    if (page_header.has_crc && file_reader->options.verify_checksums) {
        uint32_t computed_crc = carquet_crc32(compressed, page_header.compressed_page_size);
        uint32_t expected_crc = (uint32_t)page_header.crc;
        if (computed_crc != expected_crc) {
            free(compressed);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_CRC_MISMATCH,
                "Dictionary page CRC mismatch: expected 0x%08X, got 0x%08X",
                expected_crc, computed_crc);
            return CARQUET_ERROR_CRC_MISMATCH;
        }
    }

    /* Decompress if needed */
    uint8_t* page_data;
    size_t page_size;

    if (col_meta->codec == CARQUET_COMPRESSION_UNCOMPRESSED) {
        page_data = compressed;
        page_size = page_header.compressed_page_size;
    } else {
        page_data = malloc(page_header.uncompressed_page_size);
        if (!page_data) {
            free(compressed);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate decompress buffer");
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }

        status = decompress_page(col_meta->codec,
            compressed, page_header.compressed_page_size,
            page_data, page_header.uncompressed_page_size, &page_size);
        free(compressed);

        if (status != CARQUET_OK) {
            free(page_data);
            CARQUET_SET_ERROR(error, status, "Failed to decompress dictionary");
            return status;
        }
    }

    /* Parse dictionary */
    status = carquet_read_dictionary_page(
        reader, page_data, page_size,
        &page_header.dictionary_page_header,
        CARQUET_DATA_OWNED, error);

    /* Compute actual first data page offset from dictionary page layout.
     * Some writers (e.g. DuckDB) set data_page_offset incorrectly for
     * dictionary-encoded columns. The reliable offset is always right
     * after the dictionary page: dict_offset + header + compressed data. */
    if (status == CARQUET_OK) {
        reader->data_start_offset = col_meta->dictionary_page_offset +
                                    (int64_t)header_size +
                                    page_header.compressed_page_size;
    }

    if (status != CARQUET_OK) {
        if (page_data != compressed) {
            free(page_data);
        } else {
            free(compressed);
        }
    }

    return status;
}

/* ============================================================================
 * Helper: Load and decode a new page (mmap path with zero-copy support)
 * ============================================================================
 */

static carquet_status_t load_next_page_mmap(
    carquet_column_reader_t* reader,
    carquet_error_t* error) {

    carquet_reader_t* file_reader = reader->file_reader;
    const uint8_t* mmap_data = file_reader->mmap_data;
    const parquet_column_metadata_t* col_meta = reader->col_meta;

    /* Load dictionary if needed (may update data_start_offset) */
    if (col_meta->has_dictionary_page_offset && !reader->has_dictionary) {
        carquet_status_t status = load_dictionary_page_mmap(reader, error);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    /* Parse page header directly from mmap */
    int64_t page_offset = reader->data_start_offset + reader->current_page;
    if (page_offset < 0 || (size_t)page_offset >= file_reader->file_size) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_PAGE, "Data page offset out of range");
        return CARQUET_ERROR_INVALID_PAGE;
    }
    const uint8_t* header_ptr = mmap_data + page_offset;
    size_t page_remaining = file_reader->file_size - (size_t)page_offset;
    size_t max_hdr = page_remaining < 256 ? page_remaining : 256;

    parquet_page_header_t page_header;
    size_t header_size;
    carquet_status_t status = parquet_parse_page_header(
        header_ptr, max_hdr, &page_header, &header_size, error);
    if (status != CARQUET_OK) {
        return status;
    }

    if (page_header.type != CARQUET_PAGE_DATA && page_header.type != CARQUET_PAGE_DATA_V2) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_PAGE, "Expected data page");
        return CARQUET_ERROR_INVALID_PAGE;
    }

    /* Get pointer to page data in mmap */
    const uint8_t* page_data_ptr = header_ptr + header_size;

    /* Verify CRC32 if present */
    if (page_header.has_crc && file_reader->options.verify_checksums) {
        uint32_t computed_crc = carquet_crc32(page_data_ptr, page_header.compressed_page_size);
        uint32_t expected_crc = (uint32_t)page_header.crc;
        if (computed_crc != expected_crc) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_CRC_MISMATCH,
                "Page CRC mismatch: expected 0x%08X, got 0x%08X at offset %lld",
                expected_crc, computed_crc, (long long)page_offset);
            return CARQUET_ERROR_CRC_MISMATCH;
        }
    }

    /* Extract num_values and encoding from the correct header union member */
    bool is_v2 = (page_header.type == CARQUET_PAGE_DATA_V2);
    int32_t num_values = is_v2 ? page_header.data_page_header_v2.num_values
                               : page_header.data_page_header.num_values;
    carquet_encoding_t page_encoding = is_v2 ? page_header.data_page_header_v2.encoding
                                             : page_header.data_page_header.encoding;
    size_t value_size = get_value_size(reader->type, reader->type_length);

    /* Check if zero-copy is possible (V1 only — V2 has levels interleaved) */
    bool zero_copy_eligible = !is_v2 && carquet_page_is_zero_copy_eligible(
        col_meta->codec, page_encoding, reader->type);

    /* Additional constraint: no definition/repetition levels for zero-copy
     * (levels require RLE decoding which modifies data layout) */
    bool has_levels = (reader->max_def_level > 0 || reader->max_rep_level > 0);

    if (zero_copy_eligible && !has_levels) {
        /* ====== ZERO-COPY PATH ====== */

        /* Free previous owned buffer if any */
        if (reader->decoded_ownership == CARQUET_DATA_OWNED) {
            free(reader->decoded_values);
        }

        /* Point directly to mmap data - no copy! */
        reader->decoded_values = (uint8_t*)page_data_ptr;
        reader->decoded_ownership = CARQUET_DATA_VIEW;

        /* Zero-copy path only triggers when max_def/rep == 0 (REQUIRED columns).
         * Level buffers are unused by callers for REQUIRED columns, so set to NULL
         * to avoid unnecessary allocation and memset overhead. */
        if (reader->decoded_def_levels) {
            free(reader->decoded_def_levels);
            reader->decoded_def_levels = NULL;
        }
        if (reader->decoded_rep_levels) {
            free(reader->decoded_rep_levels);
            reader->decoded_rep_levels = NULL;
        }
        reader->decoded_capacity = 0;

        reader->page_loaded = true;
        reader->page_num_values = num_values;
        reader->page_values_read = 0;
        reader->page_header_size = (int32_t)header_size;
        reader->page_compressed_size = page_header.compressed_page_size;

        return CARQUET_OK;
    }

    /* ====== STANDARD PATH (with decompression/decoding) ====== */

    const uint8_t* page_data;
    size_t page_size;
    uint8_t* decompressed = NULL;

    if (is_v2) {
        /* V2: levels are uncompressed, only data portion is compressed.
         * Layout: [rep_levels | def_levels | data]
         * We need to decompress only the data portion. */
        const parquet_data_page_header_v2_t* v2h = &page_header.data_page_header_v2;
        size_t levels_size = (size_t)v2h->repetition_levels_byte_length +
                             (size_t)v2h->definition_levels_byte_length;
        size_t compressed_data_size = (size_t)page_header.compressed_page_size - levels_size;

        if (levels_size > (size_t)page_header.compressed_page_size) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "V2 level sizes exceed page size");
            return CARQUET_ERROR_DECODE;
        }

        bool data_is_compressed = v2h->is_compressed &&
                                  col_meta->codec != CARQUET_COMPRESSION_UNCOMPRESSED;

        if (data_is_compressed) {
            /* Decompress only the data portion, keeping levels uncompressed */
            size_t uncompressed_data_size = (size_t)page_header.uncompressed_page_size - levels_size;
            size_t total_needed = levels_size + uncompressed_data_size;

            if (total_needed > reader->decompress_capacity) {
                free(reader->decompress_buffer);
                reader->decompress_buffer = malloc(total_needed);
                if (!reader->decompress_buffer) {
                    reader->decompress_capacity = 0;
                    CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate V2 decompress buffer");
                    return CARQUET_ERROR_OUT_OF_MEMORY;
                }
                reader->decompress_capacity = total_needed;
            }
            decompressed = reader->decompress_buffer;

            /* Copy uncompressed levels as-is */
            memcpy(decompressed, page_data_ptr, levels_size);

            /* Decompress data portion */
            size_t decompressed_data_size;
            status = decompress_page(col_meta->codec,
                page_data_ptr + levels_size, compressed_data_size,
                decompressed + levels_size, uncompressed_data_size,
                &decompressed_data_size);

            if (status != CARQUET_OK) {
                CARQUET_SET_ERROR(error, status, "Failed to decompress V2 page data");
                return status;
            }
            page_data = decompressed;
            page_size = levels_size + decompressed_data_size;
        } else {
            /* No compression — use data directly */
            page_data = page_data_ptr;
            page_size = page_header.compressed_page_size;
        }
    } else {
        /* V1: entire page is compressed together */
        if (col_meta->codec == CARQUET_COMPRESSION_UNCOMPRESSED) {
            page_data = page_data_ptr;
            page_size = page_header.compressed_page_size;
        } else {
            /* Must decompress - reuse buffer across pages when possible */
            size_t needed = page_header.uncompressed_page_size;
            if (needed > reader->decompress_capacity) {
                free(reader->decompress_buffer);
                reader->decompress_buffer = malloc(needed);
                if (!reader->decompress_buffer) {
                    reader->decompress_capacity = 0;
                    CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate decompress buffer");
                    return CARQUET_ERROR_OUT_OF_MEMORY;
                }
                reader->decompress_capacity = needed;
            }
            decompressed = reader->decompress_buffer;

            status = decompress_page(col_meta->codec,
                page_data_ptr, page_header.compressed_page_size,
                decompressed, page_header.uncompressed_page_size, &page_size);

            if (status != CARQUET_OK) {
                CARQUET_SET_ERROR(error, status, "Failed to decompress page");
                return status;
            }
            page_data = decompressed;
        }
    }

    if (!is_v2 && page_values_can_be_viewed_directly(reader, page_encoding)) {
        size_t required_bytes = value_size * (size_t)num_values;
        if (page_size < required_bytes) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Truncated PLAIN page payload");
            return CARQUET_ERROR_DECODE;
        }

        if (reader->decoded_ownership == CARQUET_DATA_OWNED) {
            free(reader->decoded_values);
        }
        reader->decoded_values = (uint8_t*)page_data;
        reader->decoded_ownership = CARQUET_DATA_VIEW;
        reader->decoded_capacity = 0;
        release_decoded_level_buffers(reader);

        reader->page_loaded = true;
        reader->page_num_values = num_values;
        reader->page_values_read = 0;
        reader->page_header_size = (int32_t)header_size;
        reader->page_compressed_size = page_header.compressed_page_size;

        return CARQUET_OK;
    }

    status = ensure_decoded_page_buffers(reader, num_values, value_size, error);
    if (status != CARQUET_OK) {
        return status;
    }

    /* Decode the page */
    int64_t decoded_count;
    if (is_v2) {
        status = carquet_read_data_page_v2(
            reader, page_data, page_size,
            &page_header.data_page_header_v2,
            reader->decoded_values, num_values,
            reader->decoded_def_levels, reader->decoded_rep_levels,
            &decoded_count, error);
    } else {
        status = carquet_read_data_page_v1(
            reader, page_data, page_size,
            &page_header.data_page_header,
            reader->decoded_values, num_values,
            reader->decoded_def_levels, reader->decoded_rep_levels,
            &decoded_count, error);
    }

    /* For BYTE_ARRAY PLAIN columns with compressed data, retain a copy of the
     * decompressed buffer since carquet_byte_array_t.data pointers reference it.
     * The decompression buffer itself is reused across pages, so we must copy. */
    if (decompressed && reader->type == CARQUET_PHYSICAL_BYTE_ARRAY &&
        page_encoding == CARQUET_ENCODING_PLAIN) {
        free(reader->page_data_for_values);
        reader->page_data_for_values = malloc(page_size);
        if (!reader->page_data_for_values) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to retain page data");
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        memcpy(reader->page_data_for_values, decompressed, page_size);
        /* Fixup BYTE_ARRAY pointers to reference the retained copy */
        ptrdiff_t offset = reader->page_data_for_values - decompressed;
        carquet_byte_array_t* ba = (carquet_byte_array_t*)reader->decoded_values;
        for (int64_t i = 0; i < decoded_count; i++) {
            if (ba[i].data) {
                ba[i].data = ba[i].data + offset;
            }
        }
    }

    if (status != CARQUET_OK) {
        return status;
    }

    reader->page_loaded = true;
    reader->page_num_values = (int32_t)decoded_count;
    reader->page_values_read = 0;
    reader->page_header_size = (int32_t)header_size;
    reader->page_compressed_size = page_header.compressed_page_size;

    return CARQUET_OK;
}

/* ============================================================================
 * Helper: Load and decode a new page (fread path)
 * ============================================================================
 */

static carquet_status_t load_next_page_fread(
    carquet_column_reader_t* reader,
    carquet_error_t* error) {

    carquet_reader_t* file_reader = reader->file_reader;
    const parquet_column_metadata_t* col_meta = reader->col_meta;

    /* Load dictionary if needed (may update data_start_offset) */
    if (col_meta->has_dictionary_page_offset && !reader->has_dictionary) {
        carquet_status_t status = load_dictionary_page_fread(reader, error);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    /* Read page header (from prebuffer cache or file) */
    int64_t data_offset = reader->data_start_offset;
    int64_t page_file_offset = data_offset + reader->current_page;

    uint8_t header_buf[256];
    size_t header_read = prebuf_read_at(file_reader, page_file_offset,
                                         header_buf, sizeof(header_buf));
    if (header_read < 8) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_READ, "Failed to read page header");
        return CARQUET_ERROR_FILE_READ;
    }

    parquet_page_header_t page_header;
    size_t header_size;
    carquet_status_t status = parquet_parse_page_header(
        header_buf, header_read, &page_header, &header_size, error);
    if (status != CARQUET_OK) {
        return status;
    }

    if (page_header.type != CARQUET_PAGE_DATA && page_header.type != CARQUET_PAGE_DATA_V2) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_PAGE, "Expected data page");
        return CARQUET_ERROR_INVALID_PAGE;
    }

    /* Reject absurdly large page sizes from malformed metadata (max 256 MB) */
    if (page_header.compressed_page_size <= 0 ||
        (size_t)page_header.compressed_page_size > (256ULL * 1024 * 1024)) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_PAGE, "Page size out of range");
        return CARQUET_ERROR_INVALID_PAGE;
    }

    /* Read compressed page data into a reusable buffer (from prebuffer or file) */
    if ((size_t)page_header.compressed_page_size > reader->page_buffer_capacity) {
        uint8_t* new_buffer = realloc(reader->page_buffer, (size_t)page_header.compressed_page_size);
        if (!new_buffer) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate page buffer");
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        reader->page_buffer = new_buffer;
        reader->page_buffer_capacity = (size_t)page_header.compressed_page_size;
    }
    reader->page_buffer_size = (size_t)page_header.compressed_page_size;
    uint8_t* compressed = reader->page_buffer;

    int64_t page_data_offset = page_file_offset + (int64_t)header_size;
    if (prebuf_read_at(file_reader, page_data_offset,
                       compressed, page_header.compressed_page_size) !=
        (size_t)page_header.compressed_page_size) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_READ, "Failed to read page data");
        return CARQUET_ERROR_FILE_READ;
    }

    /* Verify CRC32 if present */
    if (page_header.has_crc && file_reader->options.verify_checksums) {
        uint32_t computed_crc = carquet_crc32(compressed, page_header.compressed_page_size);
        uint32_t expected_crc = (uint32_t)page_header.crc;
        if (computed_crc != expected_crc) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_CRC_MISMATCH,
                "Page CRC mismatch: expected 0x%08X, got 0x%08X at offset %lld",
                expected_crc, computed_crc, (long long)(data_offset + reader->current_page));
            return CARQUET_ERROR_CRC_MISMATCH;
        }
    }

    /* Decompress if needed */
    uint8_t* page_data;
    size_t page_size;

    if (col_meta->codec == CARQUET_COMPRESSION_UNCOMPRESSED) {
        page_data = compressed;
        page_size = page_header.compressed_page_size;
    } else {
        size_t needed = (size_t)page_header.uncompressed_page_size;
        if (needed > reader->decompress_capacity) {
            uint8_t* new_buffer = realloc(reader->decompress_buffer, needed);
            if (!new_buffer) {
                CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate decompress buffer");
                return CARQUET_ERROR_OUT_OF_MEMORY;
            }
            reader->decompress_buffer = new_buffer;
            reader->decompress_capacity = needed;
        }
        page_data = reader->decompress_buffer;
        if (!page_data) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate decompress buffer");
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }

        status = decompress_page(col_meta->codec,
            compressed, page_header.compressed_page_size,
            page_data, page_header.uncompressed_page_size, &page_size);

        if (status != CARQUET_OK) {
            CARQUET_SET_ERROR(error, status, "Failed to decompress page");
            return status;
        }
    }

    /* Extract num_values and encoding from the correct header union member */
    bool is_v2 = (page_header.type == CARQUET_PAGE_DATA_V2);
    int32_t num_values = is_v2 ? page_header.data_page_header_v2.num_values
                               : page_header.data_page_header.num_values;
    carquet_encoding_t page_encoding = is_v2 ? page_header.data_page_header_v2.encoding
                                             : page_header.data_page_header.encoding;
    size_t value_size = get_value_size(reader->type, reader->type_length);

    /* V2 pages: decompress only the data portion (levels are uncompressed) */
    if (is_v2 && col_meta->codec != CARQUET_COMPRESSION_UNCOMPRESSED) {
        const parquet_data_page_header_v2_t* v2h = &page_header.data_page_header_v2;
        size_t levels_size = (size_t)v2h->repetition_levels_byte_length +
                             (size_t)v2h->definition_levels_byte_length;

        if (levels_size > page_size) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "V2 level sizes exceed page size");
            return CARQUET_ERROR_DECODE;
        }

        if (v2h->is_compressed) {
            size_t compressed_data_size = page_size - levels_size;
            size_t uncompressed_data_size = (size_t)page_header.uncompressed_page_size - levels_size;
            size_t total_needed = levels_size + uncompressed_data_size;

            if (total_needed > reader->decompress_capacity) {
                uint8_t* new_buffer = realloc(reader->decompress_buffer, total_needed);
                if (!new_buffer) {
                    CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate V2 decompress buffer");
                    return CARQUET_ERROR_OUT_OF_MEMORY;
                }
                reader->decompress_buffer = new_buffer;
                reader->decompress_capacity = total_needed;
            }

            /* Copy uncompressed levels, decompress data portion */
            memcpy(reader->decompress_buffer, page_data, levels_size);

            size_t decompressed_data_size;
            status = decompress_page(col_meta->codec,
                page_data + levels_size, compressed_data_size,
                reader->decompress_buffer + levels_size, uncompressed_data_size,
                &decompressed_data_size);

            if (status != CARQUET_OK) {
                CARQUET_SET_ERROR(error, status, "Failed to decompress V2 page data");
                return status;
            }

            page_data = reader->decompress_buffer;
            page_size = levels_size + decompressed_data_size;
        }
        /* else: is_compressed==false means data is also uncompressed, use as-is */
    }

    if (!is_v2 && page_values_can_be_viewed_directly(reader, page_encoding)) {
        size_t required_bytes = value_size * (size_t)num_values;
        if (page_size < required_bytes) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Truncated PLAIN page payload");
            return CARQUET_ERROR_DECODE;
        }

        if (reader->decoded_ownership == CARQUET_DATA_OWNED) {
            free(reader->decoded_values);
        }
        reader->decoded_values = page_data;
        reader->decoded_ownership = CARQUET_DATA_VIEW;
        reader->decoded_capacity = 0;
        release_decoded_level_buffers(reader);

        reader->page_loaded = true;
        reader->page_num_values = num_values;
        reader->page_values_read = 0;
        reader->page_header_size = (int32_t)header_size;
        reader->page_compressed_size = page_header.compressed_page_size;
        return CARQUET_OK;
    }

    status = ensure_decoded_page_buffers(reader, num_values, value_size, error);
    if (status != CARQUET_OK) {
        return status;
    }

    /* Decode the entire page into our buffers */
    int64_t decoded_count;
    if (is_v2) {
        status = carquet_read_data_page_v2(
            reader, page_data, page_size,
            &page_header.data_page_header_v2,
            reader->decoded_values, num_values,
            reader->decoded_def_levels, reader->decoded_rep_levels,
            &decoded_count, error);
    } else {
        status = carquet_read_data_page_v1(
            reader, page_data, page_size,
            &page_header.data_page_header,
            reader->decoded_values, num_values,
            reader->decoded_def_levels, reader->decoded_rep_levels,
            &decoded_count, error);
    }

    /* For BYTE_ARRAY PLAIN columns, the decoded carquet_byte_array_t structs
     * have .data pointers into the page data buffer. Retain the buffer so
     * these pointers remain valid until the next page is loaded. */
    bool retain = (reader->type == CARQUET_PHYSICAL_BYTE_ARRAY &&
                   page_encoding == CARQUET_ENCODING_PLAIN);

    if (retain) {
        free(reader->page_data_for_values);
        reader->page_data_for_values = malloc(page_size);
        if (!reader->page_data_for_values) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to retain page data");
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        memcpy(reader->page_data_for_values, page_data, page_size);

        ptrdiff_t offset = reader->page_data_for_values - page_data;
        carquet_byte_array_t* ba = (carquet_byte_array_t*)reader->decoded_values;
        for (int64_t i = 0; i < decoded_count; i++) {
            if (ba[i].data) {
                ba[i].data = ba[i].data + offset;
            }
        }
    }

    if (status != CARQUET_OK) {
        return status;
    }

    /* Update page tracking state */
    reader->page_loaded = true;
    reader->page_num_values = (int32_t)decoded_count;
    reader->page_values_read = 0;
    reader->page_header_size = (int32_t)header_size;
    reader->page_compressed_size = page_header.compressed_page_size;

    return CARQUET_OK;
}

/* ============================================================================
 * Helper: Load and decode a new page (dispatcher)
 * ============================================================================
 */

static carquet_status_t load_next_page(
    carquet_column_reader_t* reader,
    carquet_error_t* error) {

    carquet_reader_t* file_reader = reader->file_reader;

    /* Use mmap/buffer path if memory-mapped or buffer-based reader */
    if (file_reader->mmap_data != NULL) {
        return load_next_page_mmap(reader, error);
    }

    /* Fall back to fread path (requires valid file handle) */
    if (file_reader->file == NULL) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_STATE, "No data source available");
        return CARQUET_ERROR_INVALID_STATE;
    }
    return load_next_page_fread(reader, error);
}

/* ============================================================================
 * Page Loading Helper
 * ============================================================================
 */

carquet_status_t carquet_column_ensure_page_loaded(
    carquet_column_reader_t* reader,
    carquet_error_t* error) {

    if (!reader) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_ARGUMENT, "NULL reader");
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (!reader->page_loaded || reader->page_values_read >= reader->page_num_values) {
        if (reader->page_loaded) {
            reader->current_page += reader->page_header_size + reader->page_compressed_size;
            reader->page_loaded = false;
        }

        return load_next_page(reader, error);
    }

    return CARQUET_OK;
}

/* ============================================================================
 * Page Reading Entry Point
 * ============================================================================
 */

carquet_status_t carquet_read_next_page(
    carquet_column_reader_t* reader,
    void* values,
    int64_t max_values,
    int16_t* def_levels,
    int16_t* rep_levels,
    int64_t* values_read,
    carquet_error_t* error) {

    if (!reader || !values || !values_read) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_ARGUMENT, "NULL argument");
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    {
        carquet_status_t status = carquet_column_ensure_page_loaded(reader, error);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    if (max_values == 0) {
        *values_read = 0;
        return CARQUET_OK;
    }

    /* Calculate how many values to return from the current page */
    int32_t available = reader->page_num_values - reader->page_values_read;
    int32_t to_copy = (int32_t)max_values;
    if (to_copy > available) {
        to_copy = available;
    }
    if (to_copy <= 0) {
        *values_read = 0;
        return CARQUET_OK;
    }

    /* Copy values from decoded buffers */
    size_t value_size = get_value_size(reader->type, reader->type_length);
    size_t offset = (size_t)reader->page_values_read * value_size;

    memcpy(values, (uint8_t*)reader->decoded_values + offset, (size_t)to_copy * value_size);

    if (def_levels) {
        if (reader->decoded_def_levels) {
            memcpy(def_levels, reader->decoded_def_levels + reader->page_values_read,
                   (size_t)to_copy * sizeof(int16_t));
        } else {
            memset(def_levels, 0, (size_t)to_copy * sizeof(int16_t));
        }
    }
    if (rep_levels) {
        if (reader->decoded_rep_levels) {
            memcpy(rep_levels, reader->decoded_rep_levels + reader->page_values_read,
                   (size_t)to_copy * sizeof(int16_t));
        } else {
            memset(rep_levels, 0, (size_t)to_copy * sizeof(int16_t));
        }
    }

    /* Update state */
    reader->page_values_read += to_copy;
    reader->values_remaining -= to_copy;
    *values_read = to_copy;

    return CARQUET_OK;
}
