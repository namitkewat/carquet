/**
 * @file fuzz_encodings.c
 * @brief Fuzz target for all carquet encoding codecs
 *
 * Tests decoders for: RLE, Delta (INT32/INT64), Plain (all types),
 * Dictionary (all types), Byte Stream Split, Delta Length Byte Array,
 * and boolean pack/unpack via SIMD dispatch.
 *
 * First byte selects encoding mode, second byte provides parameters.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <carquet/carquet.h>

/* Internal encoding headers — use proper headers to avoid signature drift */
#include "encoding/rle.h"
#include "encoding/plain.h"

/* Delta decode/encode — no header, forward-declare with correct signatures */
carquet_status_t carquet_delta_decode_int32(
    const uint8_t* data, size_t data_size,
    int32_t* values, int32_t num_values, size_t* bytes_consumed);
carquet_status_t carquet_delta_decode_int64(
    const uint8_t* data, size_t data_size,
    int64_t* values, int32_t num_values, size_t* bytes_consumed);

/* Dictionary decode — correct signatures with dict_count parameter */
carquet_status_t carquet_dictionary_decode_int32(
    const uint8_t* dict_data, size_t dict_size, int32_t dict_count,
    const uint8_t* indices_data, size_t indices_size,
    int32_t* output, int64_t output_count);
carquet_status_t carquet_dictionary_decode_int64(
    const uint8_t* dict_data, size_t dict_size, int32_t dict_count,
    const uint8_t* indices_data, size_t indices_size,
    int64_t* output, int64_t output_count);
carquet_status_t carquet_dictionary_decode_float(
    const uint8_t* dict_data, size_t dict_size, int32_t dict_count,
    const uint8_t* indices_data, size_t indices_size,
    float* output, int64_t output_count);
carquet_status_t carquet_dictionary_decode_double(
    const uint8_t* dict_data, size_t dict_size, int32_t dict_count,
    const uint8_t* indices_data, size_t indices_size,
    double* output, int64_t output_count);

/* Byte stream split decode */
carquet_status_t carquet_byte_stream_split_decode_float(
    const uint8_t* data, size_t data_size, float* values, int64_t count);
carquet_status_t carquet_byte_stream_split_decode_double(
    const uint8_t* data, size_t data_size, double* values, int64_t count);
carquet_status_t carquet_byte_stream_split_decode(
    const uint8_t* data, size_t data_size, int32_t type_length,
    uint8_t* values, int64_t count);

/* Delta length byte array decode */
carquet_status_t carquet_delta_length_decode(
    const uint8_t* data, size_t data_size,
    carquet_byte_array_t* values, int32_t num_values, size_t* bytes_consumed);

/* SIMD dispatch for bool packing */
void carquet_dispatch_unpack_bools(const uint8_t* input, uint8_t* output, int64_t count);
void carquet_dispatch_pack_bools(const uint8_t* input, uint8_t* output, int64_t count);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 3) return 0;
    (void)carquet_init();

    uint8_t encoding = data[0];
    uint8_t param = data[1];
    const uint8_t* payload = data + 2;
    size_t payload_size = size - 2;

    int64_t max_values = 10000;
    void* output = malloc((size_t)max_values * 16);
    if (!output) return 0;

    switch (encoding % 17) {
        case 0: {
            /* RLE decode — variable bit width */
            int bit_width = (param % 32) + 1;
            (void)carquet_rle_decode_all(payload, payload_size, bit_width,
                                         (uint32_t*)output, max_values);
            break;
        }
        case 1: {
            /* Delta INT32 */
            int32_t count = (param % 200) + 1;
            size_t consumed = 0;
            (void)carquet_delta_decode_int32(payload, payload_size,
                                             (int32_t*)output, count, &consumed);
            break;
        }
        case 2: {
            /* Delta INT64 */
            int32_t count = (param % 200) + 1;
            size_t consumed = 0;
            (void)carquet_delta_decode_int64(payload, payload_size,
                                             (int64_t*)output, count, &consumed);
            break;
        }
        case 3: {
            /* Plain INT32 */
            int64_t count = payload_size / 4;
            if (count > max_values) count = max_values;
            if (count > 0)
                (void)carquet_decode_plain_int32(payload, payload_size,
                                                 (int32_t*)output, count);
            break;
        }
        case 4: {
            /* Plain INT64 */
            int64_t count = payload_size / 8;
            if (count > max_values) count = max_values;
            if (count > 0)
                (void)carquet_decode_plain_int64(payload, payload_size,
                                                 (int64_t*)output, count);
            break;
        }
        case 5: {
            /* Plain DOUBLE */
            int64_t count = payload_size / 8;
            if (count > max_values) count = max_values;
            if (count > 0)
                (void)carquet_decode_plain_double(payload, payload_size,
                                                  (double*)output, count);
            break;
        }
        case 6: {
            /* Dictionary INT32 — split payload into dict + indices */
            if (payload_size < 8) break;
            int32_t dict_count = (param % 64) + 1;
            size_t dict_size = (size_t)dict_count * 4;
            if (dict_size >= payload_size) break;
            size_t indices_size = payload_size - dict_size;
            int64_t num_values = (param % 200) + 1;
            if (num_values > max_values) num_values = max_values;
            (void)carquet_dictionary_decode_int32(
                payload, dict_size, dict_count,
                payload + dict_size, indices_size,
                (int32_t*)output, num_values);
            break;
        }
        case 7: {
            /* Dictionary INT64 */
            if (payload_size < 16) break;
            int32_t dict_count = (param % 32) + 1;
            size_t dict_size = (size_t)dict_count * 8;
            if (dict_size >= payload_size) break;
            size_t indices_size = payload_size - dict_size;
            int64_t num_values = (param % 200) + 1;
            if (num_values > max_values) num_values = max_values;
            (void)carquet_dictionary_decode_int64(
                payload, dict_size, dict_count,
                payload + dict_size, indices_size,
                (int64_t*)output, num_values);
            break;
        }
        case 8: {
            /* Dictionary FLOAT */
            if (payload_size < 8) break;
            int32_t dict_count = (param % 64) + 1;
            size_t dict_size = (size_t)dict_count * 4;
            if (dict_size >= payload_size) break;
            size_t indices_size = payload_size - dict_size;
            int64_t num_values = (param % 200) + 1;
            if (num_values > max_values) num_values = max_values;
            (void)carquet_dictionary_decode_float(
                payload, dict_size, dict_count,
                payload + dict_size, indices_size,
                (float*)output, num_values);
            break;
        }
        case 9: {
            /* Dictionary DOUBLE */
            if (payload_size < 16) break;
            int32_t dict_count = (param % 32) + 1;
            size_t dict_size = (size_t)dict_count * 8;
            if (dict_size >= payload_size) break;
            size_t indices_size = payload_size - dict_size;
            int64_t num_values = (param % 200) + 1;
            if (num_values > max_values) num_values = max_values;
            (void)carquet_dictionary_decode_double(
                payload, dict_size, dict_count,
                payload + dict_size, indices_size,
                (double*)output, num_values);
            break;
        }
        case 10: {
            /* Byte stream split FLOAT */
            int64_t count = payload_size / 4;
            if (count > max_values) count = max_values;
            if (count > 0)
                (void)carquet_byte_stream_split_decode_float(
                    payload, (size_t)count * 4, (float*)output, count);
            break;
        }
        case 11: {
            /* Byte stream split DOUBLE */
            int64_t count = payload_size / 8;
            if (count > max_values) count = max_values;
            if (count > 0)
                (void)carquet_byte_stream_split_decode_double(
                    payload, (size_t)count * 8, (double*)output, count);
            break;
        }
        case 12: {
            /* Plain BOOLEAN */
            int64_t count = payload_size * 8;
            if (count > max_values) count = max_values;
            if (count > 0)
                (void)carquet_decode_plain_boolean(payload, payload_size,
                                                   (uint8_t*)output, count);
            break;
        }
        case 13: {
            /* Plain FLOAT */
            int64_t count = payload_size / 4;
            if (count > max_values) count = max_values;
            if (count > 0)
                (void)carquet_decode_plain_float(payload, payload_size,
                                                 (float*)output, count);
            break;
        }
        case 14: {
            /* Delta length byte array */
            int32_t count = (param % 100) + 1;
            carquet_byte_array_t* arrays = calloc((size_t)count, sizeof(carquet_byte_array_t));
            if (arrays) {
                size_t consumed = 0;
                (void)carquet_delta_length_decode(payload, payload_size,
                                                  arrays, count, &consumed);
                free(arrays);
            }
            break;
        }
        case 15: {
            /* RLE levels decode (int16 output) */
            int bit_width = (param % 16) + 1;
            int16_t* levels = (int16_t*)output;
            (void)carquet_rle_decode_levels(payload, payload_size, bit_width,
                                            levels, max_values);
            break;
        }
        case 16: {
            /* Byte stream split generic (variable type_length) */
            int32_t type_len = (param % 8) + 1;
            int64_t count = payload_size / type_len;
            if (count > max_values) count = max_values;
            if (count > 0)
                (void)carquet_byte_stream_split_decode(
                    payload, (size_t)count * type_len,
                    type_len, (uint8_t*)output, count);
            break;
        }
    }

    free(output);
    return 0;
}

#ifdef AFL_MAIN
#include <stdio.h>
#include <sys/stat.h>
int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "Usage: %s <input_file>\n", argv[0]); return 1; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    struct stat st; fstat(fileno(f), &st);
    uint8_t* d = malloc((size_t)st.st_size);
    if (!d) { fclose(f); return 1; }
    fread(d, 1, (size_t)st.st_size, f); fclose(f);
    int r = LLVMFuzzerTestOneInput(d, (size_t)st.st_size);
    free(d); return r;
}
#endif
