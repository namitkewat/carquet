/**
 * @file fuzz_writer.c
 * @brief Fuzz target for carquet Parquet writer
 *
 * Tests the writer with random schemas, data types (including BYTE_ARRAY),
 * compression codecs, nullable columns, and various configuration options,
 * then verifies the output is readable via roundtrip test.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <carquet/carquet.h>

#define MAX_COLUMNS 16
#define MAX_ROWS 500
#define MAX_STRING_LEN 128

typedef struct {
    const uint8_t* data;
    size_t size;
    size_t pos;
} fuzz_input_t;

static uint8_t consume_byte(fuzz_input_t* in) {
    if (in->pos >= in->size) return 0;
    return in->data[in->pos++];
}
static uint16_t consume_u16(fuzz_input_t* in) {
    uint16_t lo = consume_byte(in), hi = consume_byte(in);
    return (hi << 8) | lo;
}
static uint32_t consume_u32(fuzz_input_t* in) {
    uint32_t a = consume_byte(in), b = consume_byte(in);
    uint32_t c = consume_byte(in), d = consume_byte(in);
    return a | (b << 8) | (c << 16) | (d << 24);
}
static uint64_t consume_u64(fuzz_input_t* in) {
    uint64_t lo = consume_u32(in), hi = consume_u32(in);
    return lo | (hi << 32);
}
static float consume_float(fuzz_input_t* in) {
    union { uint32_t u; float f; } v; v.u = consume_u32(in); return v.f;
}
static double consume_double(fuzz_input_t* in) {
    union { uint64_t u; double d; } v; v.u = consume_u64(in); return v.d;
}

static const carquet_physical_type_t FUZZ_TYPES[] = {
    CARQUET_PHYSICAL_BOOLEAN,
    CARQUET_PHYSICAL_INT32,
    CARQUET_PHYSICAL_INT64,
    CARQUET_PHYSICAL_FLOAT,
    CARQUET_PHYSICAL_DOUBLE,
    CARQUET_PHYSICAL_BYTE_ARRAY,
};
#define NUM_FUZZ_TYPES (sizeof(FUZZ_TYPES) / sizeof(FUZZ_TYPES[0]))

static const carquet_compression_t FUZZ_CODECS[] = {
    CARQUET_COMPRESSION_UNCOMPRESSED,
    CARQUET_COMPRESSION_SNAPPY,
    CARQUET_COMPRESSION_GZIP,
    CARQUET_COMPRESSION_LZ4,
    CARQUET_COMPRESSION_ZSTD,
};
#define NUM_FUZZ_CODECS (sizeof(FUZZ_CODECS) / sizeof(FUZZ_CODECS[0]))

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 10) return 0;
    (void)carquet_init();

    fuzz_input_t input = { data, size, 0 };
    carquet_error_t err = CARQUET_ERROR_INIT;

    uint8_t num_columns = (consume_byte(&input) % MAX_COLUMNS) + 1;
    uint16_t num_rows = (consume_u16(&input) % MAX_ROWS) + 1;
    uint8_t codec_idx = consume_byte(&input) % NUM_FUZZ_CODECS;
    uint8_t nullable_mask = consume_byte(&input);
    uint8_t config_byte = consume_byte(&input);

    carquet_compression_t codec = FUZZ_CODECS[codec_idx];

    /* Create schema */
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) return 0;

    carquet_physical_type_t* col_types = malloc(num_columns * sizeof(carquet_physical_type_t));
    if (!col_types) { carquet_schema_free(schema); return 0; }

    for (int i = 0; i < num_columns; i++) {
        char col_name[32];
        snprintf(col_name, sizeof(col_name), "col_%d", i);

        uint8_t type_idx = consume_byte(&input) % NUM_FUZZ_TYPES;
        col_types[i] = FUZZ_TYPES[type_idx];

        carquet_field_repetition_t rep = (nullable_mask & (1 << (i % 8)))
            ? CARQUET_REPETITION_OPTIONAL : CARQUET_REPETITION_REQUIRED;

        if (carquet_schema_add_column(schema, col_name, col_types[i], NULL, rep, 0, 0) != CARQUET_OK) {
            free(col_types);
            carquet_schema_free(schema);
            return 0;
        }
    }

    /* Temp file */
    char tmp_path[] = "/tmp/fuzz_writer_XXXXXX";
    int fd = mkstemp(tmp_path);
    if (fd < 0) { free(col_types); carquet_schema_free(schema); return 0; }
    close(fd);

    /* Writer options — vary based on fuzz input */
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = codec;

    /* Vary page and row group sizes */
    uint32_t page_sizes[] = { 1024, 4096, 8192, 16384 };
    uint32_t rg_sizes[] = { 16384, 65536, 131072, 262144 };
    opts.page_size = page_sizes[config_byte & 0x03];
    opts.row_group_size = rg_sizes[(config_byte >> 2) & 0x03];

    carquet_writer_t* writer = carquet_writer_create(tmp_path, schema, &opts, &err);
    if (!writer) {
        remove(tmp_path); free(col_types); carquet_schema_free(schema);
        return 0;
    }

    /* Allocate value buffers */
    void* values = malloc(num_rows * 16);
    int16_t* def_levels = malloc(num_rows * sizeof(int16_t));
    carquet_byte_array_t* ba_values = NULL;
    char* string_pool = NULL;

    if (!values || !def_levels) goto cleanup_write;

    /* Write each column */
    for (int col = 0; col < num_columns; col++) {
        bool is_nullable = nullable_mask & (1 << (col % 8));
        void* write_ptr = values;

        switch (col_types[col]) {
            case CARQUET_PHYSICAL_BOOLEAN: {
                uint8_t* bools = (uint8_t*)values;
                for (int i = 0; i < num_rows; i++) {
                    bools[i] = consume_byte(&input) & 1;
                    def_levels[i] = is_nullable ? (consume_byte(&input) & 1) : 1;
                }
                break;
            }
            case CARQUET_PHYSICAL_INT32: {
                int32_t* ints = (int32_t*)values;
                for (int i = 0; i < num_rows; i++) {
                    ints[i] = (int32_t)consume_u32(&input);
                    def_levels[i] = is_nullable ? (consume_byte(&input) & 1) : 1;
                }
                break;
            }
            case CARQUET_PHYSICAL_INT64: {
                int64_t* longs = (int64_t*)values;
                for (int i = 0; i < num_rows; i++) {
                    longs[i] = (int64_t)consume_u64(&input);
                    def_levels[i] = is_nullable ? (consume_byte(&input) & 1) : 1;
                }
                break;
            }
            case CARQUET_PHYSICAL_FLOAT: {
                float* floats = (float*)values;
                for (int i = 0; i < num_rows; i++) {
                    floats[i] = consume_float(&input);
                    def_levels[i] = is_nullable ? (consume_byte(&input) & 1) : 1;
                }
                break;
            }
            case CARQUET_PHYSICAL_DOUBLE: {
                double* doubles = (double*)values;
                for (int i = 0; i < num_rows; i++) {
                    doubles[i] = consume_double(&input);
                    def_levels[i] = is_nullable ? (consume_byte(&input) & 1) : 1;
                }
                break;
            }
            case CARQUET_PHYSICAL_BYTE_ARRAY: {
                /* Real BYTE_ARRAY testing */
                if (!ba_values) {
                    ba_values = calloc(MAX_ROWS, sizeof(carquet_byte_array_t));
                    string_pool = malloc(MAX_ROWS * MAX_STRING_LEN);
                }
                if (!ba_values || !string_pool) goto cleanup_write;

                for (int i = 0; i < num_rows; i++) {
                    def_levels[i] = is_nullable ? (consume_byte(&input) & 1) : 1;
                    if (def_levels[i]) {
                        uint8_t slen = consume_byte(&input) % MAX_STRING_LEN;
                        char* s = string_pool + (i * MAX_STRING_LEN);
                        for (int j = 0; j < slen; j++)
                            s[j] = (char)(consume_byte(&input) % 95 + 32); /* printable ASCII */
                        ba_values[i].data = (uint8_t*)s;
                        ba_values[i].length = slen;
                    } else {
                        ba_values[i].data = NULL;
                        ba_values[i].length = 0;
                    }
                }
                write_ptr = ba_values;
                break;
            }
            default:
                break;
        }

        carquet_status_t status = carquet_writer_write_batch(
            writer, col, write_ptr, num_rows,
            is_nullable ? def_levels : NULL, NULL);

        if (status != CARQUET_OK) goto cleanup_write;
    }

    {
        carquet_status_t close_status = carquet_writer_close(writer);
        writer = NULL; /* writer consumed by close */

        free(values); values = NULL;
        free(def_levels); def_levels = NULL;
        free(ba_values); ba_values = NULL;
        free(string_pool); string_pool = NULL;

        if (close_status != CARQUET_OK) {
            free(col_types);
            carquet_schema_free(schema);
            remove(tmp_path);
            return 0;
        }
    }

    /* Roundtrip: read back and verify metadata */
    {
        carquet_reader_t* reader = carquet_reader_open(tmp_path, NULL, &err);
        if (reader) {
            int64_t read_rows = carquet_reader_num_rows(reader);
            int32_t read_cols = carquet_reader_num_columns(reader);

            if (read_rows != num_rows || read_cols != num_columns) {
                carquet_reader_close(reader);
                free(col_types);
                carquet_schema_free(schema);
                remove(tmp_path);
                __builtin_trap();
            }

            /* Read all data via batch reader */
            carquet_batch_reader_config_t config;
            carquet_batch_reader_config_init(&config);
            config.batch_size = 100;

            carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(reader, &config, &err);
            if (batch_reader) {
                carquet_row_batch_t* batch = NULL;
                int64_t total_read = 0;
                while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
                    int64_t n = carquet_row_batch_num_rows(batch);
                    /* Access column data to exercise decoders */
                    for (int32_t c = 0; c < read_cols; c++) {
                        const void* col_data;
                        const uint8_t* nulls;
                        int64_t count;
                        (void)carquet_row_batch_column(batch, c, &col_data, &nulls, &count);
                    }
                    total_read += n;
                    carquet_row_batch_free(batch);
                    batch = NULL;
                }
                carquet_batch_reader_free(batch_reader);

                if (total_read != num_rows) {
                    carquet_reader_close(reader);
                    free(col_types);
                    carquet_schema_free(schema);
                    remove(tmp_path);
                    __builtin_trap();
                }
            }

            /* Also exercise low-level column reader */
            int32_t num_rg = carquet_reader_num_row_groups(reader);
            for (int32_t rg = 0; rg < num_rg && rg < 5; rg++) {
                carquet_row_group_metadata_t rg_meta;
                if (carquet_reader_row_group_metadata(reader, rg, &rg_meta) != CARQUET_OK)
                    continue;
                for (int32_t c = 0; c < read_cols && c < 8; c++) {
                    carquet_column_reader_t* col_reader =
                        carquet_reader_get_column(reader, rg, c, &err);
                    if (col_reader) {
                        void* col_vals = malloc((size_t)rg_meta.num_rows * 16);
                        int16_t* dl = malloc((size_t)rg_meta.num_rows * sizeof(int16_t));
                        if (col_vals && dl) {
                            (void)carquet_column_read_batch(col_reader, col_vals,
                                                           rg_meta.num_rows, dl, NULL);
                        }
                        free(col_vals);
                        free(dl);
                        carquet_column_reader_free(col_reader);
                    }
                }
            }

            carquet_reader_close(reader);
        }
    }

    free(col_types);
    carquet_schema_free(schema);
    remove(tmp_path);
    return 0;

cleanup_write:
    free(values);
    free(def_levels);
    free(ba_values);
    free(string_pool);
    if (writer) carquet_writer_abort(writer);
    free(col_types);
    carquet_schema_free(schema);
    remove(tmp_path);
    return 0;
}

#ifdef AFL_MAIN
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
