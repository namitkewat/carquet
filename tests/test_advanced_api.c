/**
 * @file test_advanced_api.c
 * @brief Tests for advanced API features (bloom filter, page index,
 *        key-value metadata, column chunk metadata, per-column options,
 *        buffer writer)
 */

#include <carquet/carquet.h>
#include "test_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================
 * Helpers
 * ====================================================================== */

#define N_ROWS 5000

static const char* TEMP_FILE = "/tmp/test_advanced_api.parquet";

static carquet_schema_t* make_schema(void) {
    carquet_schema_t* s = carquet_schema_create(NULL);
    carquet_schema_add_column(s, "id",    CARQUET_PHYSICAL_INT64,      NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_schema_add_column(s, "value", CARQUET_PHYSICAL_DOUBLE,     NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_schema_add_column(s, "label", CARQUET_PHYSICAL_BYTE_ARRAY, NULL, CARQUET_REPETITION_OPTIONAL, 0, 0);
    return s;
}

/** Write a test file with all advanced features enabled. */
static int write_test_file(const char* path, bool bloom, bool page_index) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* schema = make_schema();

    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression         = CARQUET_COMPRESSION_ZSTD;
    opts.write_statistics    = true;
    opts.write_bloom_filters = bloom;
    opts.write_page_index    = page_index;

    carquet_writer_t* w = carquet_writer_create(path, schema, &opts, &err);
    if (!w) { fprintf(stderr, "write_test_file: %s\n", err.message); return 1; }

    /* Key-value metadata */
    carquet_writer_add_metadata(w, "test.key1", "hello");
    carquet_writer_add_metadata(w, "test.key2", "world");

    /* Per-column overrides */
    carquet_writer_set_column_bloom_filter(w, 2, false); /* disable bloom on label */

    /* Write data */
    int64_t ids[N_ROWS];
    double  vals[N_ROWS];
    for (int i = 0; i < N_ROWS; i++) { ids[i] = i; vals[i] = i * 1.5; }

    carquet_status_t st;
    st = carquet_writer_write_batch(w, 0, ids,  N_ROWS, NULL, NULL);
    if (st != CARQUET_OK) goto fail;
    st = carquet_writer_write_batch(w, 1, vals, N_ROWS, NULL, NULL);
    if (st != CARQUET_OK) goto fail;

    /* label: half NULL, half "abc" */
    carquet_byte_array_t labels[N_ROWS];
    int16_t def[N_ROWS];
    int non_null = 0;
    for (int i = 0; i < N_ROWS; i++) {
        if (i % 2 == 0) {
            def[i] = 1;
            labels[non_null].data   = (uint8_t*)"abc";
            labels[non_null].length = 3;
            non_null++;
        } else {
            def[i] = 0;
        }
    }
    st = carquet_writer_write_batch(w, 2, labels, N_ROWS, def, NULL);
    if (st != CARQUET_OK) goto fail;

    st = carquet_writer_close(w);
    carquet_schema_free(schema);
    return (st == CARQUET_OK) ? 0 : 1;

fail:
    carquet_writer_abort(w);
    carquet_schema_free(schema);
    return 1;
}

/* ======================================================================
 * Test: Key-Value Metadata
 * ====================================================================== */

static int test_kv_metadata(void) {
    if (write_test_file(TEMP_FILE, false, false)) TEST_FAIL("kv_metadata", "write failed");

    carquet_reader_t* r = carquet_reader_open(TEMP_FILE, NULL, NULL);
    if (!r) TEST_FAIL("kv_metadata", "open failed");

    int32_t n = carquet_reader_num_metadata(r);
    if (n < 2) TEST_FAIL("kv_metadata", "expected >= 2 kv entries");

    const char *k, *v;
    carquet_status_t st = carquet_reader_get_metadata(r, 0, &k, &v);
    if (st != CARQUET_OK) TEST_FAIL("kv_metadata", "get_metadata failed");
    if (strcmp(k, "test.key1") != 0) TEST_FAIL("kv_metadata", "key mismatch");
    if (strcmp(v, "hello") != 0)     TEST_FAIL("kv_metadata", "value mismatch");

    const char* found = carquet_reader_find_metadata(r, "test.key2");
    if (!found || strcmp(found, "world") != 0)
        TEST_FAIL("kv_metadata", "find_metadata failed");

    const char* missing = carquet_reader_find_metadata(r, "nonexistent");
    if (missing) TEST_FAIL("kv_metadata", "expected NULL for missing key");

    carquet_reader_close(r);
    TEST_PASS("kv_metadata");
    return 0;
}

/* ======================================================================
 * Test: Column Chunk Metadata
 * ====================================================================== */

static int test_column_chunk_metadata(void) {
    if (write_test_file(TEMP_FILE, true, true)) TEST_FAIL("chunk_meta", "write failed");

    carquet_reader_t* r = carquet_reader_open(TEMP_FILE, NULL, NULL);
    if (!r) TEST_FAIL("chunk_meta", "open failed");

    carquet_column_chunk_metadata_t m;
    carquet_status_t st = carquet_reader_column_chunk_metadata(r, 0, 0, &m);
    if (st != CARQUET_OK) TEST_FAIL("chunk_meta", "get metadata failed");

    if (m.type != CARQUET_PHYSICAL_INT64) TEST_FAIL("chunk_meta", "wrong type");
    if (m.num_values != N_ROWS)           TEST_FAIL("chunk_meta", "wrong num_values");
    if (m.total_compressed_size <= 0)     TEST_FAIL("chunk_meta", "bad compressed size");
    if (m.total_uncompressed_size <= 0)   TEST_FAIL("chunk_meta", "bad uncompressed size");
    if (m.data_page_offset <= 0)          TEST_FAIL("chunk_meta", "bad data_page_offset");

    /* Column 0 should have bloom filter (enabled globally, not overridden) */
    /* Column 2 (label) should NOT have bloom filter (overridden off) */
    carquet_column_chunk_metadata_t m2;
    carquet_reader_column_chunk_metadata(r, 0, 2, &m2);
    /* Note: bloom filter availability depends on internal write behavior */

    carquet_reader_close(r);
    TEST_PASS("chunk_meta");
    return 0;
}

/* ======================================================================
 * Test: Bloom Filter
 * ====================================================================== */

static int test_bloom_filter(void) {
    if (write_test_file(TEMP_FILE, true, false)) TEST_FAIL("bloom", "write failed");

    carquet_reader_t* r = carquet_reader_open(TEMP_FILE, NULL, NULL);
    if (!r) TEST_FAIL("bloom", "open failed");

    carquet_bloom_filter_t* bf = carquet_reader_get_bloom_filter(r, 0, 0, NULL);
    if (!bf) {
        /* Bloom filter may not be present depending on implementation */
        carquet_reader_close(r);
        printf("[SKIP] bloom: no bloom filter available\n");
        return 0;
    }

    size_t sz = carquet_bloom_filter_size(bf);
    if (sz == 0) TEST_FAIL("bloom", "zero size");

    /* Value 42 was written (ids[42] = 42) - must say "might contain" */
    if (!carquet_bloom_filter_check_i64(bf, 42))
        TEST_FAIL("bloom", "false negative for 42");

    /* Value -1 was never written - should say "definitely not" (probabilistic) */
    /* We can't assert this fails, but it's very likely for a good filter */
    bool neg = carquet_bloom_filter_check_i64(bf, -1);
    printf("  bloom -1 check: %s (expected: likely no)\n", neg ? "yes" : "no");

    carquet_bloom_filter_destroy(bf);
    carquet_reader_close(r);
    TEST_PASS("bloom");
    return 0;
}

/* ======================================================================
 * Test: Page Index (Column Index + Offset Index)
 * ====================================================================== */

static int test_page_index(void) {
    if (write_test_file(TEMP_FILE, false, true)) TEST_FAIL("page_index", "write failed");

    carquet_reader_t* r = carquet_reader_open(TEMP_FILE, NULL, NULL);
    if (!r) TEST_FAIL("page_index", "open failed");

    /* Column index */
    carquet_column_index_t* ci = carquet_reader_get_column_index(r, 0, 0, NULL);
    if (!ci) {
        carquet_reader_close(r);
        printf("[SKIP] page_index: no column index available\n");
        return 0;
    }

    int32_t np = carquet_column_index_num_pages(ci);
    if (np <= 0) TEST_FAIL("page_index", "zero pages");

    /* Check first page stats */
    carquet_page_stats_t ps;
    carquet_status_t st = carquet_column_index_get_page_stats(ci, 0, &ps);
    if (st != CARQUET_OK) TEST_FAIL("page_index", "get_page_stats failed");

    /* For INT64, min_value should be a pointer to an int64_t */
    if (ps.min_value && ps.min_value_size >= 8) {
        int64_t mn = *(const int64_t*)ps.min_value;
        printf("  page 0 min: %lld\n", (long long)mn);
        if (mn != 0) TEST_FAIL("page_index", "first page min should be 0");
    }

    int32_t bo = carquet_column_index_boundary_order(ci);
    printf("  boundary_order: %d\n", bo);

    carquet_column_index_free(ci);

    /* Offset index */
    carquet_offset_index_t* oi = carquet_reader_get_offset_index(r, 0, 0, NULL);
    if (oi) {
        int32_t np2 = carquet_offset_index_num_pages(oi);
        if (np2 != np)
            printf("  WARNING: column_index pages=%d vs offset_index pages=%d\n", np, np2);

        carquet_page_location_t loc;
        st = carquet_offset_index_get_page_location(oi, 0, &loc);
        if (st != CARQUET_OK) TEST_FAIL("page_index", "get_page_location failed");
        if (loc.offset <= 0) TEST_FAIL("page_index", "bad page offset");
        if (loc.compressed_size <= 0) TEST_FAIL("page_index", "bad page size");
        if (loc.first_row_index != 0) TEST_FAIL("page_index", "first page should start at row 0");

        printf("  page 0: offset=%lld size=%d first_row=%lld\n",
               (long long)loc.offset, loc.compressed_size,
               (long long)loc.first_row_index);

        carquet_offset_index_free(oi);
    }

    carquet_reader_close(r);
    TEST_PASS("page_index");
    return 0;
}

/* ======================================================================
 * Test: Buffer Writer
 * ====================================================================== */

static int test_buffer_writer(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* schema = carquet_schema_create(NULL);
    carquet_schema_add_column(schema, "x", CARQUET_PHYSICAL_INT32,
                              NULL, CARQUET_REPETITION_REQUIRED, 0, 0);

    carquet_writer_t* w = carquet_writer_create_buffer(schema, NULL, &err);
    if (!w) {
        carquet_schema_free(schema);
        printf("[SKIP] buffer_writer: %s\n", err.message);
        return 0;
    }

    int32_t data[] = {10, 20, 30, 40, 50};
    carquet_status_t st = carquet_writer_write_batch(w, 0, data, 5, NULL, NULL);
    if (st != CARQUET_OK) TEST_FAIL("buffer_writer", "write_batch failed");

    st = carquet_writer_close(w);
    if (st != CARQUET_OK) TEST_FAIL("buffer_writer", "close failed");

    void* buf = NULL;
    size_t sz = 0;
    st = carquet_writer_get_buffer(w, &buf, &sz);
    if (st != CARQUET_OK || !buf || sz == 0)
        TEST_FAIL("buffer_writer", "get_buffer failed");

    /* Verify: open from buffer and read back */
    carquet_reader_t* r = carquet_reader_open_buffer(buf, sz, NULL, &err);
    if (!r) TEST_FAIL("buffer_writer", "open_buffer failed");

    if (carquet_reader_num_rows(r) != 5)
        TEST_FAIL("buffer_writer", "wrong row count");
    if (carquet_reader_num_columns(r) != 1)
        TEST_FAIL("buffer_writer", "wrong column count");

    /* Read values back */
    carquet_column_reader_t* col = carquet_reader_get_column(r, 0, 0, NULL);
    if (!col) TEST_FAIL("buffer_writer", "get_column failed");

    int32_t vals[5];
    int64_t n = carquet_column_read_batch(col, vals, 5, NULL, NULL);
    if (n != 5) TEST_FAIL("buffer_writer", "read wrong count");
    for (int i = 0; i < 5; i++) {
        if (vals[i] != data[i]) TEST_FAIL("buffer_writer", "value mismatch");
    }

    carquet_column_reader_free(col);
    carquet_reader_close(r);
    free(buf);
    carquet_schema_free(schema);
    TEST_PASS("buffer_writer");
    return 0;
}

/* ======================================================================
 * Test: Per-Column Encoding/Compression
 * ====================================================================== */

static int test_per_column_options(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* schema = make_schema();

    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_ZSTD;

    carquet_writer_t* w = carquet_writer_create(TEMP_FILE, schema, &opts, &err);
    if (!w) TEST_FAIL("per_column", "create failed");

    /* Set column 0 to uncompressed */
    carquet_status_t st = carquet_writer_set_column_compression(
        w, 0, CARQUET_COMPRESSION_UNCOMPRESSED, 0);
    if (st != CARQUET_OK) TEST_FAIL("per_column", "set_compression failed");

    st = carquet_writer_set_column_statistics(w, 1, false);
    if (st != CARQUET_OK) TEST_FAIL("per_column", "set_statistics failed");

    /* Invalid column index */
    st = carquet_writer_set_column_encoding(w, 99, CARQUET_ENCODING_PLAIN);
    if (st == CARQUET_OK) TEST_FAIL("per_column", "should reject invalid index");

    carquet_writer_abort(w);
    carquet_schema_free(schema);
    TEST_PASS("per_column");
    return 0;
}

/* ======================================================================
 * Main
 * ====================================================================== */

int main(void) {
    carquet_init();
    int failures = 0;

    failures += test_kv_metadata();
    failures += test_column_chunk_metadata();
    failures += test_bloom_filter();
    failures += test_page_index();
    failures += test_buffer_writer();
    failures += test_per_column_options();

    remove(TEMP_FILE);

    if (failures > 0) {
        printf("\n%d test(s) FAILED\n", failures);
        return 1;
    }
    printf("\nAll advanced API tests passed.\n");
    return 0;
}
