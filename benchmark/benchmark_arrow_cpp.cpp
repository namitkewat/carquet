#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/result.h>
#include <arrow/util/config.h>

#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>

#define WARMUP_ITERATIONS 3
#define BENCH_ITERATIONS_SMALL 51
#define BENCH_ITERATIONS_MEDIUM 21
#define BENCH_ITERATIONS_LARGE 11
#define MAX_BENCH_ITERATIONS 51

typedef struct {
    const char* name;
    int rows;
} dataset_t;

typedef struct {
    parquet::Compression::type codec;
    const char* name;
} compression_config_t;

typedef struct {
    int num_rows;
    std::shared_ptr<arrow::Table> table;
} test_data_t;

static int get_benchmark_zstd_level(void) {
    const char* env = std::getenv("CARQUET_BENCH_ZSTD_LEVEL");
    if (!env || env[0] == '\0') {
        return 1;
    }

    char* end = NULL;
    long level = std::strtol(env, &end, 10);
    if (end == env || *end != '\0') {
        return 1;
    }
    if (level < 1) level = 1;
    if (level > 22) level = 22;
    return static_cast<int>(level);
}

static double get_time_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    return static_cast<double>(counter.QuadPart) * 1000.0 /
           static_cast<double>(freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
#endif
}

static long get_file_size(const char* filename) {
    struct stat st;
    if (stat(filename, &st) == 0) {
        return static_cast<long>(st.st_size);
    }
    return 0;
}

static const char* get_temp_dir(void) {
    const char* override = std::getenv("CARQUET_BENCH_TMPDIR");
    if (override && override[0] != '\0') {
        return override;
    }
#ifdef _WIN32
    static char temp_dir[512] = {0};
    if (temp_dir[0] == 0) {
        const char* tmp = std::getenv("TEMP");
        if (!tmp) tmp = std::getenv("TMP");
        if (!tmp) tmp = ".";
        std::snprintf(temp_dir, sizeof(temp_dir), "%s", tmp);
    }
    return temp_dir;
#else
    return "/tmp";
#endif
}

static void purge_file_cache(const char* filename) {
#if defined(__APPLE__)
    char tmp[512];
    std::snprintf(tmp, sizeof(tmp), "%s.nocache", filename);

    int src = open(filename, O_RDONLY);
    if (src < 0) return;

    int dst = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (dst < 0) {
        close(src);
        return;
    }

    fcntl(dst, F_NOCACHE, 1);
    fcntl(src, F_NOCACHE, 1);

    char buf[262144];
    ssize_t n = 0;
    while ((n = read(src, buf, sizeof(buf))) > 0) {
        (void)write(dst, buf, static_cast<size_t>(n));
    }

    close(src);
    close(dst);
    unlink(filename);
    rename(tmp, filename);
#elif defined(__linux__)
    int fd = open(filename, O_RDONLY);
    if (fd < 0) return;

    struct stat st;
    if (fstat(fd, &st) == 0) {
        posix_fadvise(fd, 0, st.st_size, POSIX_FADV_DONTNEED);
    }
    close(fd);
#else
    (void)filename;
#endif
}

static uint32_t lcg_state = 42;

static void lcg_seed(uint32_t seed) {
    lcg_state = seed;
}

static uint32_t lcg_rand(void) {
    lcg_state = lcg_state * 1103515245u + 12345u;
    return (lcg_state >> 16) & 0x7FFFu;
}

static double lcg_normal(void) {
    double u1 = (lcg_rand() + 1.0) / 32768.0;
    double u2 = (lcg_rand() + 1.0) / 32768.0;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979 * u2);
}

static bool check_status(const char* what, const arrow::Status& status) {
    if (status.ok()) {
        return true;
    }
    std::fprintf(stderr, "%s: %s\n", what, status.ToString().c_str());
    return false;
}

template <typename T>
static bool take_result(const char* what, arrow::Result<T> result, T* out) {
    if (!result.ok()) {
        std::fprintf(stderr, "%s: %s\n", what, result.status().ToString().c_str());
        return false;
    }
    *out = result.MoveValueUnsafe();
    return true;
}

static std::unique_ptr<test_data_t> test_data_create(int num_rows) {
    std::unique_ptr<arrow::Buffer> ids_buf_tmp;
    std::unique_ptr<arrow::Buffer> values_buf_tmp;
    std::unique_ptr<arrow::Buffer> categories_buf_tmp;

    if (!take_result("Allocate ids buffer",
                     arrow::AllocateBuffer(num_rows * static_cast<int64_t>(sizeof(int64_t))),
                     &ids_buf_tmp)) {
        return nullptr;
    }
    if (!take_result("Allocate values buffer",
                     arrow::AllocateBuffer(num_rows * static_cast<int64_t>(sizeof(double))),
                     &values_buf_tmp)) {
        return nullptr;
    }
    if (!take_result("Allocate categories buffer",
                     arrow::AllocateBuffer(num_rows * static_cast<int64_t>(sizeof(int32_t))),
                     &categories_buf_tmp)) {
        return nullptr;
    }

    std::shared_ptr<arrow::Buffer> ids_buf(std::move(ids_buf_tmp));
    std::shared_ptr<arrow::Buffer> values_buf(std::move(values_buf_tmp));
    std::shared_ptr<arrow::Buffer> categories_buf(std::move(categories_buf_tmp));

    int64_t* ids = reinterpret_cast<int64_t*>(ids_buf->mutable_data());
    double* values = reinterpret_cast<double*>(values_buf->mutable_data());
    int32_t* categories = reinterpret_cast<int32_t*>(categories_buf->mutable_data());

    lcg_seed(42);
    for (int i = 0; i < num_rows; ++i) {
        ids[i] = 1000000 + static_cast<int64_t>(lcg_rand() % 9000000u);
        values[i] = std::fabs(100.0 + 50.0 * lcg_normal());
        categories[i] = static_cast<int32_t>(lcg_rand() % 100u);
    }

    auto ids_array = std::make_shared<arrow::Int64Array>(num_rows, ids_buf);
    auto values_array = std::make_shared<arrow::DoubleArray>(num_rows, values_buf);
    auto categories_array = std::make_shared<arrow::Int32Array>(num_rows, categories_buf);

    auto td = std::make_unique<test_data_t>();
    td->num_rows = num_rows;
    td->table = arrow::Table::Make(
        arrow::schema({
            arrow::field("id", arrow::int64(), false),
            arrow::field("value", arrow::float64(), false),
            arrow::field("category", arrow::int32(), false),
        }),
        {ids_array, values_array, categories_array},
        num_rows);
    return td;
}

static double trimmed_median(const double* values, int n) {
    std::vector<double> sorted(values, values + n);
    std::sort(sorted.begin(), sorted.end());
    if (n <= 2) {
        return sorted[n / 2];
    }
    int lo = 1;
    int hi = n - 2;
    int trimmed = hi - lo + 1;
    if ((trimmed & 1) != 0) {
        return sorted[lo + trimmed / 2];
    }
    return (sorted[lo + trimmed / 2 - 1] + sorted[lo + trimmed / 2]) / 2.0;
}

template <typename ArrayType>
static void accumulate_first_last(const std::shared_ptr<arrow::ChunkedArray>& column,
                                  volatile int64_t* checksum) {
    for (const auto& chunk : column->chunks()) {
        auto array = std::static_pointer_cast<ArrayType>(chunk);
        if (array->length() <= 0) {
            continue;
        }
        *checksum += static_cast<int64_t>(array->Value(0));
        *checksum += static_cast<int64_t>(array->Value(array->length() - 1));
    }
}

static double benchmark_write(const char* filename, const test_data_t* td,
                              parquet::Compression::type codec) {
    double start = get_time_ms();

    std::shared_ptr<arrow::io::FileOutputStream> sink;
    if (!take_result("Open output file",
                     arrow::io::FileOutputStream::Open(filename, false), &sink)) {
        return -1.0;
    }

    parquet::WriterProperties::Builder props_builder;
    props_builder.disable_dictionary();
    props_builder.enable_page_checksum();
    props_builder.compression(codec);

    if (codec != parquet::Compression::UNCOMPRESSED) {
        props_builder.encoding("value", parquet::Encoding::BYTE_STREAM_SPLIT);
    }
    if (codec == parquet::Compression::ZSTD) {
        props_builder.compression_level(get_benchmark_zstd_level());
    }

    parquet::ArrowWriterProperties::Builder arrow_props_builder;
    auto props = props_builder.build();
    auto arrow_props = arrow_props_builder.build();

    int64_t rg_rows = td->num_rows > 1000000 ? td->num_rows / 10 : 100000;

    arrow::Status status = parquet::arrow::WriteTable(
        *td->table,
        arrow::default_memory_pool(),
        sink,
        rg_rows,
        props,
        arrow_props);
    if (!check_status("WriteTable", status)) {
        (void)sink->Close();
        return -1.0;
    }
    if (!check_status("Close output file", sink->Close())) {
        return -1.0;
    }
    return get_time_ms() - start;
}

static double benchmark_read(const char* filename, int expected_rows) {
    double start = get_time_ms();

    parquet::ReaderProperties reader_props;
    reader_props.set_page_checksum_verification(false);

    parquet::ArrowReaderProperties arrow_props;
    arrow_props.set_use_threads(true);

    parquet::arrow::FileReaderBuilder builder;
    if (!check_status("Open input file",
                      builder.OpenFile(filename, true, reader_props))) {
        return -1.0;
    }
    builder.properties(arrow_props);

    std::unique_ptr<parquet::arrow::FileReader> reader;
    if (!take_result("Build file reader", builder.Build(), &reader)) {
        return -1.0;
    }
    reader->set_use_threads(true);

    std::shared_ptr<arrow::Table> table;
    if (!check_status("Read table", reader->ReadTable(&table))) {
        return -1.0;
    }

    volatile int64_t checksum = 0;
    accumulate_first_last<arrow::Int64Array>(table->column(0), &checksum);
    accumulate_first_last<arrow::DoubleArray>(table->column(1), &checksum);
    accumulate_first_last<arrow::Int32Array>(table->column(2), &checksum);
    (void)checksum;

    if (table->num_rows() != expected_rows) {
        std::fprintf(stderr, "Warning: row count mismatch %lld vs %d\n",
                     static_cast<long long>(table->num_rows()), expected_rows);
    }
    table.reset();
    reader.reset();
    return get_time_ms() - start;
}

static bool run_benchmark(const char* dataset_name, int num_rows,
                          parquet::Compression::type codec,
                          const char* compression_name) {
    char filename[256];
    std::snprintf(filename, sizeof(filename), "%s/benchmark_%s_%s_arrow_cpp.parquet",
                  get_temp_dir(), dataset_name, compression_name);

    std::printf("\n=== %s (%d rows, %s) ===\n", dataset_name, num_rows, compression_name);

    std::unique_ptr<test_data_t> td = test_data_create(num_rows);
    if (!td) {
        std::fprintf(stderr, "Failed to create test data\n");
        return false;
    }

    int iters = BENCH_ITERATIONS_LARGE;
    if (num_rows <= 100000) {
        iters = BENCH_ITERATIONS_SMALL;
    } else if (num_rows <= 1000000) {
        iters = BENCH_ITERATIONS_MEDIUM;
    }

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        if (benchmark_write(filename, td.get(), codec) <= 0.0) {
            std::remove(filename);
            return false;
        }
    }

    double write_times[MAX_BENCH_ITERATIONS];
    for (int i = 0; i < iters; ++i) {
        write_times[i] = benchmark_write(filename, td.get(), codec);
        if (write_times[i] <= 0.0) {
            std::remove(filename);
            return false;
        }
    }

    long file_size = get_file_size(filename);
    if (file_size <= 0) {
        std::fprintf(stderr, "Benchmark file is empty or missing\n");
        std::remove(filename);
        return false;
    }
    purge_file_cache(filename);

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        if (benchmark_read(filename, num_rows) <= 0.0) {
            std::remove(filename);
            return false;
        }
    }

    double read_times[MAX_BENCH_ITERATIONS];
    for (int i = 0; i < iters; ++i) {
        read_times[i] = benchmark_read(filename, num_rows);
        if (read_times[i] <= 0.0) {
            std::remove(filename);
            return false;
        }
    }

    double write_med = trimmed_median(write_times, iters);
    double read_med = trimmed_median(read_times, iters);
    double rows_per_sec_write = (num_rows / write_med) * 1000.0;
    double rows_per_sec_read = (num_rows / read_med) * 1000.0;

    std::printf("  Write: %.2f ms (%.2f M rows/sec)\n",
                write_med, rows_per_sec_write / 1e6);
    std::printf("  Read:  %.2f ms (%.2f M rows/sec)\n",
                read_med, rows_per_sec_read / 1e6);
    std::printf("  File:  %.2f MB (%.2f bytes/row)\n",
                file_size / (1024.0 * 1024.0), static_cast<double>(file_size) / num_rows);

    std::printf("CSV:arrow_cpp,%s,%s,%d,%.2f,%.2f,%ld\n",
                dataset_name, compression_name, num_rows, write_med, read_med, file_size);

    std::remove(filename);
    return true;
}

static int find_dataset(const char* name, dataset_t* datasets, int n) {
    for (int i = 0; i < n; ++i) {
        if (std::strcmp(datasets[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static int find_compression(const char* name, compression_config_t* comps, int n) {
    for (int i = 0; i < n; ++i) {
        if (std::strcmp(comps[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

int main(int argc, char* argv[]) {
    std::setvbuf(stdout, NULL, _IONBF, 0);

    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        std::printf("%s\n", ARROW_VERSION_STRING);
        return 0;
    }

    dataset_t datasets[] = {
        {"small", 100000},
        {"medium", 1000000},
        {"large", 10000000},
        {"xlarge", 100000000}
    };
    int num_datasets = sizeof(datasets) / sizeof(datasets[0]);

    compression_config_t compressions[] = {
        {parquet::Compression::UNCOMPRESSED, "none"},
        {parquet::Compression::SNAPPY, "snappy"},
        {parquet::Compression::ZSTD, "zstd"},
        {parquet::Compression::LZ4, "lz4"}
    };
    int num_compressions = sizeof(compressions) / sizeof(compressions[0]);

    if (argc == 3) {
        int d = find_dataset(argv[1], datasets, num_datasets);
        int c = find_compression(argv[2], compressions, num_compressions);
        if (d < 0 || c < 0) {
            std::fprintf(stderr, "Usage: %s [dataset] [compression]\n", argv[0]);
            std::fprintf(stderr, "  datasets:     small, medium, large, xlarge\n");
            std::fprintf(stderr, "  compressions: none, snappy, zstd, lz4\n");
            return 1;
        }
        return run_benchmark(datasets[d].name, datasets[d].rows,
                             compressions[c].codec, compressions[c].name) ? 0 : 1;
    }

    std::printf("Arrow C++ Benchmark\n");

    for (int d = 0; d < num_datasets; ++d) {
        for (int c = 0; c < num_compressions; ++c) {
            if (!run_benchmark(datasets[d].name, datasets[d].rows,
                               compressions[c].codec, compressions[c].name)) {
                return 1;
            }
        }
    }

    std::printf("\nBenchmark complete.\n");
    return 0;
}
