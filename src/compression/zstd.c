/**
 * @file zstd.c
 * @brief ZSTD compression/decompression using libzstd
 *
 * Uses streaming context for better performance on repeated decompressions.
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <zstd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Thread-local storage compatibility:
 * - __thread requires macOS 10.7+ / iOS 9.0+
 * - For older targets, use pthread-based TLS
 */
#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#if defined(MAC_OS_X_VERSION_MIN_REQUIRED) && MAC_OS_X_VERSION_MIN_REQUIRED < 1070
#define CARQUET_USE_PTHREAD_TLS 1
#endif
#endif

/* Thread-local decompression contexts for parallel column reading */
#ifdef _OPENMP

#if defined(CARQUET_USE_PTHREAD_TLS)
/* Use pthread TLS for older macOS (< 10.7) */
#include <pthread.h>

static pthread_key_t tls_dctx_key;
static pthread_once_t tls_dctx_once = PTHREAD_ONCE_INIT;
static pthread_key_t tls_cctx_key;
static pthread_once_t tls_cctx_once = PTHREAD_ONCE_INIT;

static void destroy_dctx(void* ctx) {
    if (ctx) {
        ZSTD_freeDCtx((ZSTD_DCtx*)ctx);
    }
}

static void destroy_cctx(void* ctx) {
    if (ctx) {
        ZSTD_freeCCtx((ZSTD_CCtx*)ctx);
    }
}

static void init_tls_key(void) {
    pthread_key_create(&tls_dctx_key, destroy_dctx);
}

static void init_cctx_key(void) {
    pthread_key_create(&tls_cctx_key, destroy_cctx);
}

static ZSTD_DCtx* get_dctx(void) {
    pthread_once(&tls_dctx_once, init_tls_key);
    ZSTD_DCtx* dctx = (ZSTD_DCtx*)pthread_getspecific(tls_dctx_key);
    if (!dctx) {
        dctx = ZSTD_createDCtx();
        if (dctx) {
            pthread_setspecific(tls_dctx_key, dctx);
        }
    }
    return dctx;
}

static ZSTD_CCtx* get_cctx(void) {
    pthread_once(&tls_cctx_once, init_cctx_key);
    ZSTD_CCtx* cctx = (ZSTD_CCtx*)pthread_getspecific(tls_cctx_key);
    if (!cctx) {
        cctx = ZSTD_createCCtx();
        if (cctx) {
            pthread_setspecific(tls_cctx_key, cctx);
        }
    }
    return cctx;
}

#else
/* Use thread-local storage for modern systems */
#ifdef _MSC_VER
static __declspec(thread) ZSTD_DCtx* tls_dctx = NULL;
static __declspec(thread) ZSTD_CCtx* tls_cctx = NULL;
#else
static __thread ZSTD_DCtx* tls_dctx = NULL;
static __thread ZSTD_CCtx* tls_cctx = NULL;
#endif

static ZSTD_DCtx* get_dctx(void) {
    if (!tls_dctx) {
        tls_dctx = ZSTD_createDCtx();
    }
    return tls_dctx;
}

static ZSTD_CCtx* get_cctx(void) {
    if (!tls_cctx) {
        tls_cctx = ZSTD_createCCtx();
    }
    return tls_cctx;
}
#endif /* CARQUET_USE_PTHREAD_TLS */

#else
/* No OpenMP - use global contexts */
static ZSTD_DCtx* global_dctx = NULL;
static ZSTD_CCtx* global_cctx = NULL;

static ZSTD_DCtx* get_dctx(void) {
    if (!global_dctx) {
        global_dctx = ZSTD_createDCtx();
    }
    return global_dctx;
}

static ZSTD_CCtx* get_cctx(void) {
    if (!global_cctx) {
        global_cctx = ZSTD_createCCtx();
    }
    return global_cctx;
}
#endif /* _OPENMP */

int carquet_zstd_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!src || !dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Use streaming context for better buffer reuse */
    ZSTD_DCtx* dctx = get_dctx();
    if (!dctx) {
        /* Fallback to simple API */
        size_t result = ZSTD_decompress(dst, dst_capacity, src, src_size);
        if (ZSTD_isError(result)) {
            return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
        }
        *dst_size = result;
        return CARQUET_OK;
    }

    size_t result = ZSTD_decompressDCtx(dctx, dst, dst_capacity, src, src_size);
    if (ZSTD_isError(result)) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    *dst_size = result;
    return CARQUET_OK;
}

int carquet_zstd_compress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size,
    int level) {

    if (!src || !dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (level < 1) level = 1;
    if (level > ZSTD_maxCLevel()) level = ZSTD_maxCLevel();

    /* Use cached context for repeated compressions (e.g., per-page).
     * Only enable multi-threading for large inputs (>4MB) where the
     * parallelism overhead is worthwhile. */
    ZSTD_CCtx* cctx = get_cctx();
    if (cctx) {
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);

        /* Only use multi-threading for large inputs where parallelism
         * outweighs coordination overhead. For typical 1MB pages, single-
         * threaded with a cached context is faster. */
        if (src_size > 4 * 1024 * 1024) {
            ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, 4);
        } else {
            ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, 0);
        }

        size_t result = ZSTD_compress2(cctx, dst, dst_capacity, src, src_size);
        if (!ZSTD_isError(result)) {
            *dst_size = result;
            return CARQUET_OK;
        }
    }

    /* Fallback to simple API */
    size_t result = ZSTD_compress(dst, dst_capacity, src, src_size, level);
    if (ZSTD_isError(result)) {
        return CARQUET_ERROR_COMPRESSION;
    }

    *dst_size = result;
    return CARQUET_OK;
}

size_t carquet_zstd_compress_bound(size_t src_size) {
    return ZSTD_compressBound(src_size);
}

void carquet_zstd_init_tables(void) {
    /* No-op - libzstd handles initialization internally */
}
