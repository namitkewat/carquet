/**
 * @file allocator.c
 * @brief Global memory-allocator registry for the Carquet library.
 *
 * @section overview Overview
 *
 * Carquet decouples itself from the C standard library allocator by routing
 * all externally observable memory-management operations through a thin
 * dispatch layer: the @c carquet_allocator_t interface.  This file owns
 * the single process-wide instance of that interface (@c g_allocator) and
 * exposes two public symbols:
 *
 *   - carquet_set_allocator() — replace the active allocator (or reset to
 *     the default C allocator by passing @c NULL).
 *   - carquet_get_allocator()  — inspect the currently active allocator
 *     (useful for diagnostics and for composing allocators, e.g. a
 *     statistics-gathering wrapper around an existing backend).
 *
 * @section default Default allocator
 *
 * Unless the application calls carquet_set_allocator(), the library behaves
 * as if the following C99 shim were installed:
 *
 * @code{.c}
 * static void* cstdlib_malloc (size_t n,        void* ctx) { (void)ctx; return malloc(n);       }
 * static void* cstdlib_realloc(void* p, size_t n, void* ctx) { (void)ctx; return realloc(p, n);  }
 * static void  cstdlib_free   (void* p,           void* ctx) { (void)ctx; free(p);               }
 * @endcode
 *
 * The wrappers are required because the standard @c malloc / @c realloc /
 * @c free signatures do not carry the @c ctx parameter demanded by the
 * @c carquet_allocator_t function-pointer contract.
 *
 * @section integration Integration model
 *
 * @subsection immediate Immediate allocation (current behaviour)
 *
 * In the current release, Carquet's internal subsystems (codec buffers,
 * schema trees, I/O staging buffers, …) allocate directly via the C
 * standard library.  The allocator registry therefore acts as the
 * authoritative source of truth for embedders that wrap or intercept
 * Carquet's public API while performing their own bookkeeping above the
 * library layer.
 *
 * @subsection roadmap Planned internal integration
 *
 * Future releases will migrate every internal @c malloc / @c realloc /
 * @c free call site to dispatch through @c g_allocator, making the custom
 * allocator the single control point for all memory used by the library.
 * No API change is required for that migration — the @c carquet_allocator_t
 * interface is already fully specified.
 *
 * @section threadsafety Thread-safety contract
 *
 * carquet_set_allocator() is intentionally *not* thread-safe.  It must be
 * called exactly once during program initialisation, before any concurrent
 * Carquet activity begins.  This matches the design of analogous hooks in
 * other C libraries (OpenSSL's @c CRYPTO_set_mem_functions, jemalloc's
 * @c je_malloc_usable_size, etc.) and avoids the overhead of a per-call
 * memory barrier on the hot allocation path.
 *
 * carquet_get_allocator() reads the global without synchronisation and is
 * therefore safe to call from any thread once the allocator has been set.
 *
 * @section validation Input validation
 *
 * carquet_set_allocator() performs a complete NULL-pointer check on every
 * field of the supplied @c carquet_allocator_t before committing the new
 * allocator.  If any function pointer is NULL the call is a no-op and the
 * previous allocator remains active.  Passing @c NULL for the @p allocator
 * argument itself unconditionally resets to the default C allocator.
 *
 * @section example Extended example — memory-tracking allocator
 *
 * The following snippet shows how to build a lightweight allocator shim
 * that counts live allocations without replacing the underlying C allocator:
 *
 * @code{.c}
 * #include <carquet/carquet.h>
 * #include <stdatomic.h>
 * #include <stdlib.h>
 *
 * static atomic_long g_live_allocs = 0;
 *
 * static void* tracking_malloc(size_t n, void* ctx) {
 *     (void)ctx;
 *     void* p = malloc(n);
 *     if (p) atomic_fetch_add(&g_live_allocs, 1);
 *     return p;
 * }
 * static void* tracking_realloc(void* p, size_t n, void* ctx) {
 *     (void)ctx;
 *     void* q = realloc(p, n);
 *     if (!p && q)  atomic_fetch_add(&g_live_allocs, 1);  // NULL → allocation
 *     if (p && !q && n == 0) atomic_fetch_sub(&g_live_allocs, 1); // deallocation
 *     return q;
 * }
 * static void tracking_free(void* p, void* ctx) {
 *     (void)ctx;
 *     if (p) {
 *         atomic_fetch_sub(&g_live_allocs, 1);
 *         free(p);
 *     }
 * }
 *
 * void install_tracking_allocator(void) {
 *     static const carquet_allocator_t tracker = {
 *         .malloc  = tracking_malloc,
 *         .realloc = tracking_realloc,
 *         .free    = tracking_free,
 *         .ctx     = NULL,
 *     };
 *     carquet_set_allocator(&tracker);
 * }
 *
 * long carquet_live_allocation_count(void) {
 *     return atomic_load(&g_live_allocs);
 * }
 * @endcode
 */

#include <carquet/carquet.h>

#include <stddef.h>
#include <stdlib.h>

/* ============================================================================
 * Module-private helpers — default C standard-library allocator shims
 * ============================================================================
 *
 * The standard C functions malloc/realloc/free cannot be stored directly in
 * carquet_allocator_t because their signatures lack the 'ctx' parameter.
 * These thin wrappers absorb the unused context argument and forward to the
 * corresponding standard function, giving the default allocator the same
 * call interface as any custom allocator.
 */

/**
 * @brief Default malloc shim: delegates to the C standard library malloc.
 *
 * @param[in] size  Number of bytes to allocate.  A request for zero bytes
 *                  behaves as defined by the C standard (implementation-defined
 *                  return, either NULL or a unique non-NULL pointer).
 * @param[in] ctx   Ignored; present only to satisfy the allocator interface.
 * @return          Pointer to newly allocated, uninitialized memory, or NULL
 *                  on allocation failure.
 */
static void* default_malloc(size_t size, void* ctx)
{
    (void)ctx;
    return malloc(size);
}

/**
 * @brief Default realloc shim: delegates to the C standard library realloc.
 *
 * Handles all four realloc edge-case variants correctly:
 *   - ptr==NULL, size >0  →  equivalent to malloc(size)
 *   - ptr!=NULL, size==0  →  equivalent to free(ptr), returns NULL or unique ptr
 *   - ptr!=NULL, size >0  →  resize the existing allocation
 *   - ptr==NULL, size==0  →  implementation-defined (C11 §7.22.3.5)
 *
 * @param[in]  ptr   Existing allocation to resize, or NULL to allocate fresh.
 * @param[in]  size  New desired byte count.
 * @param[in]  ctx   Ignored; present only to satisfy the allocator interface.
 * @return           Pointer to resized allocation, or NULL on failure.
 *                   On failure the original @p ptr is left unchanged.
 */
static void* default_realloc(void* ptr, size_t size, void* ctx)
{
    (void)ctx;
    return realloc(ptr, size);
}

/**
 * @brief Default free shim: delegates to the C standard library free.
 *
 * Passing a NULL @p ptr is well-defined (a no-op, per C11 §7.22.3.3).
 *
 * @param[in]  ptr   Pointer previously returned by default_malloc or
 *                   default_realloc, or NULL.
 * @param[in]  ctx   Ignored; present only to satisfy the allocator interface.
 */
static void default_free(void* ptr, void* ctx)
{
    (void)ctx;
    free(ptr);
}

/* ============================================================================
 * Module-private state
 * ============================================================================
 *
 * A single process-wide instance of carquet_allocator_t.  Initialised to the
 * default C allocator shims so that the library is functional from the very
 * first allocation, before any call to carquet_set_allocator().
 *
 * Declared 'static' to give the symbol internal linkage; the public API is
 * exposed only through carquet_set_allocator() / carquet_get_allocator().
 */
static carquet_allocator_t g_allocator = {
    /* Use the shim wrappers, not &malloc / &realloc / &free directly, because
     * the standard function signatures do not include the 'ctx' parameter
     * required by the carquet_allocator_t function-pointer contract.        */
    .malloc  = default_malloc,
    .realloc = default_realloc,
    .free    = default_free,
    .ctx     = NULL,
};

/* ============================================================================
 * Public API implementation
 * ============================================================================ */

/**
 * @brief Install a custom memory allocator as the library-wide allocator.
 *
 * Replaces the currently active @c carquet_allocator_t with the one pointed
 * to by @p allocator, after validating that none of its function pointers are
 * NULL.  If @p allocator is @c NULL, or if any of its function fields are
 * @c NULL, the function resets the allocator to the built-in C standard
 * library shims (equivalent to the state before any call to this function).
 *
 * @par Lifecycle guarantee
 * The library copies the entire @c carquet_allocator_t value into its
 * internal state; the caller's original struct need not remain live after
 * this call returns.
 *
 * @par Thread-safety
 * This function is @b not thread-safe.  Call it during process initialisation
 * before spawning threads that use Carquet, or ensure exclusive access through
 * external synchronisation.
 *
 * @param[in] allocator
 *   Pointer to a fully populated @c carquet_allocator_t, or @c NULL to reset
 *   to the default C allocator.  When non-NULL, all three function pointers
 *   (@c malloc, @c realloc, @c free) must be non-NULL; the @c ctx field may
 *   be @c NULL.
 *
 * @see carquet_get_allocator()
 */
void carquet_set_allocator(const carquet_allocator_t* allocator)
{
    /* NULL argument or incomplete allocator: restore the default C shims.
     * Checking all three pointers guards against partially-initialised structs
     * passed by caller (e.g. zero-initialised on the stack).               */
    if (allocator == NULL
            || allocator->malloc  == NULL
            || allocator->realloc == NULL
            || allocator->free    == NULL)
    {
        g_allocator.malloc  = default_malloc;
        g_allocator.realloc = default_realloc;
        g_allocator.free    = default_free;
        g_allocator.ctx     = NULL;
        return;
    }

    /* Atomically (from the calling thread's perspective) replace the entire
     * allocator record with the caller-supplied value.                      */
    g_allocator = *allocator;
}

/**
 * @brief Return a pointer to the currently active library allocator.
 *
 * The returned pointer is valid for the lifetime of the process; it always
 * refers to the current @c g_allocator snapshot.  Callers must not free or
 * modify the returned pointer.
 *
 * @par Composing allocators
 * A common pattern is to call @c carquet_get_allocator() before installing a
 * custom allocator, store the returned pointer, and delegate to it from the
 * new allocator's functions.  This enables transparent layering (e.g. a
 * statistics shim on top of a pool allocator).
 *
 * @par Thread-safety
 * Safe to call concurrently from multiple threads @e after
 * carquet_set_allocator() has returned.  The read is not protected by a lock,
 * but since the write in carquet_set_allocator() is required to happen-before
 * any concurrent Carquet usage, no data race occurs in a correctly written
 * program.
 *
 * @return Non-NULL pointer to the active @c carquet_allocator_t.
 *
 * @see carquet_set_allocator()
 */
const carquet_allocator_t* carquet_get_allocator(void)
{
    return &g_allocator;
}
