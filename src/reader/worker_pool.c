/**
 * @file worker_pool.c
 * @brief Persistent thread pool for batch reader parallelism
 *
 * A minimal, high-performance worker pool using pthreads (POSIX) or
 * Windows threads. Workers spin on a condition variable waiting for tasks.
 * The pool persists across batch_reader_next() calls, eliminating the
 * per-batch OpenMP fork/join overhead (~10-50us per batch × 400 batches).
 */

#include "worker_pool.h"
#include <stdlib.h>

/* ============================================================================
 * Platform Abstraction
 * ============================================================================ */

#ifdef _WIN32

static DWORD WINAPI worker_thread_func(LPVOID arg);

#define POOL_LOCK(p)   EnterCriticalSection(&(p)->mutex)
#define POOL_UNLOCK(p) LeaveCriticalSection(&(p)->mutex)
#define POOL_WAIT_WORK(p)  SleepConditionVariableCS(&(p)->work_available, &(p)->mutex, INFINITE)
#define POOL_SIGNAL_WORK(p) WakeConditionVariable(&(p)->work_available)
#define POOL_BROADCAST_WORK(p) WakeAllConditionVariable(&(p)->work_available)
#define POOL_SIGNAL_DONE(p) WakeAllConditionVariable(&(p)->work_done)
#define POOL_WAIT_DONE(p)  SleepConditionVariableCS(&(p)->work_done, &(p)->mutex, INFINITE)

#else

static void* worker_thread_func(void* arg);

#define POOL_LOCK(p)   pthread_mutex_lock(&(p)->mutex)
#define POOL_UNLOCK(p) pthread_mutex_unlock(&(p)->mutex)
#define POOL_WAIT_WORK(p)  pthread_cond_wait(&(p)->work_available, &(p)->mutex)
#define POOL_SIGNAL_WORK(p) pthread_cond_signal(&(p)->work_available)
#define POOL_BROADCAST_WORK(p) pthread_cond_broadcast(&(p)->work_available)
#define POOL_SIGNAL_DONE(p) pthread_cond_broadcast(&(p)->work_done)
#define POOL_WAIT_DONE(p)  pthread_cond_wait(&(p)->work_done, &(p)->mutex)

#endif

/* ============================================================================
 * Worker Thread
 * ============================================================================ */

#ifdef _WIN32
static DWORD WINAPI worker_thread_func(LPVOID arg) {
#else
static void* worker_thread_func(void* arg) {
#endif
    carquet_worker_pool_t* pool = (carquet_worker_pool_t*)arg;

    for (;;) {
        POOL_LOCK(pool);

        /* Wait for work or shutdown */
        while (pool->queue_count == 0 && !pool->shutdown) {
            POOL_WAIT_WORK(pool);
        }

        if (pool->shutdown && pool->queue_count == 0) {
            POOL_UNLOCK(pool);
            break;
        }

        /* Dequeue task */
        carquet_task_t task = pool->queue[pool->queue_head];
        pool->queue_head = (pool->queue_head + 1) % CARQUET_POOL_QUEUE_CAPACITY;
        pool->queue_count--;
        pool->active_tasks++;
        POOL_UNLOCK(pool);

        /* Execute task outside lock */
        task.fn(task.arg);

        POOL_LOCK(pool);
        pool->active_tasks--;
        if (pool->active_tasks == 0 && pool->queue_count == 0) {
            POOL_SIGNAL_DONE(pool);
        }
        POOL_UNLOCK(pool);
    }

#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

/* ============================================================================
 * Pool Lifecycle
 * ============================================================================ */

carquet_worker_pool_t* carquet_worker_pool_create(int32_t num_threads) {
    if (num_threads < 1) return NULL;

    carquet_worker_pool_t* pool = calloc(1, sizeof(carquet_worker_pool_t));
    if (!pool) return NULL;

    pool->num_threads = num_threads;
    pool->shutdown = false;
    pool->queue_head = 0;
    pool->queue_tail = 0;
    pool->queue_count = 0;
    pool->active_tasks = 0;

#ifdef _WIN32
    InitializeCriticalSection(&pool->mutex);
    InitializeConditionVariable(&pool->work_available);
    InitializeConditionVariable(&pool->work_done);

    pool->threads = calloc(num_threads, sizeof(HANDLE));
    if (!pool->threads) {
        DeleteCriticalSection(&pool->mutex);
        free(pool);
        return NULL;
    }
    for (int32_t i = 0; i < num_threads; i++) {
        pool->threads[i] = CreateThread(NULL, 0, worker_thread_func, pool, 0, NULL);
        if (!pool->threads[i]) {
            pool->shutdown = true;
            WakeAllConditionVariable(&pool->work_available);
            for (int32_t j = 0; j < i; j++) {
                WaitForSingleObject(pool->threads[j], INFINITE);
                CloseHandle(pool->threads[j]);
            }
            DeleteCriticalSection(&pool->mutex);
            free(pool->threads);
            free(pool);
            return NULL;
        }
    }
#else
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->work_available, NULL);
    pthread_cond_init(&pool->work_done, NULL);

    pool->threads = calloc(num_threads, sizeof(pthread_t));
    if (!pool->threads) {
        pthread_mutex_destroy(&pool->mutex);
        pthread_cond_destroy(&pool->work_available);
        pthread_cond_destroy(&pool->work_done);
        free(pool);
        return NULL;
    }
    for (int32_t i = 0; i < num_threads; i++) {
        if (pthread_create(&pool->threads[i], NULL, worker_thread_func, pool) != 0) {
            pool->shutdown = true;
            pthread_cond_broadcast(&pool->work_available);
            for (int32_t j = 0; j < i; j++) {
                pthread_join(pool->threads[j], NULL);
            }
            pthread_mutex_destroy(&pool->mutex);
            pthread_cond_destroy(&pool->work_available);
            pthread_cond_destroy(&pool->work_done);
            free(pool->threads);
            free(pool);
            return NULL;
        }
    }
#endif

    return pool;
}

void carquet_worker_pool_submit(carquet_worker_pool_t* pool,
                                 carquet_task_fn fn, void* arg) {
    POOL_LOCK(pool);

    /* Spin-wait if queue is full (shouldn't happen with reasonable usage) */
    while (pool->queue_count >= CARQUET_POOL_QUEUE_CAPACITY) {
        POOL_UNLOCK(pool);
        POOL_LOCK(pool);
    }

    pool->queue[pool->queue_tail].fn = fn;
    pool->queue[pool->queue_tail].arg = arg;
    pool->queue_tail = (pool->queue_tail + 1) % CARQUET_POOL_QUEUE_CAPACITY;
    pool->queue_count++;

    POOL_SIGNAL_WORK(pool);
    POOL_UNLOCK(pool);
}

void carquet_worker_pool_wait(carquet_worker_pool_t* pool) {
    POOL_LOCK(pool);
    while (pool->queue_count > 0 || pool->active_tasks > 0) {
        POOL_WAIT_DONE(pool);
    }
    POOL_UNLOCK(pool);
}

void carquet_worker_pool_parallel_for(carquet_worker_pool_t* pool,
                                       carquet_task_fn fn,
                                       void** args, int32_t count) {
    for (int32_t i = 0; i < count; i++) {
        carquet_worker_pool_submit(pool, fn, args[i]);
    }
    carquet_worker_pool_wait(pool);
}

void carquet_worker_pool_destroy(carquet_worker_pool_t* pool) {
    if (!pool) return;

    POOL_LOCK(pool);
    pool->shutdown = true;
    POOL_BROADCAST_WORK(pool);
    POOL_UNLOCK(pool);

#ifdef _WIN32
    for (int32_t i = 0; i < pool->num_threads; i++) {
        WaitForSingleObject(pool->threads[i], INFINITE);
        CloseHandle(pool->threads[i]);
    }
    DeleteCriticalSection(&pool->mutex);
#else
    for (int32_t i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->work_available);
    pthread_cond_destroy(&pool->work_done);
#endif

    free(pool->threads);
    free(pool);
}
