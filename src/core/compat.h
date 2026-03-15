#ifndef CARQUET_CORE_COMPAT_H
#define CARQUET_CORE_COMPAT_H

#include <stdlib.h>
#include <string.h>

static inline char* carquet_heap_strdup(const char* str) {
    if (!str) {
        return NULL;
    }

    size_t len = strlen(str) + 1;
    char* copy = (char*)malloc(len);
    if (!copy) {
        return NULL;
    }

    memcpy(copy, str, len);
    return copy;
}

#endif /* CARQUET_CORE_COMPAT_H */
