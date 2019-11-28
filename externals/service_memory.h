/* file: service_memory.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Declaration of memory service functions
//--
*/

#ifndef __SERVICE_MEMORY_H__
#define __SERVICE_MEMORY_H__

#include "daal_defines.h"
#include "daal_memory.h"
#include "service_defines.h"
#include "threading.h"

namespace daal
{
namespace services
{
namespace internal
{
template <typename T, CpuType cpu>
T * service_calloc(size_t size, size_t alignment = 64)
{
    T * ptr = (T *)daal::services::daal_malloc(size * sizeof(T), alignment);
    if (ptr == NULL)
    {
        return NULL;
    }

    char * cptr        = (char *)ptr;
    size_t sizeInBytes = size * sizeof(T);

    for (size_t i = 0; i < sizeInBytes; i++)
    {
        cptr[i] = '\0';
    }

    return ptr;
}

template <typename T, CpuType cpu>
T * service_malloc(size_t size, size_t alignment = 64)
{
    T * ptr = (T *)daal::services::daal_malloc(size * sizeof(T), alignment);
    if (ptr == NULL)
    {
        return NULL;
    }
    return ptr;
}

template <typename T, CpuType cpu>
void service_free(T * ptr)
{
    daal::services::daal_free(ptr);
    return;
}

template <typename T, CpuType cpu>
T * service_scalable_calloc(size_t size, size_t alignment = 64)
{
    T * ptr = (T *)threaded_scalable_malloc(size * sizeof(T), alignment);

    if (ptr == NULL)
    {
        return NULL;
    }

    char * cptr        = (char *)ptr;
    size_t sizeInBytes = size * sizeof(T);

    for (size_t i = 0; i < sizeInBytes; i++)
    {
        cptr[i] = '\0';
    }

    return ptr;
}

template <typename T, CpuType cpu>
T * service_scalable_malloc(size_t size, size_t alignment = 64)
{
    T * ptr = (T *)threaded_scalable_malloc(size * sizeof(T), alignment);
    if (ptr == NULL)
    {
        return NULL;
    }
    return ptr;
}

template <typename T, CpuType cpu>
void service_scalable_free(T * ptr)
{
    threaded_scalable_free(ptr);
    return;
}

template <typename T, CpuType cpu>
T * service_memset(T * const ptr, const T value, const size_t num)
{
    const size_t blockSize = 512;
    size_t nBlocks         = num / blockSize;
    if (nBlocks * blockSize < num)
    {
        nBlocks++;
    }

    threader_for(nBlocks, nBlocks, [&](size_t block) {
        size_t end = (block + 1) * blockSize;
        if (end > num)
        {
            end = num;
        }

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = block * blockSize; i < end; i++)
        {
            ptr[i] = value;
        }
    });
    return ptr;
}

template <typename T, CpuType cpu>
void service_memset_seq(T * const ptr, const T value, const size_t num)
{
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < num; i++)
    {
        ptr[i] = value;
    }
}

} // namespace internal
} // namespace services
} // namespace daal

#endif
