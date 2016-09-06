/* file: service_memory.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
#include "tbb/scalable_allocator.h"

namespace daal
{
namespace services
{

namespace internal
{

template<typename T, CpuType cpu>
T *service_calloc(size_t size, size_t alignment = 64)
{
    T *ptr = (T *)daal::services::daal_malloc(size * sizeof(T), alignment );
    if( ptr == NULL ) { return NULL; }

    char *cptr = (char *)ptr;
    size_t sizeInBytes = size * sizeof(T);

    for (size_t i = 0; i < sizeInBytes; i++)
    {
        cptr[i] = '\0';
    }

    return ptr;
}

template<typename T, CpuType cpu>
T *service_malloc(size_t size, size_t alignment = 64)
{
    T *ptr = (T *)daal::services::daal_malloc(size * sizeof(T), alignment );
    if( ptr == NULL ) { return NULL; }

    return ptr;
}

template<typename T, CpuType cpu>
void service_free(T * ptr)
{
    daal::services::daal_free(ptr);

    return;
}


template<typename T, CpuType cpu>
T *service_scalable_calloc(size_t size, size_t alignment = 64)
{
    T *ptr = (T *)scalable_aligned_malloc(size * sizeof(T), alignment );
    if( ptr == NULL ) { return NULL; }

    char *cptr = (char *)ptr;
    size_t sizeInBytes = size * sizeof(T);

    for (size_t i = 0; i < sizeInBytes; i++)
    {
        cptr[i] = '\0';
    }

    return ptr;
}

template<typename T, CpuType cpu>
T *service_scalable_malloc(size_t size, size_t alignment = 64)
{
    T *ptr = (T *)scalable_aligned_malloc(size * sizeof(T), alignment );
    if( ptr == NULL ) { return NULL; }

    return ptr;
}

template<typename T, CpuType cpu>
void service_scalable_free(T * ptr)
{
    scalable_aligned_free(ptr);

    return;
}


template<typename T, CpuType cpu>
T *service_memset(T *ptr, T value, size_t num)
{
    for( size_t i = 0 ; i < num ; i++ )
    {
        ptr[i] = value;
    }
    return ptr;
}

} // namespace daal::internal

}
} // namespace daal

#endif
