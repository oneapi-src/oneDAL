/* file: service_memory.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
    T *ptr = (T *)threaded_scalable_malloc(size * sizeof(T), alignment );

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
    T *ptr = (T *)threaded_scalable_malloc(size * sizeof(T), alignment );
    if( ptr == NULL ) { return NULL; }
    return ptr;
}

template<typename T, CpuType cpu>
void service_scalable_free(T * ptr)
{
    threaded_scalable_free(ptr);
    return;
}


template<typename T, CpuType cpu>
T* service_memset(T * const ptr, const T value, const size_t num)
{
    const size_t blockSize = 512;
    size_t nBlocks = num / blockSize;
    if (nBlocks * blockSize < num) { nBlocks++; }

    threader_for(nBlocks, nBlocks, [&](size_t block)
    {
        size_t end = (block + 1) * blockSize;
        if (end > num) { end = num; }

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for( size_t i = block * blockSize; i < end ; i++ )
        {
            ptr[i] = value;
        }
    } );
    return ptr;
}


template<typename T, CpuType cpu>
void service_memset_seq(T * const ptr, const T value, const size_t num)
{
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for( size_t i = 0; i < num ; i++)
    {
        ptr[i] = value;
    }
}


} // namespace internal
} // namespace services
} // namespace daal

#endif
