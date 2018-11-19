/* file: service_memory.cpp */
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
//  Implementation of memory service functions
//--
*/

#include "service_memory.h"
#include "service_service.h"

void *daal::services::daal_malloc(size_t size, size_t alignment)
{
    return daal::internal::Service<>::serv_malloc(size, alignment);
}

void daal::services::daal_free(void *ptr)
{
    daal::internal::Service<>::serv_free(ptr);
}

namespace daal
{
namespace services
{
void daal_free_buffers()
{
    daal::internal::Service<>::serv_free_buffers();
}
}
}

void daal::services::daal_memcpy_s(void *dest, size_t destSize, const void *src, size_t srcSize)
{
    size_t copySize = srcSize;
    if(destSize < srcSize) {copySize = destSize;}

    size_t BLOCKSIZE = 200000000; // approx 200MB
    size_t nBlocks = copySize / BLOCKSIZE;
    size_t sizeOfLastBlock = 0;
    if(nBlocks * BLOCKSIZE != copySize) {sizeOfLastBlock = copySize - (nBlocks * BLOCKSIZE);}

    char* dstChar = (char*)dest;
    char* srcChar = (char*)src;
    for(size_t i = 0; i < nBlocks; i++)
    {
        daal::internal::Service<>::serv_memcpy_s(&dstChar[i * BLOCKSIZE], BLOCKSIZE, &srcChar[i * BLOCKSIZE], BLOCKSIZE);
    }
    if(sizeOfLastBlock != 0)
    {
        daal::internal::Service<>::serv_memcpy_s(&dstChar[nBlocks * BLOCKSIZE], sizeOfLastBlock, &srcChar[nBlocks * BLOCKSIZE], sizeOfLastBlock);
    }
}
