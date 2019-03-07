/* file: service_environment.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

#ifndef __SERVICE_ENVIRONMENT_H__
#define __SERVICE_ENVIRONMENT_H__

namespace daal
{
namespace services
{
namespace internal
{

//returns size of L1 cache in bytes
unsigned getL1CacheSize();
//returns size of LL (last level) cache in bytes
unsigned getLLCacheSize();

//returns number of elements that fit into the memory of given size
//param sizeofAnElement - size of an element in bytes
//param defaultNumElements - return this number of element if sizeofMemory is 0
unsigned getNumElementsFitInMemory(size_t sizeofMemory, size_t sizeofAnElement, size_t defaultNumElements);

//returns number of elements that fit into L1 cache
//param sizeofAnElement - size of an element in bytes
//param defaultNumElements - return this number of elements if cache size system method call failed
unsigned getNumElementsFitInL1Cache(size_t sizeofAnElement, size_t defaultNumElements);

//returns number of elements that fit into LL cache
//param sizeofAnElement - size of an element in bytes
//param defaultNumElements - return this number of element if cache size system method call failed
unsigned getNumElementsFitInLLCache(size_t sizeofAnElement, size_t defaultNumElements);

}
}
}

#endif /* __SERVICE_ENVIRONMENT_H__ */
