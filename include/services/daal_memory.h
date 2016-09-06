/* file: daal_memory.h */
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
//  Common definitions.
//--
*/

#ifndef __DAAL_MEMORY_H__
#define __DAAL_MEMORY_H__

#include "services/daal_defines.h"

namespace daal
{
/**
 * \brief Contains classes that implement service functionality, including error handling,
 * memory allocation, and library version information
 */
namespace services
{
/**
 * @ingroup memory
 * @{
 */
/**
 * Allocates an aligned block of memory
 * \param[in] size      Size of the block of memory in bytes
 * \param[in] alignment Alignment constraint. Must be a power of two
 * \return Pointer to the beginning of a newly allocated block of memory
 */
DAAL_EXPORT void *daal_malloc(size_t size, size_t alignment = DAAL_MALLOC_DEFAULT_ALIGNMENT);

/**
 * Deallocates the space previously allocated by daal_malloc
 * \param[in] ptr   Pointer to the beginning of a block of memory to deallocate
 */
DAAL_EXPORT void  daal_free(void *ptr);

/**
 * Copies bytes between buffers
 * \param[out] dest               Pointer to new buffer
 * \param[in]  numberOfElements   Size of new buffer
 * \param[in]  src                Pointer to source buffer
 * \param[in]  count              Number of bytes to copy.
 */
DAAL_EXPORT void  daal_memcpy_s(void *dest, size_t numberOfElements, const void *src, size_t count);
/** @} */

}
} // namespace daal

#endif
