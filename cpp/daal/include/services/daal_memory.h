/* file: daal_memory.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
DAAL_EXPORT void * daal_malloc(size_t size, size_t alignment = DAAL_MALLOC_DEFAULT_ALIGNMENT);

/**
 * Allocates and initializes with zero an aligned block of memory
 * \param[in] size      Size of the block of memory in bytes
 * \param[in] alignment Alignment constraint. Must be a power of two
 * \return Pointer to the beginning of a newly allocated zero-filled block of memory
 */
DAAL_EXPORT void * daal_calloc(size_t size, size_t alignment = DAAL_MALLOC_DEFAULT_ALIGNMENT);

/**
 * Deallocates the space previously allocated by daal_malloc
 * \param[in] ptr   Pointer to the beginning of a block of memory to deallocate
 */
DAAL_EXPORT void daal_free(void * ptr);

/**
 * Copies bytes between buffers
 * \param[out] dest               Pointer to new buffer
 * \param[in]  numberOfElements   Size of new buffer
 * \param[in]  src                Pointer to source buffer
 * \param[in]  count              Number of bytes to copy.
 * \DAAL_DEPRECATED
 */
DAAL_DEPRECATED DAAL_EXPORT void daal_memcpy_s(void * dest, size_t numberOfElements, const void * src, size_t count);

namespace internal
{
/**
* Saved version of bytes copy between buffers
* \param[out] dest               Pointer to new buffer
* \param[in]  destSize           Size of new buffer
* \param[in]  src                Pointer to source buffer
* \param[in]  srcSize            Number of bytes to copy.
* \return Status of memory copy, memory copy is successful if zero is returned
*/
DAAL_EXPORT int daal_memcpy_s(void * dest, size_t destSize, const void * src, size_t srcSize);
} // namespace internal

/**
 * Copies smax bytes from the region pointed to by src into the region pointed to by dest
 * \param[out] dest               Pointer that will be replaced by src
 * \param[in]  destSize           Size of the resulting dest
 * \param[in]  src                Pointer to source buffer
 * \param[in]  count              Number of bytes to copy.
 */
DAAL_EXPORT void daal_memmove_s(void * dest, size_t destSize, const void * src, size_t count);
/** @} */

DAAL_EXPORT float daal_string_to_float(const char * nptr, char ** endptr);

DAAL_EXPORT double daal_string_to_double(const char * nptr, char ** endptr);

DAAL_EXPORT int daal_string_to_int(const char * nptr, char ** endptr);

DAAL_EXPORT int daal_int_to_string(char * buffer, size_t n, int value);

DAAL_EXPORT int daal_double_to_string(char * buffer, size_t n, double value);
} // namespace services
} // namespace daal

#endif
