/* file: daal_memory.h */
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

DAAL_EXPORT float daal_string_to_float(const char * nptr, char ** endptr);

DAAL_EXPORT double daal_string_to_double(const char * nptr, char ** endptr);
}
} // namespace daal

#endif
