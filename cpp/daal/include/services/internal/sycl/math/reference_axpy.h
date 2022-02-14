/* file: reference_axpy.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Wrappers for BLAS functions.
//--
*/

#ifndef __ONEAPI_INTERNAL_MATH_REFERENCE_AXPY_H__
#define __ONEAPI_INTERNAL_MATH_REFERENCE_AXPY_H__

#include "services/internal/buffer.h"
namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
{
namespace math
{
namespace interface1
{
/** @ingroup oneapi_internal
 * @{
 */

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__REFERENCEAXPY"></a>
 *  \brief Adapter for reference AXPY routine
 */
template <typename algorithmFPType>
struct DAAL_EXPORT ReferenceAxpy
{
    ReferenceAxpy() {}

    Status operator()(const int n, const algorithmFPType a, const Buffer<algorithmFPType> & x_buffer, const int incx,
                      Buffer<algorithmFPType> & y_buffer, const int incy);
};

/** @} */
} // namespace interface1

using interface1::ReferenceAxpy;

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
