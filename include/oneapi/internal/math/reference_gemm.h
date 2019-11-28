/* file: reference_gemm.h */
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
//  Wrappers for BLAS functions.
//--
*/

#ifndef __ONEAPI_INTERNAL_MATH_REFERENCE_GEMM_H__
#define __ONEAPI_INTERNAL_MATH_REFERENCE_GEMM_H__

#include "types.h"
#include "oneapi/internal/types_utils.h"
#include "services/buffer.h"
#include "services/env_detect.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace math
{
namespace interface1
{
/** @ingroup oneapi_internal
 * @{
 */

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__REFERENCEGEMM"></a>
 *  \brief Adapter for reference GEMM routine
 */
template <typename algorithmFPType>
struct DAAL_EXPORT ReferenceGemm
{
    ReferenceGemm() {}

    services::Status operator()(const math::Transpose transa, const math::Transpose transb, const size_t m, const size_t n, const size_t k,
                                const algorithmFPType alpha, const services::Buffer<algorithmFPType> & a_buffer, const size_t lda,
                                const size_t offsetA, const services::Buffer<algorithmFPType> & b_buffer, const size_t ldb, const size_t offsetB,
                                const algorithmFPType beta, services::Buffer<algorithmFPType> & c_buffer, const size_t ldc, const size_t offsetC);
};

/** @} */
} // namespace interface1

using interface1::ReferenceGemm;

} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
