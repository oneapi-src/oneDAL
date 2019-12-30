/* file: blas_gpu.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Template wrappers for common GPU blas functions.
//--
*/

#ifndef __SERVICE_ONEAPI_BLAS_GPU_H__
#define __SERVICE_ONEAPI_BLAS_GPU_H__

#include "services/env_detect.h"
#include "oneapi/internal/execution_context.h"
#include "oneapi/internal/types_utils.h"
#include "oneapi/math_service_types.h"
#include "services/buffer.h"
#include "oneapi/internal/math/types.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
template <typename algorithmFPType>
struct BlasGpu
{
    static services::Status xgemm(const math::Layout layout, const math::Transpose transa, const math::Transpose transb, const uint32_t m,
                                  const uint32_t n, const uint32_t k, const algorithmFPType alpha, const UniversalBuffer a_buffer, const uint32_t lda,
                                  const uint32_t offsetA, const UniversalBuffer b_buffer, const uint32_t ldb, const uint32_t offsetB,
                                  const algorithmFPType beta, UniversalBuffer c_buffer, const uint32_t ldc, const uint32_t offsetC)
    {
        services::Status status;

        ExecutionContextIface & ctx = services::Environment::getInstance()->getDefaultExecutionContext();

        if (layout == math::Layout::ColMajor)
        {
            ctx.gemm(transa, transb, m, n, k, alpha, a_buffer, lda, offsetA, b_buffer, ldb, offsetB, beta, c_buffer, ldc, offsetC, &status);
        }
        else
        {
            ctx.gemm(transb, transa, n, m, k, alpha, b_buffer, ldb, offsetB, a_buffer, lda, offsetA, beta, c_buffer, ldc, offsetC, &status);
        }

        return status;
    }

    static services::Status xsyrk(const math::Layout layout, const math::UpLo upper_lower, const math::Transpose trans, const uint32_t n,
                                  const uint32_t k, const algorithmFPType alpha, const UniversalBuffer a_buffer, const uint32_t lda,
                                  const uint32_t offsetA, const algorithmFPType beta, UniversalBuffer c_buffer, const uint32_t ldc,
                                  const uint32_t offsetC)
    {
        services::Status status;

        ExecutionContextIface & ctx = services::Environment::getInstance()->getDefaultExecutionContext();

        if (layout == math::Layout::ColMajor)
        {
            ctx.syrk(upper_lower, trans, n, k, alpha, a_buffer, lda, offsetA, beta, c_buffer, ldc, offsetC, &status);
        }
        else
        {
            ctx.syrk(upper_lower, trans, k, n, alpha, a_buffer, lda, offsetA, beta, c_buffer, ldc, offsetC, &status);
        }

        return status;
    }

    static services::Status xaxpy(const uint32_t n, const double a, const UniversalBuffer x_buffer, const int incx,
                                 const UniversalBuffer y_buffer, const int incy)
    {
        services::Status status;

        ExecutionContextIface & ctx = services::Environment::getInstance()->getDefaultExecutionContext();

        ctx.axpy(n, a, x_buffer, incx, y_buffer, incy, &status);

        return status;
    }
};

} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
