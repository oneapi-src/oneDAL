/* file: lapack_gpu.h */
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
//  Template wrappers for common GPU lapack functions.
//--
*/

#ifndef __SERVICE_ONEAPI_LAPACK_GPU_H__
#define __SERVICE_ONEAPI_LAPACK_GPU_H__

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
struct LapackGpu
{
    static services::Status xpotrf(const math::UpLo uplo, const uint32_t n, UniversalBuffer a_buffer, const uint32_t lda)
    {
        services::Status status;

        ExecutionContextIface & ctx = services::Environment::getInstance()->getDefaultExecutionContext();

        ctx.potrf(uplo, n, a_buffer, lda, &status);

        return status;
    }

    static services::Status xpotrs(const math::UpLo uplo, const uint32_t n, const uint32_t ny, UniversalBuffer a_buffer, const uint32_t lda,
                                   UniversalBuffer b_buffer, const uint32_t ldb)
    {
        services::Status status;

        ExecutionContextIface & ctx = services::Environment::getInstance()->getDefaultExecutionContext();

        ctx.potrs(uplo, n, ny, a_buffer, lda, b_buffer, ldb, &status);

        return status;
    }
};

} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
