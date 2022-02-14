/* file: spblas_gpu.h */
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
//  Template wrappers for common GPU blas functions.
//--
*/

#ifndef __SERVICE_ONEAPI_SPBLAS_GPU_H__
#define __SERVICE_ONEAPI_SPBLAS_GPU_H__

#include "services/internal/sycl/execution_context.h"
#include "services/internal/sycl/types_utils.h"
#include "src/sycl/math_service_types.h"
#include "services/internal/buffer.h"
#include "services/internal/execution_context.h"
#include "services/internal/sycl/math/types.h"

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
template <typename algorithmFPType>
struct DAAL_EXPORT SpBlasGpu
{
    static services::Status xgemm(const Transpose transa, const Transpose transb, const size_t m, const size_t n, const size_t k,
                                  const algorithmFPType alpha, const services::internal::Buffer<algorithmFPType> & a_buffer,
                                  const services::internal::Buffer<size_t> & ja_buffer, const services::internal::Buffer<size_t> & ia_buffer,
                                  const services::internal::Buffer<algorithmFPType> & b_buffer, const services::internal::Buffer<size_t> & jb_buffer,
                                  const services::internal::Buffer<size_t> & ib_buffer, const algorithmFPType beta,
                                  services::internal::Buffer<algorithmFPType> & c_buffer, const size_t ldc, const size_t offsetC);
};

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
