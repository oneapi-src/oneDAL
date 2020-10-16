/* file: math_service_types.h */
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

#ifndef __MATH_SERVICE_TYPES_H__
#define __MATH_SERVICE_TYPES_H__

#include "services/internal/sycl/types.h"
#include "services/internal/execution_context.h"
#include "services/internal/sycl/math/types.h"
#include "services/internal/sycl/execution_context.h"
#include "services/internal/buffer.h"
#include "services/error_handling.h"
#include "src/sycl/cl_kernels/math.cl"

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
static algorithmFPType expThreshold()
{
    return IsSameType<algorithmFPType, double>::value ? -650.0 : -75.0f;
}

template <typename algorithmFPType>
static services::Status vLog(const services::internal::Buffer<algorithmFPType> & x, services::internal::Buffer<algorithmFPType> & result,
                             const uint32_t n)
{
    services::Status status;

    services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    const services::String options = services::internal::sycl::getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_math_");
    cachekey.add(options);
    factory.build(services::internal::sycl::ExecutionTargetIds::device, cachekey.c_str(), clKernelMath, options.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName              = "vLog";
    services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    services::internal::sycl::KernelArguments args(2);
    args.set(0, x, services::internal::sycl::AccessModeIds::read);
    args.set(1, result, services::internal::sycl::AccessModeIds::write);

    services::internal::sycl::KernelRange range(n);

    ctx.run(range, kernel, args, status);

    return status;
}

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
