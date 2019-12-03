/* file: math_service_types.h */
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

#ifndef __MATH_SERVICE_TYPES_H__
#define __MATH_SERVICE_TYPES_H__

#include "oneapi/internal/math/types.h"
#include "oneapi/internal/types.h"
#include "oneapi/internal/execution_context.h"
#include "env_detect.h"
#include "services/buffer.h"
#include "services/error_handling.h"
#include "cl_kernels/math.cl"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace math
{
template <typename algorithmFPType>
static algorithmFPType expThreshold()
{
    return IsSameType<algorithmFPType, double>::value ? -650.0 : -75.0f;
}

template <typename algorithmFPType>
static services::Status vLog(const services::Buffer<algorithmFPType> & x, services::Buffer<algorithmFPType> & result, const uint32_t n)
{
    services::Status status;

    oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    const services::String options = oneapi::internal::getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_math_");
    cachekey.add(options);
    factory.build(oneapi::internal::ExecutionTargetIds::device, cachekey.c_str(), clKernelMath, options.c_str());

    const char * const kernelName      = "vLog";
    oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

    oneapi::internal::KernelArguments args(2);
    args.set(0, x, oneapi::internal::AccessModeIds::read);
    args.set(1, result, oneapi::internal::AccessModeIds::write);

    oneapi::internal::KernelRange range(n);

    ctx.run(range, kernel, args, &status);

    return status;
}

} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
