/* file: fill_buffer_helper.cpp */
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

#include "src/sycl/cl_kernels/fill_buffer_helper.cl"
#include "sycl/internal/types_utils.h"
#include "sycl/internal/execution_context.h"
#include "sycl/internal/utils.h"
#include "sycl/internal/types.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
template <typename algorithmType>
static void buildProgram(ClKernelFactoryIface & kernelFactory)
{
    services::String cachekey("__daal_oneapi_internal_fill_buffer_helper");
    services::String buildOptions = oneapi::internal::getKeyFPType<algorithmType>();
    cachekey.add(buildOptions);
    kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), clFillBufferHelper, buildOptions.c_str());
}

template <typename algorithmType>
services::Status fillBuffer(services::Buffer<algorithmType> & buf, size_t nElems, algorithmType val)
{
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram<algorithmType>(factory);

    const char * const kernelName = "fillBuffer";
    KernelPtr kernel              = factory.getKernel(kernelName);

    {
        KernelArguments args(2);
        args.set(0, buf, AccessModeIds::write);
        args.set(1, val);

        KernelRange global_range(nElems);

        ctx.run(global_range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template DAAL_EXPORT services::Status fillBuffer<float>(services::Buffer<float> & buf, size_t nElems, float val);
template DAAL_EXPORT services::Status fillBuffer<double>(services::Buffer<double> & buf, size_t nElems, double val);
template DAAL_EXPORT services::Status fillBuffer<int8_t>(services::Buffer<int8_t> & buf, size_t nElems, int8_t val);
template DAAL_EXPORT services::Status fillBuffer<int16_t>(services::Buffer<int16_t> & buf, size_t nElems, int16_t val);
template DAAL_EXPORT services::Status fillBuffer<int32_t>(services::Buffer<int32_t> & buf, size_t nElems, int32_t val);
template DAAL_EXPORT services::Status fillBuffer<int64_t>(services::Buffer<int64_t> & buf, size_t nElems, int64_t val);
template DAAL_EXPORT services::Status fillBuffer<uint8_t>(services::Buffer<uint8_t> & buf, size_t nElems, uint8_t val);
template DAAL_EXPORT services::Status fillBuffer<uint16_t>(services::Buffer<uint16_t> & buf, size_t nElems, uint16_t val);
template DAAL_EXPORT services::Status fillBuffer<uint32_t>(services::Buffer<uint32_t> & buf, size_t nElems, uint32_t val);
template DAAL_EXPORT services::Status fillBuffer<uint64_t>(services::Buffer<uint64_t> & buf, size_t nElems, uint64_t val);

} // namespace internal
} // namespace oneapi
} // namespace daal
