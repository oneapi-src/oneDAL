/* file: sorter.cpp */
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

#include "oneapi/sorter.h"
#include "oneapi/internal/utils.h"
#include "service_ittnotify.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace sort
{

DAAL_ITTNOTIFY_DOMAIN(daal.oneapi.internal.select.select_indexed);

void buildProgram(ClKernelFactoryIface& kernelFactory,
                  const TypeId& vectorTypeId)
{
    services::String fptype_name = getKeyFPType(vectorTypeId);
    auto build_options = fptype_name;
    // add type from name at the end
    build_options.add("-cl-std=CL1.2 -D sortedType=int");

    services::String cachekey("__daal_oneapi_internal_sort_radix_sort__");
    cachekey.add(fptype_name);
    kernelFactory.build(ExecutionTargetIds::device,
                        cachekey.c_str(),
                        radix_sort_simd,
                        build_options.c_str());
}


void run_radix_sort_simd( ExecutionContextIface& context,
                            ClKernelFactoryIface& kernelFactory,
                            const UniversalBuffer& input,
                            const UniversalBuffer& output,
                            const UniversalBuffer& buffer,
                            uint32_t nVectors,
                            uint32_t vectorSize,
                            uint32_t vectorOffset,
                            services::Status* status)
{
    auto sum_kernel = kernelFactory.getKernel("radix_sort_group");

    const uint32_t maxWorkItemsPerGroup = 32;
    KernelRange localRange(1, maxWorkItemsPerGroup);
    KernelRange globalRange(nVectors, maxWorkItemsPerGroup);

    KernelNDRange range(2);
    range.global(globalRange, status); DAAL_CHECK_STATUS_PTR(status);
    range.local(localRange, status);   DAAL_CHECK_STATUS_PTR(status);

    KernelArguments args(5);
    args.set(0, input, AccessModeIds::read);
    args.set(1, output, AccessModeIds::write);
    args.set(2, buffer, AccessModeIds::read);
    args.set(3, vectorSize);
    args.set(4, vectorOffset);

    context.run(range, sum_kernel, args, status);
}

void RadixSort::sort( const UniversalBuffer& input, const UniversalBuffer& output,  const UniversalBuffer& buffer,
                                    uint32_t nVectors, uint32_t vectorSize, uint32_t vectorOffset, services::Status* status)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(RadixSort.sort);

    auto& context = oneapi::internal::getDefaultContext();
    auto& kernelFactory = context.getClKernelFactory();

    buildProgram(kernelFactory, input.type());

    run_radix_sort_simd(context, kernelFactory,
                       input, output, buffer, nVectors, vectorSize, vectorOffset, status);

}

} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal
