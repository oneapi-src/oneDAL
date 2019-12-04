/* file: select_indexed.cpp */
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

#include "oneapi/select_indexed.h"
#include "oneapi/internal/utils.h"
#include "service_ittnotify.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace selection
{

DAAL_ITTNOTIFY_DOMAIN(daal.oneapi.internal.select.select_indexed);

void buildProgram(ClKernelFactoryIface& kernelFactory,
                  const TypeId& vectorTypeId)
{
    services::String fptype_name = getKeyFPType(vectorTypeId);
    auto build_options = fptype_name;
    build_options.add("-cl-std=CL1.2 ");

    services::String cachekey("__daal_oneapi_internal_select_indexed_");
    cachekey.add(fptype_name);
    kernelFactory.build(ExecutionTargetIds::device,
                        cachekey.c_str(),
                        quick_select_simd,
                        build_options.c_str());
}


void run_quick_select_simd( ExecutionContextIface& context,
                            ClKernelFactoryIface& kernelFactory,
                            const UniversalBuffer& dataVectors,
                            const UniversalBuffer& indexVectors,
                            const UniversalBuffer& rndSeq,
                            uint32_t nRndSeq,
                            uint32_t K,
                            uint32_t nVectors,
                            uint32_t vectorSize,
                            uint32_t vectorOffset,
                            QuickSelectIndexed::Result& result,
                            services::Status* status)
{
    auto sum_kernel = kernelFactory.getKernel("quick_select_group");

    const uint32_t maxWorkItemsPerGroup = 16;
    KernelRange localRange(1, maxWorkItemsPerGroup);
    KernelRange globalRange(nVectors, maxWorkItemsPerGroup);

    KernelNDRange range(2);
    range.global(globalRange, status); DAAL_CHECK_STATUS_PTR(status);
    range.local(localRange, status);   DAAL_CHECK_STATUS_PTR(status);

    KernelArguments args(9);
    args.set(0, dataVectors, AccessModeIds::read);
    args.set(1, indexVectors, AccessModeIds::read);
    args.set(2, result.values, AccessModeIds::write);
    args.set(3, result.indices, AccessModeIds::write);
    args.set(4, rndSeq, AccessModeIds::read);
    args.set(5, nRndSeq);
    args.set(6, vectorSize);
    args.set(7, K);
    args.set(8, vectorOffset);

    context.run(range, sum_kernel, args, status);
}

QuickSelectIndexed::Result QuickSelectIndexed::select(const UniversalBuffer& dataVectors, const UniversalBuffer& indexVectors,
                        const UniversalBuffer& rndSeq, uint32_t nRndSeq,
                        uint32_t K, uint32_t nVectors, uint32_t vectorSize,
                        uint32_t vectorOffset, services::Status* status)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(QuickSelectIndexed.select);

    auto& context = oneapi::internal::getDefaultContext();
    auto& kernelFactory = context.getClKernelFactory();

    Result result(context, K, nVectors, dataVectors.type(), indexVectors.type(), status);

    buildProgram(kernelFactory, dataVectors.type());
    run_quick_select_simd(context, kernelFactory,
                       dataVectors, indexVectors, rndSeq,
                       nRndSeq, K, nVectors, vectorSize,
                       vectorOffset, result, status);
    return result;
}


QuickSelectIndexed::Result& QuickSelectIndexed::select(const UniversalBuffer& dataVectors, const UniversalBuffer& indexVectors,
                        const UniversalBuffer& rndSeq, uint32_t nRndSeq,
                        uint32_t K, uint32_t nVectors, uint32_t vectorSize,
                        uint32_t vectorOffset, Result& result, services::Status* status)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(QuickSelectIndexed.select);

    auto& context = oneapi::internal::getDefaultContext();
    auto& kernelFactory = context.getClKernelFactory();

    buildProgram(kernelFactory, dataVectors.type());

    run_quick_select_simd(context, kernelFactory,
                       dataVectors, indexVectors, rndSeq,
                       nRndSeq, K, nVectors, vectorSize,
                       vectorOffset, result, status);

    return result;
}

} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal
