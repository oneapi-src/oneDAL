/* file: sorter.cpp */
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

#include "service/kernel/oneapi/sorter.h"
#include "oneapi/internal/utils.h"
#include "externals/service_ittnotify.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace sort
{
services::String GetIntegerTypeForFPType(const TypeId & vectorTypeId)
{
    if (vectorTypeId == TypeIds::Id::float32)
    {
        return " -D radixIntType=uint ";
    }
    else
    {
        return " -D radixIntType=ulong ";
    }
}

DAAL_ITTNOTIFY_DOMAIN(daal.oneapi.internal.select.select_indexed);

void buildProgram(ClKernelFactoryIface & kernelFactory, const TypeId & vectorTypeId)
{
    services::String fptype_name = getKeyFPType(vectorTypeId);
    // add type from name at the end
    auto radixtype_name = GetIntegerTypeForFPType(vectorTypeId);
    auto build_options  = fptype_name + radixtype_name;
    build_options.add("-cl-std=CL1.2 -D sortedType=int");

    services::String cachekey("__daal_oneapi_internal_sort_radix_sort__");
    cachekey.add(build_options);
    kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), radix_sort_simd, build_options.c_str());
}

void runRadixSortSimd(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, const UniversalBuffer & input,
                      const UniversalBuffer & output, const UniversalBuffer & buffer, uint32_t nVectors, uint32_t vectorSize, uint32_t vectorOffset,
                      services::Status * status)
{
    auto sum_kernel = kernelFactory.getKernel("radix_sort_group");

    const uint32_t maxWorkItemsPerGroup = 32;
    KernelRange localRange(1, maxWorkItemsPerGroup);
    KernelRange globalRange(nVectors, maxWorkItemsPerGroup);

    KernelNDRange range(2);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_PTR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_PTR(status);

    KernelArguments args(5);
    args.set(0, input, AccessModeIds::read);
    args.set(1, output, AccessModeIds::write);
    args.set(2, buffer, AccessModeIds::read);
    args.set(3, vectorSize);
    args.set(4, vectorOffset);

    context.run(range, sum_kernel, args, status);
}

void RadixSort::sort(const UniversalBuffer & input, const UniversalBuffer & output, const UniversalBuffer & buffer, uint32_t nVectors,
                     uint32_t vectorSize, uint32_t vectorOffset, services::Status * status)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(RadixSort.sort);

    auto & context       = oneapi::internal::getDefaultContext();
    auto & kernelFactory = context.getClKernelFactory();

    buildProgram(kernelFactory, input.type());

    runRadixSortSimd(context, kernelFactory, input, output, buffer, nVectors, vectorSize, vectorOffset, status);
}

services::Status RadixSort::sortIndices(UniversalBuffer & values, UniversalBuffer & indices, UniversalBuffer & valuesOut,
                                        UniversalBuffer & indicesOut, int nRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(RadixSort.sortIndices);
    services::Status status;

    auto & context       = oneapi::internal::getDefaultContext();
    auto & kernelFactory = context.getClKernelFactory();

    buildProgram(kernelFactory, values.type());

    const size_t sizeFPtype = values.type() == TypeIds::Id::float32 ? 4 : values.type() == TypeIds::Id::float64 ? 8 : 0;

    const int radixBits      = 4;
    const int subSize        = _preferableSubGroup;
    const int localSize      = _preferableSubGroup;
    const int nLocalHists    = 1024 * localSize < nRows ? 1024 : (nRows / localSize) + !!(nRows % localSize);
    const int nSubgroupHists = nLocalHists * (localSize / subSize);

    auto partialHists       = context.allocate(TypeIds::id<int>(), (nSubgroupHists + 1) << _radixBits, &status);
    auto partialPrefixHists = context.allocate(TypeIds::id<int>(), (nSubgroupHists + 1) << _radixBits, &status);

    DAAL_CHECK_STATUS_VAR(status);

    size_t rev = 0;

    for (size_t bitOffset = 0; bitOffset < 8 * sizeFPtype; bitOffset += radixBits, rev ^= 1)
    {
        if (!rev)
        {
            DAAL_CHECK_STATUS_VAR(radixScan(values, partialHists, nRows, bitOffset, localSize, nLocalHists));
            DAAL_CHECK_STATUS_VAR(radixHistScan(values, partialHists, partialPrefixHists, localSize, nSubgroupHists));
            DAAL_CHECK_STATUS_VAR(radixReorder(values, indices, partialPrefixHists, valuesOut, indicesOut, nRows, bitOffset, localSize, nLocalHists));
        }
        else
        {
            DAAL_CHECK_STATUS_VAR(radixScan(valuesOut, partialHists, nRows, bitOffset, localSize, nLocalHists));
            DAAL_CHECK_STATUS_VAR(radixHistScan(values, partialHists, partialPrefixHists, localSize, nSubgroupHists));
            DAAL_CHECK_STATUS_VAR(radixReorder(valuesOut, indicesOut, partialPrefixHists, values, indices, nRows, bitOffset, localSize, nLocalHists));
        }
    }

    DAAL_ASSERT(rev == 0); // if not, we need to swap values/indices and valuesOut/indices_bufus);
    return status;
}

services::Status RadixSort::radixScan(UniversalBuffer & values, UniversalBuffer & partialHists, int nRows, int bitOffset, int localSize,
                                      int nLocalHists)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(RadixSort.radixScan);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();
    buildProgram(factory, values.type());

    auto kernel = factory.getKernel("radixScan");

    {
        KernelArguments args(4);
        args.set(0, values, AccessModeIds::read);
        args.set(1, partialHists, AccessModeIds::write);
        args.set(2, nRows);
        args.set(3, bitOffset);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalHists);

        KernelNDRange range(1);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

services::Status RadixSort::radixHistScan(UniversalBuffer & values, UniversalBuffer & partialHists, UniversalBuffer & partialPrefixHists,
                                          int localSize, int nSubgroupHists)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(RadixSort.radixHistScan);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();

    buildProgram(factory, values.type());
    auto kernel = factory.getKernel("radixHistScan");

    {
        KernelArguments args(3);
        args.set(0, partialHists, AccessModeIds::read);
        args.set(1, partialPrefixHists, AccessModeIds::write);
        args.set(2, nSubgroupHists);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize);

        KernelNDRange range(1);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

services::Status RadixSort::radixReorder(UniversalBuffer & valuesSrc, UniversalBuffer & indicesSrc, UniversalBuffer & partialPrefixHists,
                                         UniversalBuffer & valuesDst, UniversalBuffer & indicesDst, int nRows, int bitOffset, int localSize,
                                         int nLocalHists)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(RadixSort.radixReorder);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();

    buildProgram(factory, valuesSrc.type());
    auto kernel = factory.getKernel("radixReorder");

    {
        KernelArguments args(7);
        args.set(0, valuesSrc, AccessModeIds::read);
        args.set(1, indicesSrc, AccessModeIds::read);
        args.set(2, partialPrefixHists, AccessModeIds::read);
        args.set(3, valuesDst, AccessModeIds::write);
        args.set(4, indicesDst, AccessModeIds::write);
        args.set(5, nRows);
        args.set(6, bitOffset);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalHists);

        KernelNDRange range(1);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

} // namespace sort
} // namespace internal
} // namespace oneapi
} // namespace daal
