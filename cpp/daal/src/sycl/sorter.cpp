/* file: sorter.cpp */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

#include "src/sycl/sorter.h"
#include "services/internal/execution_context.h"
#include "src/externals/service_profiler.h"

namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
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

services::Status buildProgram(ClKernelFactoryIface & kernelFactory, const TypeId & vectorTypeId)
{
    services::Status status;

    services::String fptype_name = getKeyFPType(vectorTypeId);
    // add type from name at the end
    auto radixtype_name = GetIntegerTypeForFPType(vectorTypeId);
    auto build_options  = fptype_name + radixtype_name;
    build_options.add("-cl-std=CL1.2 -D sortedType=int");

    services::String cachekey("__daal_oneapi_internal_sort_radix_sort__");
    cachekey.add(build_options);
    kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), radix_sort_simd, build_options.c_str(), status);
    return status;
}

static services::Status runRadixSortSimd(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, const UniversalBuffer & input,
                                         const UniversalBuffer & output, const UniversalBuffer & buffer, uint32_t nVectors, uint32_t vectorSize,
                                         uint32_t vectorOffset)
{
    services::Status status;
    auto sum_kernel = kernelFactory.getKernel("radix_sort_group", status);
    DAAL_CHECK_STATUS_VAR(status);

    const uint32_t maxWorkItemsPerGroup = 32;
    KernelRange localRange(1, maxWorkItemsPerGroup);
    KernelRange globalRange(nVectors, maxWorkItemsPerGroup);

    KernelNDRange range(2);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_ASSERT_UNIVERSAL_BUFFER(input, int, nVectors);
    DAAL_ASSERT_UNIVERSAL_BUFFER(output, int, nVectors);
    DAAL_ASSERT_UNIVERSAL_BUFFER(buffer, int, nVectors);

    KernelArguments args(5, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, input, AccessModeIds::read);
    args.set(1, output, AccessModeIds::write);
    args.set(2, buffer, AccessModeIds::read);
    args.set(3, vectorSize);
    args.set(4, vectorOffset);

    context.run(range, sum_kernel, args, status);

    return status;
}

services::Status RadixSort::sort(const UniversalBuffer & input, const UniversalBuffer & output, const UniversalBuffer & buffer, uint32_t nVectors,
                                 uint32_t vectorSize, uint32_t vectorOffset)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(RadixSort.sort);

    auto & context       = services::internal::getDefaultContext();
    auto & kernelFactory = context.getClKernelFactory();

    services::Status status = buildProgram(kernelFactory, input.type());
    DAAL_CHECK_STATUS_VAR(status);

    status |= runRadixSortSimd(context, kernelFactory, input, output, buffer, nVectors, vectorSize, vectorOffset);
    return status;
}

services::Status RadixSort::sortIndices(UniversalBuffer & values, UniversalBuffer & indices, UniversalBuffer & valuesOut,
                                        UniversalBuffer & indicesOut, uint32_t nRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(RadixSort.sortIndices);
    services::Status status;

    auto & context       = services::internal::getDefaultContext();
    auto & kernelFactory = context.getClKernelFactory();

    DAAL_CHECK_STATUS_VAR(buildProgram(kernelFactory, values.type()));

    const uint32_t sizeFPtype = values.type() == TypeIds::Id::float32 ? 4 : values.type() == TypeIds::Id::float64 ? 8 : 0;

    const uint32_t radixBits      = 4;
    const uint32_t subSize        = _preferableSubGroup;
    const uint32_t localSize      = _preferableSubGroup;
    const uint32_t nLocalHists    = 1024 * localSize < nRows ? 1024 : (nRows / localSize) + !!(nRows % localSize);
    const uint32_t nSubgroupHists = nLocalHists * (localSize / subSize);

    auto partialHists = context.allocate(TypeIds::id<int>(), (nSubgroupHists + 1) << _radixBits, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto partialPrefixHists = context.allocate(TypeIds::id<int>(), (nSubgroupHists + 1) << _radixBits, status);
    DAAL_CHECK_STATUS_VAR(status);

    uint32_t rev = 0;

    for (uint32_t bitOffset = 0; bitOffset < 8 * sizeFPtype; bitOffset += radixBits, rev ^= 1)
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

    DAAL_ASSERT(rev == 0); // if not, we need to swap values/indices and
                           // valuesOut/indices_bufus);
    return status;
}

services::Status RadixSort::radixScan(UniversalBuffer & values, UniversalBuffer & partialHists, uint32_t nRows, uint32_t bitOffset,
                                      uint32_t localSize, uint32_t nLocalHists)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(RadixSort.radixScan);

    services::Status status;

    auto & context = services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(factory, values.type()));

    auto kernel = factory.getKernel("radixScan", status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialHists, int, (nLocalHists + 1) << _radixBits);

        KernelArguments args(4, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, values, AccessModeIds::read);
        args.set(1, partialHists, AccessModeIds::write);
        args.set(2, nRows);
        args.set(3, bitOffset);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalHists);

        KernelNDRange range(1);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

services::Status RadixSort::radixHistScan(UniversalBuffer & values, UniversalBuffer & partialHists, UniversalBuffer & partialPrefixHists,
                                          uint32_t localSize, uint32_t nSubgroupHists)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(RadixSort.radixHistScan);

    services::Status status;

    auto & context = services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();

    DAAL_CHECK_STATUS_VAR(buildProgram(factory, values.type()));
    auto kernel = factory.getKernel("radixHistScan", status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialHists, int, (nSubgroupHists + 1) << _radixBits);
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialPrefixHists, int, (nSubgroupHists + 1) << _radixBits);

        KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, partialHists, AccessModeIds::read);
        args.set(1, partialPrefixHists, AccessModeIds::write);
        args.set(2, nSubgroupHists);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize);

        KernelNDRange range(1);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

services::Status RadixSort::radixReorder(UniversalBuffer & valuesSrc, UniversalBuffer & indicesSrc, UniversalBuffer & partialPrefixHists,
                                         UniversalBuffer & valuesDst, UniversalBuffer & indicesDst, uint32_t nRows, uint32_t bitOffset,
                                         uint32_t localSize, uint32_t nLocalHists)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(RadixSort.radixReorder);

    services::Status status;

    auto & context = services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();

    DAAL_CHECK_STATUS_VAR(buildProgram(factory, valuesSrc.type()));
    auto kernel = factory.getKernel("radixReorder", status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER2(indicesSrc, int, uint32_t, nRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialPrefixHists, int, (nLocalHists + 1) << _radixBits);
        DAAL_ASSERT_UNIVERSAL_BUFFER2(indicesDst, int, uint32_t, nRows);

        KernelArguments args(7, status);
        DAAL_CHECK_STATUS_VAR(status);
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
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

} // namespace sort
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
