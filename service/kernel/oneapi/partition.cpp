/* file: partition.cpp */
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

#include "service/kernel/oneapi/partition.h"
#include "services/env_detect.h"
#include "externals/service_ittnotify.h"
#include "service/kernel/oneapi/cl_kernels/partition.cl"

namespace daal
{
namespace oneapi
{
namespace internal
{
DAAL_ITTNOTIFY_DOMAIN(daal.oneapi.internal.partition);

services::Status Partition::buildProgram(ClKernelFactoryIface & factory, const TypeId & vectorTypeId)
{
    services::String fptype_name = getKeyFPType(vectorTypeId);
    auto build_options           = fptype_name;
    services::String cachekey("__daal_oneapi_internal_partition_");
    cachekey.add(build_options);

    services::Status status;
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), kernelsPartition, build_options.c_str(), &status);
    return status;
}

services::Status Partition::scan(ClKernelFactoryIface & factory, UniversalBuffer & mask, UniversalBuffer & partialSums, size_t nElems,
                                 size_t localSize, size_t nLocalSums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(partition.scan);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto kernel    = factory.getKernel("scan");

    KernelArguments args(3);
    args.set(0, mask, AccessModeIds::read);
    args.set(1, partialSums, AccessModeIds::write);
    args.set(2, (int)nElems);

    KernelRange local_range(localSize);
    KernelRange global_range(localSize * nLocalSums);

    KernelNDRange range(1);
    range.global(global_range, &status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(local_range, &status);
    DAAL_CHECK_STATUS_VAR(status);

    context.run(range, kernel, args, &status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

services::Status Partition::scanIndex(ClKernelFactoryIface & factory, UniversalBuffer & mask, UniversalBuffer & data, UniversalBuffer & partialSums,
                                      size_t nElems, size_t localSize, size_t nLocalSums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(partition.scanIndex);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto kernel    = factory.getKernel("scanIndex");

    KernelArguments args(4);
    args.set(0, mask, AccessModeIds::read);
    args.set(1, data, AccessModeIds::read);
    args.set(2, partialSums, AccessModeIds::write);
    args.set(3, (int)nElems);

    KernelRange local_range(localSize);
    KernelRange global_range(localSize * nLocalSums);

    KernelNDRange range(1);
    range.global(global_range, &status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(local_range, &status);
    DAAL_CHECK_STATUS_VAR(status);

    context.run(range, kernel, args, &status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

services::Status Partition::sumScan(ClKernelFactoryIface & factory, UniversalBuffer & partialSums, UniversalBuffer & partialPrefixSums,
                                    UniversalBuffer & totalSum, size_t localSize, size_t nSubgroupSums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(partition.sumScan);
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto kernel    = factory.getKernel("sumScan");

    KernelArguments args(4);
    args.set(0, partialSums, AccessModeIds::read);
    args.set(1, partialPrefixSums, AccessModeIds::write);
    args.set(2, totalSum, AccessModeIds::write);
    args.set(3, (int)nSubgroupSums);

    KernelRange local_range(localSize);
    KernelRange global_range(localSize);

    KernelNDRange range(1);
    range.global(global_range, &status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(local_range, &status);
    DAAL_CHECK_STATUS_VAR(status);

    context.run(range, kernel, args, &status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

services::Status Partition::reorder(ClKernelFactoryIface & factory, UniversalBuffer & mask, UniversalBuffer & data, UniversalBuffer & outData,
                                    UniversalBuffer & partialPrefixSums, size_t nElems, size_t localSize, size_t nLocalSums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(partition.reorder);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto kernel    = factory.getKernel("reorder");

    KernelArguments args(5);
    args.set(0, mask, AccessModeIds::read);
    args.set(1, data, AccessModeIds::read);
    args.set(2, outData, AccessModeIds::write);
    args.set(3, partialPrefixSums, AccessModeIds::read);
    args.set(4, (int)nElems);

    KernelRange local_range(localSize);
    KernelRange global_range(localSize * nLocalSums);

    KernelNDRange range(1);
    range.global(global_range, &status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(local_range, &status);
    DAAL_CHECK_STATUS_VAR(status);

    context.run(range, kernel, args, &status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

services::Status Partition::reorderIndex(ClKernelFactoryIface & factory, UniversalBuffer & mask, UniversalBuffer & data, UniversalBuffer & outData,
                                         UniversalBuffer & partialPrefixSums, size_t nElems, size_t localSize, size_t nLocalSums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(partition.reorderIndex);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto kernel    = factory.getKernel("reorderIndex");

    KernelArguments args(5);
    args.set(0, mask, AccessModeIds::read);
    args.set(1, data, AccessModeIds::read);
    args.set(2, outData, AccessModeIds::write);
    args.set(3, partialPrefixSums, AccessModeIds::read);
    args.set(4, (int)nElems);

    KernelRange local_range(localSize);
    KernelRange global_range(localSize * nLocalSums);

    KernelNDRange range(1);
    range.global(global_range, &status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(local_range, &status);
    DAAL_CHECK_STATUS_VAR(status);

    context.run(range, kernel, args, &status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

services::Status Partition::flagged(UniversalBuffer mask, UniversalBuffer data, UniversalBuffer outData, const size_t nElems, size_t & nSelect)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(flagged);

    services::Status status;
    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();

    buildProgram(factory, data.type());

    const uint32_t subSize       = _preferableSubGroup;
    const uint32_t localSize     = _preferableSubGroup;
    const uint32_t nLocalSums    = _maxLocalSums * localSize < nElems ? _maxLocalSums : (nElems / localSize) + !!(nElems % localSize);
    const uint32_t nSubgroupSums = nLocalSums * (localSize / subSize);

    auto partialSums       = context.allocate(TypeIds::id<int>(), nSubgroupSums + 1, &status);
    auto partialPrefixSums = context.allocate(TypeIds::id<int>(), nSubgroupSums + 1, &status);
    auto totalSum          = context.allocate(TypeIds::id<int>(), 1, &status);

    DAAL_CHECK_STATUS_VAR(scan(factory, mask, partialSums, nElems, localSize, nLocalSums));
    DAAL_CHECK_STATUS_VAR(sumScan(factory, partialSums, partialPrefixSums, totalSum, localSize, nSubgroupSums));
    DAAL_CHECK_STATUS_VAR(reorder(factory, mask, data, outData, partialPrefixSums, nElems, localSize, nLocalSums));

    DAAL_CHECK_STATUS_VAR(status);

    {
        auto totalSumHost = totalSum.template get<int>().toHost(data_management::ReadWriteMode::readOnly, &status);
        nSelect           = totalSumHost.get()[0];
    }

    return status;
}

services::Status Partition::flaggedIndex(UniversalBuffer mask, UniversalBuffer data, UniversalBuffer outData, const size_t nElems, size_t & nSelect)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(flaggedIndex);

    services::Status status;
    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();

    buildProgram(factory, data.type());

    const uint32_t subSize       = _preferableSubGroup;
    const uint32_t localSize     = _preferableSubGroup;
    const uint32_t nLocalSums    = _maxLocalSums * localSize < nElems ? _maxLocalSums : (nElems / localSize) + !!(nElems % localSize);
    const uint32_t nSubgroupSums = nLocalSums * (localSize / subSize);

    auto partialSums       = context.allocate(TypeIds::id<int>(), nSubgroupSums + 1, &status);
    auto partialPrefixSums = context.allocate(TypeIds::id<int>(), nSubgroupSums + 1, &status);
    auto totalSum          = context.allocate(TypeIds::id<int>(), 1, &status);

    DAAL_CHECK_STATUS_VAR(scanIndex(factory, mask, data, partialSums, nElems, localSize, nLocalSums));
    DAAL_CHECK_STATUS_VAR(sumScan(factory, partialSums, partialPrefixSums, totalSum, localSize, nSubgroupSums));
    DAAL_CHECK_STATUS_VAR(reorderIndex(factory, mask, data, outData, partialPrefixSums, nElems, localSize, nLocalSums));

    DAAL_CHECK_STATUS_VAR(status);

    {
        auto totalSumHost = totalSum.template get<int>().toHost(data_management::ReadWriteMode::readOnly, &status);
        nSelect           = totalSumHost.get()[0];
    }

    return status;
}

} // namespace internal
} // namespace oneapi
} // namespace daal
