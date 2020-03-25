/* file: dbscan_dense_default_batch_ucapi_impl.i */
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

/*
//++
//  Implementation of default method for DBSCAN algorithm on GPU.
//--
*/

#include "algorithms/dbscan/dbscan_types.h"
#include "algorithms/kernel/dbscan/oneapi/dbscan_kernel_ucapi.h"
#include "algorithms/kernel/dbscan/oneapi/cl_kernels/dbscan_cl_kernels.cl"
#include "externals/service_ittnotify.h"

using namespace daal::services;
using namespace daal::oneapi::internal;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{
struct QueueBlock
{
    QueueBlock(uint32_t queueBegin, uint32_t queueEnd, uint32_t maxQueueBlockSize)
    {
        count = queueEnd - queueBegin;
        if (count > maxQueueBlockSize)
        {
            count = maxQueueBlockSize;
        }
    }
    uint32_t count;
};

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::processResultsToCompute(DAAL_UINT64 resultsToCompute, int * const isCore, NumericTable * ntData,
                                                                        NumericTable * ntCoreIndices, NumericTable * ntCoreObservations)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.processResultsToCompute);
    if (!isCore)
    {
        return Status(ErrorNullPtr);
    }

    auto & context         = Environment::getInstance()->getDefaultExecutionContext();
    const size_t nRows     = ntData->getNumberOfRows();
    const size_t nFeatures = ntData->getNumberOfColumns();

    size_t nCoreObservations = 0;

    for (size_t i = 0; i < nRows; i++)
    {
        if (!isCore[i])
        {
            continue;
        }
        nCoreObservations++;
    }

    if (nCoreObservations == 0)
    {
        return Status();
    }

    if (resultsToCompute & computeCoreIndices)
    {
        DAAL_CHECK_STATUS_VAR(ntCoreIndices->resize(nCoreObservations));
        BlockDescriptor<int> indexRows;
        DAAL_CHECK_STATUS_VAR(ntCoreIndices->getBlockOfRows(0, nCoreObservations, writeOnly, indexRows));
        auto coreIndices = indexRows.getBuffer().toHost(ReadWriteMode::writeOnly);
        if (!coreIndices.get())
        {
            return Status(ErrorNullPtr);
        }

        size_t pos = 0;
        for (size_t i = 0; i < nRows; i++)
        {
            if (!isCore[i])
            {
                continue;
            }
            coreIndices.get()[pos] = i;
            pos++;
        }
    }

    if (resultsToCompute & computeCoreObservations)
    {
        DAAL_CHECK_STATUS_VAR(ntCoreObservations->resize(nCoreObservations));
        BlockDescriptor<algorithmFPType> coreObservationsRows;
        DAAL_CHECK_STATUS_VAR(ntCoreObservations->getBlockOfRows(0, nCoreObservations, writeOnly, coreObservationsRows));
        auto coreObservations = coreObservationsRows.getBuffer();

        size_t pos = 0;
        int result = 0;
        for (size_t i = 0; i < nRows; i++)
        {
            if (!isCore[i])
            {
                continue;
            }
            BlockDescriptor<algorithmFPType> dataRows;
            DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(i, 1, readOnly, dataRows));
            auto data = dataRows.getBuffer();

            Status st;
            context.copy(UniversalBuffer(coreObservations), pos * nFeatures, UniversalBuffer(data), 0, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            pos++;
            DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(dataRows));
        }
        if (result)
        {
            return Status(services::ErrorMemoryCopyFailedInternal);
        }
    }

    return Status();
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::compute(const NumericTable * x, const NumericTable * ntWeights, NumericTable * ntAssignments,
                                                        NumericTable * ntNClusters, NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                                        const Parameter * par)
{
    Status s;
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    const algorithmFPType epsilon         = par->epsilon;
    const algorithmFPType minObservations = par->minObservations;
    const uint32_t minkowskiPower         = (uint32_t)2.0;
    algorithmFPType epsP                  = 1.0;
    for (uint32_t i = 0; i < minkowskiPower; i++) epsP *= epsilon;

    NumericTable * ntData = const_cast<NumericTable *>(x);
    const size_t nRows    = ntData->getNumberOfRows();
    const size_t dim      = ntData->getNumberOfColumns();

    BlockDescriptor<algorithmFPType> dataRows;
    ntData->getBlockOfRows(0, nRows, readOnly, dataRows);
    auto data = dataRows.getBuffer();

    BlockDescriptor<int> assignRows;
    DAAL_CHECK_STATUS_VAR(ntAssignments->getBlockOfRows(0, nRows, writeOnly, assignRows));
    auto assignments = daal::oneapi::internal::UniversalBuffer(assignRows.getBuffer());
    context.fill(assignments, undefined, &s);
    DAAL_CHECK_STATUS_VAR(s);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, _queueBlockSize, nRows);

    auto queueBlockDistances = context.allocate(TypeIds::id<algorithmFPType>(), _queueBlockSize * nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto singlePointDistances = context.allocate(TypeIds::id<algorithmFPType>(), nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto queue = context.allocate(TypeIds::id<int>(), nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto isCore = context.allocate(TypeIds::id<int>(), nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto countersTotal = context.allocate(TypeIds::id<int>(), _chunkNumber, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto countersNewNeighbors = context.allocate(TypeIds::id<int>(), _chunkNumber, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto chunkOffests = context.allocate(TypeIds::id<int>(), _chunkNumber, &s);
    DAAL_CHECK_STATUS_VAR(s);

    context.fill(isCore, 0, &s);
    DAAL_CHECK_STATUS_VAR(s);

    size_t nClusters    = 0;
    uint32_t queueBegin = 0;
    uint32_t queueEnd   = 0;

    for (uint32_t i = 0; i < nRows; i++)
    {
        {
            auto pointAssignment = assignments.template get<int>().getSubBuffer(i, 1, &s);
            DAAL_CHECK_STATUS_VAR(s);
            auto assignPtr = pointAssignment.toHost(ReadWriteMode::readOnly);
            if (!assignPtr.get())
            {
                return Status(ErrorNullPtr);
            }
            if (assignPtr.get()[0] != undefined)
            {
                continue;
            }
        }
        DAAL_CHECK_STATUS_VAR(getPointDistances(data, nRows, i, dim, minkowskiPower, singlePointDistances));
        DAAL_CHECK_STATUS_VAR(countPointNeighbors(assignments, singlePointDistances, i, -1, nRows, _chunkNumber, epsP, queue, countersTotal,
                                                  countersNewNeighbors)); //done
        uint32_t numTotalNeighbors = sumCounters(countersTotal, _chunkNumber);
        uint32_t numNewNeighbors   = sumCounters(countersNewNeighbors, _chunkNumber);
        if (numTotalNeighbors < minObservations)
        {
            DAAL_CHECK_STATUS_VAR(setBufferValue(assignments, i, noise));
            continue;
        }
        nClusters++;
        DAAL_CHECK_STATUS_VAR(setBufferValue(isCore, i, 1));
        DAAL_CHECK_STATUS_VAR(countOffsets(countersNewNeighbors, _chunkNumber, chunkOffests));
        DAAL_CHECK_STATUS_VAR(
            pushNeighborsToQueue(singlePointDistances, chunkOffests, i, nClusters - 1, -1, _chunkNumber, nRows, queueEnd, epsP, assignments, queue));
        queueEnd += numNewNeighbors;
        while (queueBegin < queueEnd)
        {
            QueueBlock queueBlock(queueBegin, queueEnd, _queueBlockSize);
            getQueueBlockDistances(data, nRows, queue, queueBegin, queueBlock.count, dim, minkowskiPower, queueBlockDistances);
            for (uint32_t j = 0; j < queueBlock.count; j++)
            {
                countPointNeighbors(assignments, queueBlockDistances, queueBegin + j, nRows * j, nRows, _chunkNumber, epsP, queue, countersTotal,
                                    countersNewNeighbors);
                uint32_t curTotalNeighbors = sumCounters(countersTotal, _chunkNumber);
                uint32_t curNewNeighbors   = sumCounters(countersNewNeighbors, _chunkNumber);
                setBufferValueByQueueIndex(assignments, queue, queueBegin + j, nClusters - 1);
                if (curTotalNeighbors < minObservations)
                {
                    continue;
                }
                setBufferValueByQueueIndex(isCore, queue, queueBegin + j, 1);
                countOffsets(countersNewNeighbors, _chunkNumber, chunkOffests);
                pushNeighborsToQueue(queueBlockDistances, chunkOffests, queueBegin + j, nClusters - 1, nRows * j, _chunkNumber, nRows, queueEnd, epsP,
                                     assignments, queue);
                queueEnd += curNewNeighbors;
            }
            queueBegin += queueBlock.count;
        }
    }

    ntData->releaseBlockOfRows(dataRows);
    BlockDescriptor<int> nClustersRows;
    DAAL_CHECK_STATUS_VAR(ntNClusters->getBlockOfRows(0, 1, writeOnly, nClustersRows));
    nClustersRows.getBuffer().toHost(ReadWriteMode::writeOnly).get()[0] = nClusters;
    if (par->resultsToCompute & (computeCoreIndices | computeCoreObservations))
    {
        auto cr = isCore.template get<int>().toHost(ReadWriteMode::readOnly);
        DAAL_CHECK_STATUS_VAR(processResultsToCompute(par->resultsToCompute, cr.get(), ntData, ntCoreIndices, ntCoreObservations));
    }
    return s;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::pushNeighborsToQueue(const UniversalBuffer & distances,
                                                                               const UniversalBuffer & chunkOffests, uint32_t rowId,
                                                                               uint32_t clusterId, uint32_t chunkOffset, uint32_t numberOfChunks,
                                                                               uint32_t nRows, uint32_t queueEnd, algorithmFPType epsP,
                                                                               UniversalBuffer & assignments, UniversalBuffer & queue)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.pushNeighborsToQueue);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("push_to_queue", &st);
    DAAL_CHECK_STATUS_VAR(st);

    uint32_t chunkSize = nRows / numberOfChunks + uint32_t(bool(nRows % numberOfChunks));

    KernelArguments args(11);
    args.set(0, distances, AccessModeIds::read);
    args.set(1, chunkOffests, AccessModeIds::read);
    args.set(2, assignments, AccessModeIds::readwrite);
    args.set(3, queue, AccessModeIds::readwrite);
    args.set(4, queueEnd);
    args.set(5, rowId);
    args.set(6, clusterId);
    args.set(7, chunkOffset);
    args.set(8, chunkSize);
    args.set(9, epsP);
    args.set(10, nRows);

    KernelRange local_range(1, _maxWorkgroupSize);
    KernelRange global_range(numberOfChunks * _minSubgroupSize / _maxWorkgroupSize + 1, _maxWorkgroupSize);

    KernelNDRange range(2);
    range.global(global_range, &st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, &st);
    DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::countOffsets(const UniversalBuffer & counters, uint32_t numberOfChunks,
                                                                       UniversalBuffer & chunkOffests)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.countOffsets);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("count_offsets", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(3);
    args.set(0, counters, AccessModeIds::read);
    args.set(1, chunkOffests, AccessModeIds::write);
    args.set(2, numberOfChunks);

    KernelRange local_range(1, _minSubgroupSize);
    context.run(local_range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::setBufferValue(UniversalBuffer & buffer, uint32_t index, int value)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.setBufferValue);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("set_buffer_value", &st);
    DAAL_CHECK_STATUS_VAR(st);
    KernelArguments args(3);
    args.set(0, buffer, AccessModeIds::readwrite);
    args.set(1, index);
    args.set(2, value);

    KernelRange global_range(1);
    context.run(global_range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::setBufferValueByQueueIndex(UniversalBuffer & buffer, const UniversalBuffer & queue,
                                                                                     uint32_t posInQueue, int value)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.setBufferValueByIndirectIndex);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("set_buffer_value_by_queue_index", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(4);
    args.set(0, queue, AccessModeIds::read);
    args.set(1, buffer, AccessModeIds::readwrite);
    args.set(2, posInQueue);
    args.set(3, value);

    KernelRange global_range(1);
    context.run(global_range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::getPointDistances(const UniversalBuffer & data, uint32_t nRows, uint32_t rowId,
                                                                            uint32_t dim, uint32_t minkowskiPower, UniversalBuffer & pointDistances)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.getPointDistances);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("point_distances", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(6);
    args.set(0, data, AccessModeIds::read);
    args.set(1, pointDistances, AccessModeIds::write);
    args.set(2, rowId);
    args.set(3, minkowskiPower);
    args.set(4, dim);
    args.set(5, nRows);

    KernelRange local_range(1, _maxWorkgroupSize);
    KernelRange global_range(nRows / _minSubgroupSize + 1, _maxWorkgroupSize);

    KernelNDRange range(2);
    range.global(global_range, &st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, &st);
    DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::getQueueBlockDistances(const UniversalBuffer & data, uint32_t nRows, const UniversalBuffer & queue,
                                                                       uint32_t queueBegin, uint32_t queueBlockSize, uint32_t dim,
                                                                       uint32_t minkowskiPower, UniversalBuffer & queueBlockDistances)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.getQueueBlockDistances);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("queue_block_distances", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(8);
    args.set(0, data, AccessModeIds::read);
    args.set(1, queue, AccessModeIds::read);
    args.set(2, queueBlockDistances, AccessModeIds::write);
    args.set(3, queueBegin);
    args.set(4, queueBlockSize);
    args.set(5, minkowskiPower);
    args.set(6, dim);
    args.set(7, nRows);

    KernelRange local_range(1, _maxWorkgroupSize);
    KernelRange global_range(queueBlockSize, _maxWorkgroupSize);

    KernelNDRange range(2);
    range.global(global_range, &st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, &st);
    DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::countPointNeighbors(const UniversalBuffer & assignments, const UniversalBuffer & pointDistances,
                                                                    uint32_t rowId, int chunkOffset, uint32_t nRows, uint32_t numberOfChunks,
                                                                    algorithmFPType epsP, const UniversalBuffer & queue,
                                                                    UniversalBuffer & countersTotal, UniversalBuffer & countersNewNeighbors)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.countPointNeighbors);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("count_neighbors", &st);
    DAAL_CHECK_STATUS_VAR(st);

    uint32_t chunkSize = nRows / numberOfChunks + uint32_t(bool(nRows % numberOfChunks));
    KernelArguments args(10);
    args.set(0, assignments, AccessModeIds::read);
    args.set(1, pointDistances, AccessModeIds::read);
    args.set(2, rowId);
    args.set(3, chunkOffset);
    args.set(4, chunkSize);
    args.set(5, nRows);
    args.set(6, epsP);
    args.set(7, queue, AccessModeIds::read);
    args.set(8, countersTotal, AccessModeIds::write);
    args.set(9, countersNewNeighbors, AccessModeIds::write);

    KernelRange local_range(1, _maxWorkgroupSize);
    KernelRange global_range(numberOfChunks * _minSubgroupSize / _maxWorkgroupSize + 1, _maxWorkgroupSize);

    KernelNDRange range(2);
    range.global(global_range, &st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, &st);
    DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
uint32_t DBSCANBatchKernelUCAPI<algorithmFPType>::sumCounters(const UniversalBuffer & counters, uint32_t numberOfChunks)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.sumCounters);
    auto cntPtr  = counters.template get<int>().toHost(ReadWriteMode::writeOnly).get();
    uint32_t ret = 0;
    for (uint32_t i = 0; i < numberOfChunks; i++) ret += cntPtr[i];
    return ret;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::buildProgram(ClKernelFactoryIface & kernel_factory)
{
    Status st;
    auto fptype_name   = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    build_options.add(" -D_NOISE_=-1 -D_UNDEFINED_=-2 ");

    services::String cachekey("__daal_algorithms_dbscan_block_");
    cachekey.add(fptype_name);
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernel_factory.build(ExecutionTargetIds::device, cachekey.c_str(), dbscan_cl_kernels, build_options.c_str(), &st);
    }
    return st;
}

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal
