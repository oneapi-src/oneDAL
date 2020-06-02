/* file: dbscan_dense_default_batch_ucapi_impl.i */
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
//  Implementation of default method for DBSCAN algorithm on GPU.
//--
*/

#include "algorithms/dbscan/dbscan_types.h"
#include "algorithms/kernel/dbscan/oneapi/dbscan_kernel_ucapi.h"
#include "algorithms/kernel/dbscan/oneapi/cl_kernels/dbscan_cl_kernels.cl"
#include "externals/service_ittnotify.h"

using namespace daal::services;
using namespace daal::oneapi::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{
template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::initializeBuffers(uint32_t nRows)
{
    calculateChunks(nRows);
    Status s;
    auto & context = Environment::getInstance()->getDefaultExecutionContext();
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, _queueBlockSize, nRows);
    _queueBlockDistances = context.allocate(TypeIds::id<algorithmFPType>(), _queueBlockSize * nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    _singlePointDistances = context.allocate(TypeIds::id<algorithmFPType>(), nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    _queue = context.allocate(TypeIds::id<int>(), nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    _isCore = context.allocate(TypeIds::id<int>(), nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    _countersTotal = context.allocate(TypeIds::id<int>(), _chunkNumber, &s);
    DAAL_CHECK_STATUS_VAR(s);
    _countersNewNeighbors = context.allocate(TypeIds::id<int>(), _chunkNumber, &s);
    DAAL_CHECK_STATUS_VAR(s);
    _chunkOffsets = context.allocate(TypeIds::id<int>(), _chunkNumber, &s);
    DAAL_CHECK_STATUS_VAR(s);
    context.fill(_isCore, 0, &s);
    DAAL_CHECK_STATUS_VAR(s);
    return s;
}

template <typename algorithmFPType>
bool DBSCANBatchKernelUCAPI<algorithmFPType>::canQueryRow(const UniversalBuffer & assignments, uint32_t rowIndex, Status * s)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.canQueryRow);
    auto pointAssignment = assignments.template get<int>().getSubBuffer(rowIndex, 1, s);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(s, false);

    auto assignPtr = pointAssignment.toHost(ReadWriteMode::readOnly);
    if (!assignPtr.get())
    {
        return Status(ErrorNullPtr);
    }
    return (assignPtr.get()[0] == undefined);
}

template <typename algorithmFPType>
uint32_t DBSCANBatchKernelUCAPI<algorithmFPType>::computeQueueBlockSize(uint32_t queueBegin, uint32_t queueEnd)
{
    uint32_t size = queueEnd - queueBegin;
    if (size > _queueBlockSize)
    {
        size = _queueBlockSize;
    }
    return size;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::processResultsToCompute(DAAL_UINT64 resultsToCompute, NumericTable * ntData,
                                                                        NumericTable * ntCoreIndices, NumericTable * ntCoreObservations)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.processResultsToCompute);
    auto isCoreHost = _isCore.template get<int>().toHost(ReadWriteMode::readOnly);
    auto isCore     = isCoreHost.get();
    if (!isCore)
    {
        return Status(ErrorNullPtr);
    }

    const uint32_t nRows     = ntData->getNumberOfRows();
    const uint32_t nFeatures = ntData->getNumberOfColumns();

    uint32_t nCoreObservations = 0;

    for (uint32_t i = 0; i < nRows; i++)
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
        auto coreIndices = indexRows.getBlockPtr();
        if (!coreIndices)
        {
            return Status(ErrorNullPtr);
        }

        uint32_t pos = 0;
        for (uint32_t i = 0; i < nRows; i++)
        {
            if (!isCore[i])
            {
                continue;
            }
            coreIndices[pos] = i;
            pos++;
        }
    }

    if (resultsToCompute & computeCoreObservations)
    {
        DAAL_CHECK_STATUS_VAR(ntCoreObservations->resize(nCoreObservations));
        BlockDescriptor<algorithmFPType> coreObservationsRows;
        DAAL_CHECK_STATUS_VAR(ntCoreObservations->getBlockOfRows(0, nCoreObservations, writeOnly, coreObservationsRows));
        auto coreObservations    = coreObservationsRows.getBuffer();
        auto coreObservationsPtr = coreObservations.toHost(ReadWriteMode::writeOnly);
        BlockDescriptor<algorithmFPType> dataRows;
        DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(0, nRows, readOnly, dataRows));
        auto data    = dataRows.getBuffer();
        auto dataPtr = data.toHost(ReadWriteMode::readOnly);

        uint32_t pos = 0;
        for (uint32_t i = 0; i < nRows; i++)
        {
            if (!isCore[i])
            {
                continue;
            }
            for (uint32_t j = 0; j < nFeatures; j++) coreObservationsPtr.get()[pos * nFeatures + j] = dataPtr.get()[i * nFeatures + j];
            pos++;
        }
        DAAL_CHECK_STATUS_VAR(ntCoreObservations->releaseBlockOfRows(coreObservationsRows));
        DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(dataRows));
    }

    return Status();
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::compute(const NumericTable * x, const NumericTable * ntWeights, NumericTable * ntAssignments,
                                                        NumericTable * ntNClusters, NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                                        const Parameter * par)
{
    Status s;
    auto & context                = Environment::getInstance()->getDefaultExecutionContext();
    const uint32_t minkowskiPower = 2;
    algorithmFPType epsP          = 1.0;
    for (uint32_t i = 0; i < minkowskiPower; i++) epsP *= par->epsilon;

    NumericTable * ntData = const_cast<NumericTable *>(x);
    if (ntData->getNumberOfRows() > static_cast<size_t>(UINT_MAX) || ntData->getNumberOfColumns() > static_cast<size_t>(UINT_MAX))
    {
        return Status(ErrorBufferSizeIntegerOverflow);
    }
    const uint32_t nRows     = static_cast<uint32_t>(ntData->getNumberOfRows());
    const uint32_t nFeatures = static_cast<uint32_t>(ntData->getNumberOfColumns());

    BlockDescriptor<algorithmFPType> dataRows;
    ntData->getBlockOfRows(0, nRows, readOnly, dataRows);
    auto data = dataRows.getBuffer();

    BlockDescriptor<int> assignRows;
    DAAL_CHECK_STATUS_VAR(ntAssignments->getBlockOfRows(0, nRows, writeOnly, assignRows));
    UniversalBuffer assignments = assignRows.getBuffer();
    context.fill(assignments, undefined, &s);
    DAAL_CHECK_STATUS_VAR(s);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, _queueBlockSize, nRows);
    DAAL_CHECK_STATUS_VAR(initializeBuffers(nRows));

    uint32_t nClusters  = 0;
    uint32_t queueBegin = 0;
    uint32_t queueEnd   = 0;

    for (uint32_t i = 0; i < nRows; i++)
    {
        bool canQuery = canQueryRow(assignments, i, &s);
        DAAL_CHECK_STATUS_VAR(s);
        if (!canQuery)
        {
            continue;
        }
        DAAL_CHECK_STATUS_VAR(getPointDistances(data, nRows, i, nFeatures, minkowskiPower, _singlePointDistances));
        DAAL_CHECK_STATUS_VAR(
            countPointNeighbors(assignments, _singlePointDistances, i, -1, nRows, epsP, _queue, _countersTotal, _countersNewNeighbors));
        uint32_t numTotalNeighbors = sumCounters(_countersTotal);
        uint32_t numNewNeighbors   = sumCounters(_countersNewNeighbors);
        if (numTotalNeighbors < par->minObservations)
        {
            DAAL_CHECK_STATUS_VAR(setBufferValue(assignments, i, noise));
            continue;
        }
        nClusters++;
        DAAL_CHECK_STATUS_VAR(setBufferValue(_isCore, i, 1));
        DAAL_CHECK_STATUS_VAR(countOffsets(_countersNewNeighbors, _chunkOffsets));
        DAAL_CHECK_STATUS_VAR(
            pushNeighborsToQueue(_singlePointDistances, _chunkOffsets, i, nClusters - 1, -1, nRows, queueEnd, epsP, assignments, _queue));
        queueEnd += numNewNeighbors;
        while (queueBegin < queueEnd)
        {
            uint32_t curQueueBlockSize = computeQueueBlockSize(queueBegin, queueEnd);
            getQueueBlockDistances(data, nRows, _queue, queueBegin, curQueueBlockSize, nFeatures, minkowskiPower, _queueBlockDistances);
            for (uint32_t j = 0; j < curQueueBlockSize; j++)
            {
                countPointNeighbors(assignments, _queueBlockDistances, queueBegin + j, nRows * j, nRows, epsP, _queue, _countersTotal,
                                    _countersNewNeighbors);
                uint32_t curTotalNeighbors = sumCounters(_countersTotal);
                uint32_t curNewNeighbors   = sumCounters(_countersNewNeighbors);
                setBufferValueByQueueIndex(assignments, _queue, queueBegin + j, nClusters - 1);
                if (curTotalNeighbors < par->minObservations)
                {
                    continue;
                }
                setBufferValueByQueueIndex(_isCore, _queue, queueBegin + j, 1);
                countOffsets(_countersNewNeighbors, _chunkOffsets);
                pushNeighborsToQueue(_queueBlockDistances, _chunkOffsets, queueBegin + j, nClusters - 1, nRows * j, nRows, queueEnd, epsP,
                                     assignments, _queue);
                queueEnd += curNewNeighbors;
            }
            queueBegin += curQueueBlockSize;
        }
    }
    ntData->releaseBlockOfRows(dataRows);
    BlockDescriptor<int> nClustersRows;
    DAAL_CHECK_STATUS_VAR(ntNClusters->getBlockOfRows(0, 1, writeOnly, nClustersRows));
    nClustersRows.getBuffer().toHost(ReadWriteMode::writeOnly).get()[0] = nClusters;
    if (par->resultsToCompute & (computeCoreIndices | computeCoreObservations))
    {
        DAAL_CHECK_STATUS_VAR(processResultsToCompute(par->resultsToCompute, ntData, ntCoreIndices, ntCoreObservations));
    }
    return s;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::pushNeighborsToQueue(const UniversalBuffer & distances,
                                                                               const UniversalBuffer & chunkOffests, uint32_t rowId,
                                                                               uint32_t clusterId, uint32_t chunkOffset, uint32_t nRows,
                                                                               uint32_t queueEnd, algorithmFPType epsP, UniversalBuffer & assignments,
                                                                               UniversalBuffer & queue)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.pushNeighborsToQueue);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("push_points_to_queue", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(11);
    args.set(0, distances, AccessModeIds::read);
    args.set(1, chunkOffests, AccessModeIds::read);
    args.set(2, queueEnd);
    args.set(3, rowId);
    args.set(4, clusterId);
    args.set(5, chunkOffset);
    args.set(6, _chunkSize);
    args.set(7, epsP);
    args.set(8, nRows);
    args.set(9, assignments, AccessModeIds::readwrite);
    args.set(10, queue, AccessModeIds::readwrite);

    KernelRange local_range(1, _maxWorkgroupSize);
    KernelRange global_range(getWorkgroupNumber(_chunkNumber), _maxWorkgroupSize);

    KernelNDRange range(2);
    range.global(global_range, &st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, &st);
    DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::countOffsets(const UniversalBuffer & counters, UniversalBuffer & chunkOffests)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.countOffsets);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("compute_chunk_offsets", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(3);
    args.set(0, counters, AccessModeIds::read);
    args.set(1, _chunkNumber);
    args.set(2, chunkOffests, AccessModeIds::write);

    KernelRange local_range(_minSubgroupSize);
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
    args.set(0, index);
    args.set(1, value);
    args.set(2, buffer, AccessModeIds::readwrite);

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
    args.set(1, posInQueue);
    args.set(2, value);
    args.set(3, buffer, AccessModeIds::readwrite);

    KernelRange global_range(1);
    context.run(global_range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::getPointDistances(const UniversalBuffer & data, uint32_t nRows, uint32_t rowId,
                                                                            uint32_t nFeatures, uint32_t minkowskiPower,
                                                                            UniversalBuffer & pointDistances)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.getPointDistances);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("compute_point_distances", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(6);
    args.set(0, data, AccessModeIds::read);
    args.set(1, rowId);
    args.set(2, minkowskiPower);
    args.set(3, nFeatures);
    args.set(4, nRows);
    args.set(5, pointDistances, AccessModeIds::write);

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
                                                                       uint32_t queueBegin, uint32_t queueBlockSize, uint32_t nFeatures,
                                                                       uint32_t minkowskiPower, UniversalBuffer & queueBlockDistances)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.getQueueBlockDistances);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("compute_queue_block_distances", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(8);
    args.set(0, data, AccessModeIds::read);
    args.set(1, queue, AccessModeIds::read);
    args.set(2, queueBegin);
    args.set(3, queueBlockSize);
    args.set(4, minkowskiPower);
    args.set(5, nFeatures);
    args.set(6, nRows);
    args.set(7, queueBlockDistances, AccessModeIds::write);

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
                                                                    uint32_t rowId, int chunkOffset, uint32_t nRows, algorithmFPType epsP,
                                                                    const UniversalBuffer & queue, UniversalBuffer & countersTotal,
                                                                    UniversalBuffer & countersNewNeighbors)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.countPointNeighbors);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("count_neighbors_by_type", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(10);
    args.set(0, assignments, AccessModeIds::read);
    args.set(1, pointDistances, AccessModeIds::read);
    args.set(2, rowId);
    args.set(3, chunkOffset);
    args.set(4, _chunkSize);
    args.set(5, nRows);
    args.set(6, epsP);
    args.set(7, queue, AccessModeIds::read);
    args.set(8, countersTotal, AccessModeIds::write);
    args.set(9, countersNewNeighbors, AccessModeIds::write);

    KernelRange local_range(1, _maxWorkgroupSize);
    KernelRange global_range(getWorkgroupNumber(_chunkNumber), _maxWorkgroupSize);

    KernelNDRange range(2);
    range.global(global_range, &st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, &st);
    DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
uint32_t DBSCANBatchKernelUCAPI<algorithmFPType>::sumCounters(const UniversalBuffer & counters)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.sumCounters);
    auto countersHost      = counters.template get<int>().toHost(ReadWriteMode::writeOnly);
    auto countersPtr       = countersHost.get();
    uint32_t sumOfCounters = 0;
    for (uint32_t i = 0; i < _chunkNumber; i++)
    {
        sumOfCounters += countersPtr[i];
    }
    return sumOfCounters;
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

template <typename algorithmFPType>
void DBSCANBatchKernelUCAPI<algorithmFPType>::calculateChunks(uint32_t nRows)
{
    _chunkSize   = _recommendedChunkSize;
    _chunkNumber = nRows / _chunkSize + uint32_t(bool(nRows % _chunkSize));
    while (_chunkNumber < _minRecommendedNumberOfChunks && _chunkSize > _minChunkSize)
    {
        _chunkSize /= 2;
        _chunkNumber = nRows / _chunkSize + uint32_t(bool(nRows % _chunkSize));
    }
}

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal
