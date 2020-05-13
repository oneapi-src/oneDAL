/* file: dbscan_kernel_ucapi.h */
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
//  Declaration of template function that computes DBSCAN for GPU.
//--
*/

#ifndef __DBSCAN_KERNEL_UCAPI_H
#define __DBSCAN_KERNEL_UCAPI_H

#include "algorithms/kernel/kernel.h"
#include "data_management/data/numeric_table.h"
#include "oneapi/internal/execution_context.h"

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{
template <typename algorithmFPType>
class DBSCANBatchKernelUCAPI : public Kernel
{
public:
    DBSCANBatchKernelUCAPI();
    services::Status compute(const daal::data_management::NumericTable * ntData, const daal::data_management::NumericTable * ntWeights,
                             daal::data_management::NumericTable * ntAssignments, daal::data_management::NumericTable * ntNClusters,
                             daal::data_management::NumericTable * ntCoreIndices, daal::data_management::NumericTable * ntCoreObservations,
                             const Parameter * par);

private:
    services::Status processResultsToCompute(DAAL_UINT64 resultsToCompute, daal::data_management::NumericTable * ntData,
                                             daal::data_management::NumericTable * ntCoreIndices,
                                             daal::data_management::NumericTable * ntCoreObservations);
    services::Status pushNeighborsToQueue(const oneapi::internal::UniversalBuffer & distances, const oneapi::internal::UniversalBuffer & chunkOffests,
                                          uint32_t rowId, uint32_t clusterId, uint32_t chunkOffset, uint32_t nRows, uint32_t qEnd,
                                          algorithmFPType eps, oneapi::internal::UniversalBuffer & assignments,
                                          oneapi::internal::UniversalBuffer & queue);

    services::Status countOffsets(const oneapi::internal::UniversalBuffer & counters, oneapi::internal::UniversalBuffer & offsets);

    services::Status setBufferValue(oneapi::internal::UniversalBuffer & buffer, uint32_t index, int value);

    services::Status setBufferValueByQueueIndex(oneapi::internal::UniversalBuffer & buffer, const oneapi::internal::UniversalBuffer & queue,
                                                uint32_t posInQueue, int value);

    services::Status getPointDistances(const oneapi::internal::UniversalBuffer & data, uint32_t nRows, uint32_t rowId, uint32_t dim,
                                       uint32_t minkowskiPower, oneapi::internal::UniversalBuffer & pointDistances);

    services::Status getQueueBlockDistances(const oneapi::internal::UniversalBuffer & data, uint32_t nRows,
                                            const oneapi::internal::UniversalBuffer & queue, uint32_t queueBegin, uint32_t queueBlockSize,
                                            uint32_t dim, uint32_t minkowskiPower, oneapi::internal::UniversalBuffer & queueBlockDistances);

    services::Status countPointNeighbors(const oneapi::internal::UniversalBuffer & assignments,
                                         const oneapi::internal::UniversalBuffer & pointDistances, uint32_t rowId, int chunkOffset, uint32_t nRows,
                                         algorithmFPType epsP, const oneapi::internal::UniversalBuffer & queue,
                                         oneapi::internal::UniversalBuffer & countersTotal, oneapi::internal::UniversalBuffer & countersNewNeighbors);

    uint32_t sumCounters(const oneapi::internal::UniversalBuffer & counters);

    bool canQueryRow(const oneapi::internal::UniversalBuffer & assignments, uint32_t rowIndex, services::Status * s);

    uint32_t computeQueueBlockSize(uint32_t queueBegin, uint32_t queueEnd);

    uint32_t getWorkgroupNumber(uint32_t numberOfChunks) { return numberOfChunks * _minSubgroupSize / _maxWorkgroupSize + 1; }

    services::Status initializeBuffers(uint32_t nRows);

    services::Status buildProgram(oneapi::internal::ClKernelFactoryIface & kernel_factory);

    void calculateChunks(uint32_t nRows);

    uint32_t _minSubgroupSize;
    uint32_t _maxWorkgroupSize;
    static const uint32_t _minRecommendedNumberOfChunks = 64;
    static const uint32_t _recommendedChunkSize         = 256;
    static const uint32_t _minChunkSize                 = 16;
    static const uint32_t _queueBlockSize               = 64;
    uint32_t _chunkNumber                               = _minRecommendedNumberOfChunks;
    uint32_t _chunkSize                                 = _recommendedChunkSize;

    oneapi::internal::UniversalBuffer _queueBlockDistances;
    oneapi::internal::UniversalBuffer _singlePointDistances;
    oneapi::internal::UniversalBuffer _queue;
    oneapi::internal::UniversalBuffer _isCore;
    oneapi::internal::UniversalBuffer _countersTotal;
    oneapi::internal::UniversalBuffer _countersNewNeighbors;
    oneapi::internal::UniversalBuffer _chunkOffsets;
};

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
