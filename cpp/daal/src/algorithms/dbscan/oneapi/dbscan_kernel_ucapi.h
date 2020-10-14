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

#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "services/internal/sycl/execution_context.h"

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
    services::Status compute(const daal::data_management::NumericTable * ntData, const daal::data_management::NumericTable * ntWeights,
                             daal::data_management::NumericTable * ntAssignments, daal::data_management::NumericTable * ntNClusters,
                             daal::data_management::NumericTable * ntCoreIndices, daal::data_management::NumericTable * ntCoreObservations,
                             const Parameter * par);

private:
    services::Status processResultsToCompute(DAAL_UINT64 resultsToCompute, daal::data_management::NumericTable * ntData,
                                             daal::data_management::NumericTable * ntCoreIndices,
                                             daal::data_management::NumericTable * ntCoreObservations);
    services::Status pushNeighborsToQueue(const services::internal::sycl::UniversalBuffer & distances,
                                          const services::internal::sycl::UniversalBuffer & chunkOffests, uint32_t rowId, uint32_t clusterId,
                                          uint32_t chunkOffset, uint32_t nRows, uint32_t qEnd, algorithmFPType eps,
                                          services::internal::sycl::UniversalBuffer & assignments, services::internal::sycl::UniversalBuffer & queue);

    services::Status countOffsets(const services::internal::sycl::UniversalBuffer & counters, services::internal::sycl::UniversalBuffer & offsets);

    services::Status setBufferValue(services::internal::sycl::UniversalBuffer & buffer, uint32_t index, int value);

    services::Status setBufferValueByQueueIndex(services::internal::sycl::UniversalBuffer & buffer,
                                                const services::internal::sycl::UniversalBuffer & queue, uint32_t posInQueue, int value);

    services::Status getPointDistances(const services::internal::sycl::UniversalBuffer & data, uint32_t nRows, uint32_t rowId, uint32_t dim,
                                       uint32_t minkowskiPower, services::internal::sycl::UniversalBuffer & pointDistances);

    services::Status getQueueBlockDistances(const services::internal::sycl::UniversalBuffer & data, uint32_t nRows,
                                            const services::internal::sycl::UniversalBuffer & queue, uint32_t queueBegin, uint32_t queueBlockSize,
                                            uint32_t dim, uint32_t minkowskiPower, services::internal::sycl::UniversalBuffer & queueBlockDistances);

    services::Status countPointNeighbors(const services::internal::sycl::UniversalBuffer & assignments,
                                         const services::internal::sycl::UniversalBuffer & pointDistances, uint32_t rowId, int chunkOffset,
                                         uint32_t nRows, algorithmFPType epsP, const services::internal::sycl::UniversalBuffer & queue,
                                         services::internal::sycl::UniversalBuffer & countersTotal,
                                         services::internal::sycl::UniversalBuffer & countersNewNeighbors);

    uint32_t sumCounters(const services::internal::sycl::UniversalBuffer & counters, services::Status & s);

    bool canQueryRow(const services::internal::sycl::UniversalBuffer & assignments, uint32_t rowIndex, services::Status & s);

    uint32_t computeQueueBlockSize(uint32_t queueBegin, uint32_t queueEnd);

    uint32_t getWorkgroupNumber(uint32_t numberOfChunks) { return numberOfChunks * _minSubgroupSize / _maxWorkgroupSize + 1; }

    services::Status initializeBuffers(uint32_t nRows);

    services::Status buildProgram(services::internal::sycl::ClKernelFactoryIface & kernel_factory);

    void calculateChunks(uint32_t nRows);

    static const uint32_t _minSubgroupSize              = 16;
    static const uint32_t _maxWorkgroupSize             = 256;
    static const uint32_t _minRecommendedNumberOfChunks = 64;
    static const uint32_t _recommendedChunkSize         = 256;
    static const uint32_t _minChunkSize                 = 16;
    static const uint32_t _queueBlockSize               = 64;
    uint32_t _chunkNumber                               = _minRecommendedNumberOfChunks;
    uint32_t _chunkSize                                 = _recommendedChunkSize;

    services::internal::sycl::UniversalBuffer _queueBlockDistances;
    services::internal::sycl::UniversalBuffer _singlePointDistances;
    services::internal::sycl::UniversalBuffer _queue;
    services::internal::sycl::UniversalBuffer _isCore;
    services::internal::sycl::UniversalBuffer _countersTotal;
    services::internal::sycl::UniversalBuffer _countersNewNeighbors;
    services::internal::sycl::UniversalBuffer _chunkOffsets;
};

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
