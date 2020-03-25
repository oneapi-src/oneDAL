/* file: dbscan_kernel_ucapi.h */
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
//  Declaration of template function that computes DBSCAN for GPU.
//--
*/

#ifndef __DBSCAN_KERNEL_UCAPI_H
#define __DBSCAN_KERNEL_UCAPI_H

#include "algorithms/kernel/kernel.h"
#include "data_management/data/numeric_table.h"
#include "oneapi/internal/execution_context.h"

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
class DBSCANBatchKernelUCAPI : public Kernel
{
public:
    services::Status compute(const NumericTable * ntData, const NumericTable * ntWeights, NumericTable * ntAssignments,
                                    NumericTable * ntNClusters, NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                    const Parameter * par);

private:
    services::Status processResultsToCompute(DAAL_UINT64 resultsToCompute, int * const isCore, NumericTable * ntData,
                                             NumericTable * ntCoreIndices, NumericTable * ntCoreObservations);
    services::Status pushNeighborsToQueue(
        const oneapi::internal::UniversalBuffer& distances,
        const oneapi::internal::UniversalBuffer& chunkOffests,
        uint32_t rowId,
        uint32_t clusterId, 
        uint32_t chunkOffset,
        uint32_t numberOfChunks,
        uint32_t nRows,
        uint32_t qEnd,
        algorithmFPType eps,
        oneapi::internal::UniversalBuffer& assignments,
        oneapi::internal::UniversalBuffer& queue);

    services::Status countOffsets(
        const oneapi::internal::UniversalBuffer& counters,
        uint32_t numberOfChunks,
        oneapi::internal::UniversalBuffer& offsets);

    services::Status setBufferValue(
        oneapi::internal::UniversalBuffer& buffer,
        uint32_t index,
        int value); 
 
    services::Status setBufferValueByQueueIndex(
        oneapi::internal::UniversalBuffer& buffer,
        const oneapi::internal::UniversalBuffer& queue,
        uint32_t posInQueue,
        int value); 

    services::Status getPointDistances(
        const oneapi::internal::UniversalBuffer& data,
        uint32_t nRows, 
        uint32_t rowId,
        uint32_t dim, 
        uint32_t minkowskiPower,
        oneapi::internal::UniversalBuffer& pointDistances);

    services::Status getQueueBlockDistances(
        const oneapi::internal::UniversalBuffer& data,
        uint32_t nRows, 
        const oneapi::internal::UniversalBuffer& queue,
        uint32_t queueBegin, 
        uint32_t queueBlockSize,
        uint32_t dim, 
        uint32_t minkowskiPower,
        oneapi::internal::UniversalBuffer& queueBlockDistances);

    services::Status countPointNeighbors(
        const oneapi::internal::UniversalBuffer& assignments,
        const oneapi::internal::UniversalBuffer& pointDistances,
        uint32_t rowId, 
        int chunkOffset,
        uint32_t nRows,
        uint32_t numberOfChunks,
        algorithmFPType epsP,
        const oneapi::internal::UniversalBuffer& queue,
        oneapi::internal::UniversalBuffer& countersTotal,
        oneapi::internal::UniversalBuffer& countersNewNeighbors);

    uint32_t sumCounters(
        const oneapi::internal::UniversalBuffer& counters,
        uint32_t numberOfChunks);


    services::Status buildProgram(
        oneapi::internal::ClKernelFactoryIface & kernel_factory);

    size_t _minSubgroupSize = 16;
    size_t _maxWorkgroupSize = 256;
    size_t _chunkNumber = 64;
    size_t _queueBlockSize = 64;
};

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
