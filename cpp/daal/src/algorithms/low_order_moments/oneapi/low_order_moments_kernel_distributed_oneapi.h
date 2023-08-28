/* file: low_order_moments_kernel_distributed_oneapi.h */
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
//  Declaration of template function that calculate low order moments.
//--
*/

#ifndef __LOW_ORDER_MOMENTS_KERNEL_DISTRIBUTED_ONEAPI_H__
#define __LOW_ORDER_MOMENTS_KERNEL_DISTRIBUTED_ONEAPI_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/moments/low_order_moments_types.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_arrays.h"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace oneapi
{
namespace internal
{
template <typename algorithmFPType, EstimatesToCompute scope>
struct TaskInfoDistributed;

template <typename algorithmFPType>
struct TaskInfoDistributed<algorithmFPType, estimatesMinMax>
{
    constexpr static uint32_t nResults        = 2;
    constexpr static uint32_t nBuffers        = 2;
    constexpr static uint32_t nPartialResults = 2;
    // names of used kernels
    static const char * kMergeDistrBlocksName;
    // kernels build options
    static const char * kBldOptFNameSuff;
    static const char * kBldOptScope;
    static const char * kCacheKey;

    int resPartialIds[nPartialResults]; // required set of partial results' ids
    int resIds[nResults];               // required set of results' ids

    TaskInfoDistributed() : resPartialIds { partialMinimum, partialMaximum }, resIds { minimum, maximum } {}
};

template <typename algorithmFPType>
struct TaskInfoDistributed<algorithmFPType, estimatesMeanVariance>
{
    constexpr static uint32_t nResults         = 2;
    constexpr static uint32_t nBuffers         = 2;
    constexpr static uint32_t nPartialResults  = 2;
    constexpr static uint32_t nFinalizeResults = 2;
    // names of used kernels
    static const char * kMergeDistrBlocksName;
    static const char * kFinalizeName;
    // kernels build options
    static const char * kBldOptFNameSuff;
    static const char * kBldOptScope;
    static const char * kCacheKey;

    int resIds[nResults];                 // required set of results' ids
    int resPartialIds[nPartialResults];   // required set of partial results' ids
    int resFinalizeIds[nFinalizeResults]; // set of results' ids which will be processed on finalize stage

    TaskInfoDistributed() : resPartialIds { partialSum, partialSumSquaresCentered }, resIds { mean, variance }, resFinalizeIds { mean, variance } {}
};

template <typename algorithmFPType>
struct TaskInfoDistributed<algorithmFPType, estimatesAll>
{
    constexpr static uint32_t nResults         = lastResultId + 1;
    constexpr static uint32_t nBuffers         = 5;
    constexpr static uint32_t nPartialResults  = lastPartialResultId; // removed '+1' due to nObservations is mapped separately
    constexpr static uint32_t nFinalizeResults = 5;
    // names of used kernels
    static const char * kMergeDistrBlocksName;
    static const char * kFinalizeName;
    // kernels build options
    static const char * kBldOptFNameSuff;
    static const char * kBldOptScope;
    static const char * kCacheKey;

    int resIds[nResults];                 // required set of results' ids
    int resPartialIds[nPartialResults];   // required set of partial results' ids
    int resFinalizeIds[nFinalizeResults]; // set of results' ids which will be processed on finalize stage

    TaskInfoDistributed()
        : resPartialIds { partialMinimum, partialMaximum, partialSum, partialSumSquares, partialSumSquaresCentered },
          resIds { minimum, maximum, sum, sumSquares, sumSquaresCentered, mean, secondOrderRawMoment, variance, standardDeviation, variation },
          resFinalizeIds { mean, secondOrderRawMoment, variance, standardDeviation, variation }
    {}
};

/* distributed kernel class */
template <typename algorithmFPType, low_order_moments::Method method>
class LowOrderMomentsDistributedKernelOneAPI : public daal::algorithms::Kernel
{
public:
    services::Status compute(data_management::DataCollection * partialResultsCollection, PartialResult * partialResult, const Parameter * parameter);
    services::Status finalizeCompute(PartialResult * partialResult, Result * result, const Parameter * parameter);
};

/* distributed task class */
template <typename algorithmFPType, EstimatesToCompute scope>
class LowOrderMomentsDistributedTaskOneAPI : public TaskInfoDistributed<algorithmFPType, scope>
{
public:
    LowOrderMomentsDistributedTaskOneAPI(services::internal::sycl::ExecutionContextIface & context,
                                         data_management::DataCollection * partialResultsCollection, PartialResult * partialResult,
                                         services::Status & status);
    LowOrderMomentsDistributedTaskOneAPI(const LowOrderMomentsDistributedTaskOneAPI &)             = delete;
    LowOrderMomentsDistributedTaskOneAPI & operator=(const LowOrderMomentsDistributedTaskOneAPI &) = delete;
    virtual ~LowOrderMomentsDistributedTaskOneAPI();
    Status compute();

private:
    static constexpr size_t _uint32max = static_cast<size_t>(services::internal::MaxVal<uint32_t>::get());

    static constexpr uint32_t _blockAlignment = 64; // alignment (in bytes) for distibuted data blocks

    uint32_t nDistrBlocks;
    uint32_t nFeatures;
    uint32_t nElemsInStride; // num of elems between feature values from different distributed blocks

    data_management::DataCollection * partResultsCollection;

    NumericTablePtr nObservationsTable;
    BlockDescriptor<algorithmFPType> nObservationsBD;
    algorithmFPType * pNObservations;

    services::internal::sycl::UniversalBuffer bNVec; // contains info about num of vectors in distributed block

    NumericTablePtr resultTable[TaskInfoDistributed<algorithmFPType, scope>::nPartialResults];
    services::internal::sycl::UniversalBuffer bAuxBuffers[TaskInfoDistributed<algorithmFPType, scope>::nBuffers];
    daal::services::internal::TArray<algorithmFPType, DAAL_BASE_CPU> bAuxHostBuffers[TaskInfoDistributed<algorithmFPType, scope>::nBuffers];

    BlockDescriptor<algorithmFPType> resultBD[TaskInfoDistributed<algorithmFPType, scope>::nPartialResults];
};

/* finalize task class */
template <typename algorithmFPType, EstimatesToCompute scope>
class LowOrderMomentsDistributedFinalizeTaskOneAPI : public TaskInfoDistributed<algorithmFPType, scope>
{
public:
    LowOrderMomentsDistributedFinalizeTaskOneAPI(services::internal::sycl::ExecutionContextIface & context, PartialResult * partialResult,
                                                 Result * result, services::Status & status);
    LowOrderMomentsDistributedFinalizeTaskOneAPI(const LowOrderMomentsDistributedFinalizeTaskOneAPI &)             = delete;
    LowOrderMomentsDistributedFinalizeTaskOneAPI & operator=(const LowOrderMomentsDistributedFinalizeTaskOneAPI &) = delete;
    virtual ~LowOrderMomentsDistributedFinalizeTaskOneAPI();
    Status compute();

private:
    uint32_t nFeatures;
    constexpr static uint32_t nTotalResults =
        TaskInfoDistributed<algorithmFPType, scope>::nPartialResults + TaskInfoDistributed<algorithmFPType, scope>::nFinalizeResults;

    NumericTablePtr nObservationsTable;
    BlockDescriptor<algorithmFPType> nObservationsBD;
    algorithmFPType * pNObservations;

    NumericTablePtr resultTable[nTotalResults];

    BlockDescriptor<algorithmFPType> resultBD[nTotalResults];
};

} // namespace internal
} // namespace oneapi
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
