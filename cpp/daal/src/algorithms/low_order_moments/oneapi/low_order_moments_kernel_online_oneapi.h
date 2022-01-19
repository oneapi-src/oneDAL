/* file: low_order_moments_kernel_online_oneapi.h */
/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef __LOW_ORDER_MOMENTS_KERNEL_ONLINE_ONEAPI_H__
#define __LOW_ORDER_MOMENTS_KERNEL_ONLINE_ONEAPI_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/moments/low_order_moments_types.h"
#include "src/services/service_data_utils.h"

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
struct TaskInfoOnline;

template <typename algorithmFPType>
struct TaskInfoOnline<algorithmFPType, estimatesMinMax>
{
    constexpr static uint32_t nResults              = 2;
    constexpr static uint32_t nBuffers              = 2;
    constexpr static bool isRowsInBlockInfoRequired = false;
    constexpr static uint32_t nPartialResults       = 2;
    // names of used kernels
    static const char * kSinglePassName;
    static const char * kProcessBlocksName;
    static const char * kMergeBlocksName;
    // kernels build options
    static const char * kBldOptFNameSuff;
    static const char * kBldOptScope;
    static const char * kCacheKey;

    int resPartialIds[nPartialResults]; // required set of partial results' ids
    int resIds[nResults];               // required set of results' ids

    TaskInfoOnline() : resPartialIds { partialMinimum, partialMaximum }, resIds { minimum, maximum } {}
};

template <typename algorithmFPType>
struct TaskInfoOnline<algorithmFPType, estimatesMeanVariance>
{
    constexpr static uint32_t nResults              = 2;
    constexpr static uint32_t nBuffers              = 2;
    constexpr static bool isRowsInBlockInfoRequired = true;
    constexpr static uint32_t nPartialResults       = 2;
    constexpr static uint32_t nFinalizeResults      = 2;
    // names of used kernels
    static const char * kSinglePassName;
    static const char * kProcessBlocksName;
    static const char * kMergeBlocksName;
    static const char * kFinalizeName;
    // kernels build options
    static const char * kBldOptFNameSuff;
    static const char * kBldOptScope;
    static const char * kCacheKey;

    int resIds[nResults];                 // required set of results' ids
    int resPartialIds[nPartialResults];   // required set of partial results' ids
    int resFinalizeIds[nFinalizeResults]; // set of results' ids which will be processed on finalize stage

    TaskInfoOnline() : resPartialIds { partialSum, partialSumSquaresCentered }, resIds { mean, variance }, resFinalizeIds { mean, variance } {}
};

template <typename algorithmFPType>
struct TaskInfoOnline<algorithmFPType, estimatesAll>
{
    constexpr static uint32_t nResults              = lastResultId + 1;
    constexpr static uint32_t nBuffers              = 5;
    constexpr static bool isRowsInBlockInfoRequired = true;
    constexpr static uint32_t nPartialResults       = lastPartialResultId; // removed '+1' due to nObservations is mapped separately
    constexpr static uint32_t nFinalizeResults      = 5;
    // names of used kernels
    static const char * kSinglePassName;
    static const char * kProcessBlocksName;
    static const char * kMergeBlocksName;
    static const char * kFinalizeName;
    // kernels build options
    static const char * kBldOptFNameSuff;
    static const char * kBldOptScope;
    static const char * kCacheKey;

    int resIds[nResults];                 // required set of results' ids
    int resPartialIds[nPartialResults];   // required set of partial results' ids
    int resFinalizeIds[nFinalizeResults]; // set of results' ids which will be processed on finalize stage

    TaskInfoOnline()
        : resPartialIds { partialMinimum, partialMaximum, partialSum, partialSumSquares, partialSumSquaresCentered },
          resIds { minimum, maximum, sum, sumSquares, sumSquaresCentered, mean, secondOrderRawMoment, variance, standardDeviation, variation },
          resFinalizeIds { mean, secondOrderRawMoment, variance, standardDeviation, variation }
    {}
};

/* online kernel class */
template <typename algorithmFPType, low_order_moments::Method method>
class LowOrderMomentsOnlineKernelOneAPI : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable * dataTable, PartialResult * partialResult, const Parameter * parameter, bool isOnline);
    services::Status finalizeCompute(PartialResult * partialResult, Result * result, const Parameter * parameter);
};

/* online task class */
template <typename algorithmFPType, EstimatesToCompute scope>
class LowOrderMomentsOnlineTaskOneAPI : public TaskInfoOnline<algorithmFPType, scope>
{
public:
    LowOrderMomentsOnlineTaskOneAPI(services::internal::sycl::ExecutionContextIface & context, NumericTable * dataTable,
                                    PartialResult * partialResult, services::Status & status);
    LowOrderMomentsOnlineTaskOneAPI(const LowOrderMomentsOnlineTaskOneAPI &) = delete;
    LowOrderMomentsOnlineTaskOneAPI & operator=(const LowOrderMomentsOnlineTaskOneAPI &) = delete;
    virtual ~LowOrderMomentsOnlineTaskOneAPI();
    Status compute();

private:
    static constexpr size_t _uint32max = static_cast<size_t>(services::internal::MaxVal<uint32_t>::get());

    uint32_t nVectors;
    uint32_t nFeatures;

    const uint32_t maxWorkItemsPerGroup        = 256;
    const uint32_t maxWorkItemsPerGroupToMerge = 16;

    uint32_t nRowsBlocks;
    uint32_t nColsBlocks;
    uint32_t workItemsPerGroup;

    NumericTable * dataTable;
    BlockDescriptor<algorithmFPType> dataBD;

    NumericTablePtr nObservationsTable;
    BlockDescriptor<algorithmFPType> nObservationsBD;
    algorithmFPType * pNObservations;

    services::internal::sycl::UniversalBuffer bNVec; // contains info about num of vectors in block

    NumericTablePtr resultTable[TaskInfoOnline<algorithmFPType, scope>::nPartialResults];
    services::internal::sycl::UniversalBuffer bAuxBuffers[TaskInfoOnline<algorithmFPType, scope>::nBuffers];

    BlockDescriptor<algorithmFPType> resultBD[TaskInfoOnline<algorithmFPType, scope>::nPartialResults];
};

/* finalize task class */
template <typename algorithmFPType, EstimatesToCompute scope>
class LowOrderMomentsOnlineFinalizeTaskOneAPI : public TaskInfoOnline<algorithmFPType, scope>
{
public:
    LowOrderMomentsOnlineFinalizeTaskOneAPI(services::internal::sycl::ExecutionContextIface & context, PartialResult * partialResult, Result * result,
                                            services::Status & status);
    LowOrderMomentsOnlineFinalizeTaskOneAPI(const LowOrderMomentsOnlineFinalizeTaskOneAPI &) = delete;
    LowOrderMomentsOnlineFinalizeTaskOneAPI & operator=(const LowOrderMomentsOnlineFinalizeTaskOneAPI &) = delete;
    virtual ~LowOrderMomentsOnlineFinalizeTaskOneAPI();
    Status compute();

private:
    uint32_t nFeatures;
    constexpr static uint32_t nTotalResults =
        TaskInfoOnline<algorithmFPType, scope>::nPartialResults + TaskInfoOnline<algorithmFPType, scope>::nFinalizeResults;

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
