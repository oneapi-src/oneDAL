/* file: low_order_moments_kernel_batch_oneapi.h */
/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef __LOW_ORDER_MOMENTS_KERNEL_BATCH_ONEAPI_H__
#define __LOW_ORDER_MOMENTS_KERNEL_BATCH_ONEAPI_H__

#include "src/algorithms/kernel.h"
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
struct TaskInfoBatch;

template <typename algorithmFPType>
struct TaskInfoBatch<algorithmFPType, estimatesMinMax>
{
    constexpr static uint32_t nResults              = 2;
    constexpr static uint32_t nBuffers              = 2;
    constexpr static bool isRowsInBlockInfoRequired = false;
    // names of used kernels
    static const char * kSinglePassName;
    static const char * kProcessBlocksName;
    static const char * kMergeBlocksName;
    // kernels build options
    static const char * kBldOptFNameSuff;
    static const char * kBldOptScope;
    static const char * kCacheKey;

    int resIds[nResults]; // required set of results' ids
    TaskInfoBatch() : resIds { minimum, maximum } {}
};

template <typename algorithmFPType>
struct TaskInfoBatch<algorithmFPType, estimatesMeanVariance>
{
    constexpr static uint32_t nResults              = 2;
    constexpr static uint32_t nBuffers              = 2;
    constexpr static bool isRowsInBlockInfoRequired = true;
    // names of used kernels
    static const char * kSinglePassName;
    static const char * kProcessBlocksName;
    static const char * kMergeBlocksName;
    // kernels build options
    static const char * kBldOptFNameSuff;
    static const char * kBldOptScope;
    static const char * kCacheKey;

    int resIds[nResults]; // required set of results' ids
    TaskInfoBatch() : resIds { mean, variance } {}
};

template <typename algorithmFPType>
struct TaskInfoBatch<algorithmFPType, estimatesAll>
{
    constexpr static uint32_t nResults              = lastResultId + 1;
    constexpr static uint32_t nBuffers              = 5;
    constexpr static bool isRowsInBlockInfoRequired = true;
    // names of used kernels
    static const char * kSinglePassName;
    static const char * kProcessBlocksName;
    static const char * kMergeBlocksName;
    // kernels build options
    static const char * kBldOptFNameSuff;
    static const char * kBldOptScope;
    static const char * kCacheKey;

    int resIds[nResults]; // required set of results' ids
    TaskInfoBatch()
        : resIds { minimum, maximum, sum, sumSquares, sumSquaresCentered, mean, secondOrderRawMoment, variance, standardDeviation, variation }
    {}
};

template <typename algorithmFPType, low_order_moments::Method method>
class LowOrderMomentsBatchKernelOneAPI : public daal::algorithms::Kernel
{
public:
    services::Status compute(data_management::NumericTable * dataTable, Result * result, const Parameter * parameter);
};

template <typename algorithmFPType, EstimatesToCompute scope>
class LowOrderMomentsBatchTaskOneAPI : public TaskInfoBatch<algorithmFPType, scope>
{
public:
    LowOrderMomentsBatchTaskOneAPI(services::internal::sycl::ExecutionContextIface & context, data_management::NumericTable * dataTable,
                                   Result * result, services::Status & status);
    LowOrderMomentsBatchTaskOneAPI(const LowOrderMomentsBatchTaskOneAPI &) = delete;
    LowOrderMomentsBatchTaskOneAPI & operator=(const LowOrderMomentsBatchTaskOneAPI &) = delete;
    virtual ~LowOrderMomentsBatchTaskOneAPI();
    services::Status compute();

private:
    static constexpr size_t _uint32max = static_cast<size_t>(services::internal::MaxVal<uint32_t>::get());

    uint32_t nVectors;
    uint32_t nFeatures;

    const uint32_t maxWorkItemsPerGroup        = 256;
    const uint32_t maxWorkItemsPerGroupToMerge = 16;

    uint32_t nRowsBlocks;
    uint32_t nColsBlocks;
    uint32_t workItemsPerGroup;

    data_management::NumericTable * dataTable;
    data_management::BlockDescriptor<algorithmFPType> dataBD;

    services::internal::sycl::UniversalBuffer bNVec; // contains info about num of vectors in block

    data_management::NumericTablePtr resultTable[TaskInfoBatch<algorithmFPType, scope>::nResults];
    services::internal::sycl::UniversalBuffer bAuxBuffers[TaskInfoBatch<algorithmFPType, scope>::nBuffers];

    data_management::BlockDescriptor<algorithmFPType> resultBD[TaskInfoBatch<algorithmFPType, scope>::nResults];
};

} // namespace internal
} // namespace oneapi
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
