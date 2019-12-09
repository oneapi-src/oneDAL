/* file: low_order_moments_kernel_batch_oneapi.h */
/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "low_order_moments_types.h"

using namespace daal::services;
using namespace daal::data_management;
using namespace daal::oneapi::internal;

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
    constexpr static unsigned int nResults          = 2;
    constexpr static unsigned int nBuffers          = 2;
    constexpr static bool isRowsInBlockInfoRequired = false;
    using resultsSetType                            = int[nResults];
    using resultTableType                           = NumericTablePtr[nResults];
    using buffersType                               = UniversalBuffer[nBuffers];
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
    constexpr static unsigned int nResults          = 2;
    constexpr static unsigned int nBuffers          = 2;
    constexpr static bool isRowsInBlockInfoRequired = true;
    using resultsSetType                            = int[nResults];
    using resultTableType                           = NumericTablePtr[nResults];
    using buffersType                               = UniversalBuffer[nBuffers];
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
    constexpr static unsigned int nResults          = lastResultId + 1;
    constexpr static unsigned int nBuffers          = 5;
    constexpr static bool isRowsInBlockInfoRequired = true;
    using resultsSetType                            = int[nResults];
    using resultTableType                           = NumericTablePtr[nResults];
    using buffersType                               = UniversalBuffer[nBuffers];
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
    services::Status compute(NumericTable * dataTable, Result * result, const Parameter * parameter);
};

template <typename algorithmFPType, EstimatesToCompute scope>
class LowOrderMomentsBatchTaskOneAPI : public TaskInfoBatch<algorithmFPType, scope>
{
public:
    LowOrderMomentsBatchTaskOneAPI(ExecutionContextIface & context, NumericTable * dataTable, Result * result, services::Status * status);
    LowOrderMomentsBatchTaskOneAPI(const LowOrderMomentsBatchTaskOneAPI &) = delete;
    LowOrderMomentsBatchTaskOneAPI & operator=(const LowOrderMomentsBatchTaskOneAPI &) = delete;
    virtual ~LowOrderMomentsBatchTaskOneAPI();
    Status compute();

private:
    unsigned int nVectors;
    unsigned int nFeatures;

    const unsigned int maxWorkItemsPerGroup = 256;
    const unsigned int maxWorkItemsPerGroupToMerge = 16;

    unsigned int nRowsBlocks;
    unsigned int nColsBlocks;
    unsigned int workItemsPerGroup;

    NumericTable * dataTable;
    BlockDescriptor<algorithmFPType> dataBD;

    UniversalBuffer bNVec; // contains info about num of vectors in block

    NumericTablePtr resultTable[TaskInfoBatch<algorithmFPType, scope>::nResults];
    UniversalBuffer bAuxBuffers[TaskInfoBatch<algorithmFPType, scope>::nBuffers];

    BlockDescriptor<algorithmFPType> resultBD[TaskInfoBatch<algorithmFPType, scope>::nResults];
};

} // namespace internal
} // namespace oneapi
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
