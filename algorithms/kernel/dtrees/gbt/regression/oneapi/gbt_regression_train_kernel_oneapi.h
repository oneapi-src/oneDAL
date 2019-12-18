/* file: gbt_regression_train_kernel_oneapi.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Declaration of structure containing kernels for gradient boosted trees
//  training for GPU.
//--
*/

#ifndef __GBT_REGRESSION_TRAIN_KERNEL_ONEAPI_H__
#define __GBT_REGRESSION_TRAIN_KERNEL_ONEAPI_H__

#include "oneapi/internal/types.h"
#include "oneapi/internal/execution_context.h"
#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "gbt_regression_training_types.h"
#include "engine_batch_impl.h"
#include "oneapi/gbt_feature_type_helper_oneapi.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace training
{
namespace internal
{

template <typename algorithmFPType, Method method>
class RegressionTrainBatchKernelOneAPI : public daal::algorithms::Kernel
{
public:
    services::Status compute(HostAppIface* pHostApp, const NumericTable *x, const NumericTable *y,
        gbt::regression::Model& m, Result& res, const Parameter& par,
        engines::internal::BatchBaseImpl& engine);

private:
    services::Status scan(const services::Buffer<algorithmFPType>& values,
                          oneapi::internal::UniversalBuffer& partialSums,
                          int nRows,
                          int localSize,
                          int nLocalSums);

    services::Status reduce(oneapi::internal::UniversalBuffer& partialSums,
                            oneapi::internal::UniversalBuffer& totalSum,
                            int localSize,
                            int nSubgroupSums);

    services::Status getInitialResponse(NumericTable& y, algorithmFPType *response);

    services::Status computeOptCoeffs(NumericTable& y,
                                      oneapi::internal::UniversalBuffer& response,
                                      oneapi::internal::UniversalBuffer& optCoeffs);

    services::Status initializeTreeOrder(size_t nRows,
                                         oneapi::internal::UniversalBuffer& treeOrder);

    services::Status computePartialHistograms(const oneapi::internal::UniversalBuffer& data,
                                              oneapi::internal::UniversalBuffer& treeOrder,
                                              oneapi::internal::UniversalBuffer& optCoeffs,
                                              oneapi::internal::UniversalBuffer& partialHistograms,
                                              size_t iStart,
                                              size_t nRows,
                                              oneapi::internal::UniversalBuffer& binOffsets,
                                              size_t nTotalBins,
                                              size_t nFeatures,
                                              size_t localSize,
                                              size_t nPartialHistograms);

    services::Status reducePartialHistograms(oneapi::internal::UniversalBuffer& partialHistograms,
                                             oneapi::internal::UniversalBuffer& histograms,
                                             size_t nTotalBins,
                                             size_t reduceLocalSize,
                                             size_t nPartialHistograms);

    services::Status computeHistogram(const oneapi::internal::UniversalBuffer& data,
                                      oneapi::internal::UniversalBuffer& treeOrder,
                                      oneapi::internal::UniversalBuffer& optCoeffs,
                                      oneapi::internal::UniversalBuffer& partialHistograms,
                                      oneapi::internal::UniversalBuffer& histograms,
                                      size_t iStart,
                                      size_t nRows,
                                      oneapi::internal::UniversalBuffer& binOffsets,
                                      size_t nTotalBins,
                                      size_t nFeatures);

    services::Status computeHistogramDiff(oneapi::internal::UniversalBuffer& histogramSrc,
                                          oneapi::internal::UniversalBuffer& histogramTotal,
                                          oneapi::internal::UniversalBuffer& histogramDst,
                                          size_t nBins);

    services::Status computeTotalOptCoeffs(oneapi::internal::UniversalBuffer& histograms,
                                           oneapi::internal::UniversalBuffer& totalOptCoeffs,
                                           oneapi::internal::UniversalBuffer& binOffsets,
                                           size_t nTotalBins,
                                           size_t nFeatures,
                                           size_t localSize);

    services::Status computeBestSplitForFeatures(oneapi::internal::UniversalBuffer& histograms,
                                                 oneapi::internal::UniversalBuffer& totalOptCoeffs,
                                                 oneapi::internal::UniversalBuffer& splitInfo,
                                                 oneapi::internal::UniversalBuffer& splitValue,
                                                 oneapi::internal::UniversalBuffer& binOffsets,
                                                 size_t nTotalBins,
                                                 size_t nFeatures,
                                                 algorithmFPType lambda,
                                                 size_t localSize);

    services::Status computeBestSplit(oneapi::internal::UniversalBuffer& histograms,
                                      oneapi::internal::UniversalBuffer& binOffsets,
                                      size_t nTotalBins,
                                      size_t nFeatures,
                                      algorithmFPType lambda,
                                      gbt::internal::BestSplitOneAPI<algorithmFPType>& bestSplit,
                                      algorithmFPType* gTotal = nullptr,
                                      algorithmFPType* hTotal = nullptr);

    services::Status partitionScan(const oneapi::internal::UniversalBuffer& data,
                                   oneapi::internal::UniversalBuffer& treeOrder,
                                   oneapi::internal::UniversalBuffer& partialSums,
                                   int splitValue,
                                   size_t iStart,
                                   size_t nRows,
                                   size_t localSize,
                                   size_t nLocalSums);

    services::Status partitionSumScan(oneapi::internal::UniversalBuffer& partialSums,
                                      oneapi::internal::UniversalBuffer& partialPrefixSums,
                                      oneapi::internal::UniversalBuffer& totalSum,
                                      size_t localSize,
                                      size_t nSubgroupSums);

    services::Status partitionReorder(const oneapi::internal::UniversalBuffer& data,
                                      oneapi::internal::UniversalBuffer& treeOrder,
                                      oneapi::internal::UniversalBuffer& treeOrderBuf,
                                      oneapi::internal::UniversalBuffer& partialPrefixSums,
                                      int spliteValue,
                                      size_t iStart,
                                      size_t nRows,
                                      size_t localSize,
                                      size_t nLocalSums);

    services::Status partitionCopy(oneapi::internal::UniversalBuffer& treeOrderBuf,
                                   oneapi::internal::UniversalBuffer& treeOrder,
                                   size_t iStart,
                                   size_t nRows);

    services::Status doPartition(const oneapi::internal::UniversalBuffer& data,
                                 oneapi::internal::UniversalBuffer& treeOrder,
                                 oneapi::internal::UniversalBuffer& treeOrderBuf,
                                 int splitValue,
                                 size_t iStart,
                                 size_t nRows,
                                 size_t& nLeft,
                                 size_t& nRight);

    services::Status updateResponse(oneapi::internal::UniversalBuffer& treeOrder,
                                    oneapi::internal::UniversalBuffer& response,
                                    size_t iStart,
                                    size_t nRows,
                                    algorithmFPType inc);

    oneapi::internal::KernelPtr kernelScan;
    oneapi::internal::KernelPtr kernelReduce;
    oneapi::internal::KernelPtr kernelInitializeTreeOrder;
    oneapi::internal::KernelPtr kernelComputePartialHistograms;
    oneapi::internal::KernelPtr kernelReducePartialHistograms;
    oneapi::internal::KernelPtr kernelComputeHistogramDiff;
    oneapi::internal::KernelPtr kernelComputeOptCoeffs;
    oneapi::internal::KernelPtr kernelComputeTotalOptCoeffs;
    oneapi::internal::KernelPtr kernelComputeBestSplitForFeatures;
    oneapi::internal::KernelPtr kernelPartitionScan;
    oneapi::internal::KernelPtr kernelPartitionSumScan;
    oneapi::internal::KernelPtr kernelPartitionReorder;
    oneapi::internal::KernelPtr kernelPartitionCopy;
    oneapi::internal::KernelPtr kernelUpdateResponse;

    const uint32_t _maxWorkItemsPerGroup = 128; // should be a power of two for interal needs
    const uint32_t _maxLocalBuffer = 30000; // should be less than a half of local memory (two buffers)
    const uint32_t _preferableSubGroup = 16; // preferable maximal sub-group size
    const uint32_t _maxLocalSize = 128;
    const uint32_t _maxLocalSums = 256;
    const uint32_t _maxLocalHistograms = 256;
    const uint32_t _preferableGroupSize = 256;
};

} // namespace internal
}
}
}
}
} // namespace daal


#endif
