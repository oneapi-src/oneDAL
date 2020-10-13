/* file: gbt_regression_train_kernel_oneapi.h */
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
//  Declaration of structure containing kernels for gradient boosted trees
//  training for GPU.
//--
*/

#ifndef __GBT_REGRESSION_TRAIN_KERNEL_ONEAPI_H__
#define __GBT_REGRESSION_TRAIN_KERNEL_ONEAPI_H__

#include "services/internal/sycl/types.h"
#include "services/internal/sycl/execution_context.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_training_types.h"
#include "src/algorithms/engines/engine_batch_impl.h"
#include "src/algorithms/dtrees/gbt/oneapi/gbt_feature_type_helper_oneapi.h"

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
    services::Status compute(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, gbt::regression::Model & m, Result & res,
                             const Parameter & par, engines::internal::BatchBaseImpl & engine);

private:
    services::Status scan(const services::internal::Buffer<algorithmFPType> & values, services::internal::sycl::UniversalBuffer & partialSums,
                          uint32_t nRows, uint32_t localSize, uint32_t nLocalSums);

    services::Status reduce(services::internal::sycl::UniversalBuffer & partialSums, services::internal::sycl::UniversalBuffer & totalSum,
                            uint32_t localSize, uint32_t nSubgroupSums);

    services::Status getInitialResponse(NumericTable & y, algorithmFPType * response);

    services::Status computeOptCoeffs(NumericTable & y, services::internal::sycl::UniversalBuffer & response,
                                      services::internal::sycl::UniversalBuffer & optCoeffs);

    services::Status initializeTreeOrder(uint32_t nRows, services::internal::sycl::UniversalBuffer & treeOrder);

    services::Status computePartialHistograms(const services::internal::sycl::UniversalBuffer & data,
                                              services::internal::sycl::UniversalBuffer & treeOrder,
                                              services::internal::sycl::UniversalBuffer & optCoeffs,
                                              services::internal::sycl::UniversalBuffer & partialHistograms, uint32_t iStart, uint32_t nRows,
                                              services::internal::sycl::UniversalBuffer & binOffsets, uint32_t nTotalBins, uint32_t nFeatures,
                                              uint32_t localSize, uint32_t nPartialHistograms);

    services::Status reducePartialHistograms(services::internal::sycl::UniversalBuffer & partialHistograms,
                                             services::internal::sycl::UniversalBuffer & histograms, uint32_t nTotalBins, uint32_t reduceLocalSize,
                                             uint32_t nPartialHistograms);

    services::Status computeHistogram(const services::internal::sycl::UniversalBuffer & data, services::internal::sycl::UniversalBuffer & treeOrder,
                                      services::internal::sycl::UniversalBuffer & optCoeffs,
                                      services::internal::sycl::UniversalBuffer & partialHistograms,
                                      services::internal::sycl::UniversalBuffer & histograms, uint32_t iStart, uint32_t nRows,
                                      services::internal::sycl::UniversalBuffer & binOffsets, uint32_t nTotalBins, uint32_t nFeatures);

    services::Status computeHistogramDiff(services::internal::sycl::UniversalBuffer & histogramSrc,
                                          services::internal::sycl::UniversalBuffer & histogramTotal,
                                          services::internal::sycl::UniversalBuffer & histogramDst, uint32_t nBins);

    services::Status computeTotalOptCoeffs(services::internal::sycl::UniversalBuffer & histograms,
                                           services::internal::sycl::UniversalBuffer & totalOptCoeffs,
                                           services::internal::sycl::UniversalBuffer & binOffsets, uint32_t nTotalBins, uint32_t nFeatures,
                                           uint32_t localSize);

    services::Status computeBestSplitForFeatures(services::internal::sycl::UniversalBuffer & histograms,
                                                 services::internal::sycl::UniversalBuffer & totalOptCoeffs,
                                                 services::internal::sycl::UniversalBuffer & splitInfo,
                                                 services::internal::sycl::UniversalBuffer & splitValue,
                                                 services::internal::sycl::UniversalBuffer & binOffsets, uint32_t nTotalBins, uint32_t nFeatures,
                                                 algorithmFPType lambda, uint32_t localSize);

    services::Status computeBestSplit(services::internal::sycl::UniversalBuffer & histograms, services::internal::sycl::UniversalBuffer & binOffsets,
                                      uint32_t nTotalBins, uint32_t nFeatures, algorithmFPType lambda,
                                      gbt::internal::BestSplitOneAPI<algorithmFPType> & bestSplit, algorithmFPType * gTotal = nullptr,
                                      algorithmFPType * hTotal = nullptr);

    services::Status partitionScan(const services::internal::sycl::UniversalBuffer & data, services::internal::sycl::UniversalBuffer & treeOrder,
                                   services::internal::sycl::UniversalBuffer & partialSums, int splitValue, uint32_t iStart, uint32_t nRows,
                                   uint32_t localSize, uint32_t nLocalSums);

    services::Status partitionSumScan(services::internal::sycl::UniversalBuffer & partialSums,
                                      services::internal::sycl::UniversalBuffer & partialPrefixSums,
                                      services::internal::sycl::UniversalBuffer & totalSum, uint32_t localSize, uint32_t nSubgroupSums);

    services::Status partitionReorder(const services::internal::sycl::UniversalBuffer & data, services::internal::sycl::UniversalBuffer & treeOrder,
                                      services::internal::sycl::UniversalBuffer & treeOrderBuf,
                                      services::internal::sycl::UniversalBuffer & partialPrefixSums, int spliteValue, uint32_t iStart, uint32_t nRows,
                                      uint32_t localSize, uint32_t nLocalSums);

    services::Status partitionCopy(services::internal::sycl::UniversalBuffer & treeOrderBuf, services::internal::sycl::UniversalBuffer & treeOrder,
                                   uint32_t iStart, uint32_t nRows);

    services::Status doPartition(const services::internal::sycl::UniversalBuffer & data, services::internal::sycl::UniversalBuffer & treeOrder,
                                 services::internal::sycl::UniversalBuffer & treeOrderBuf, int splitValue, uint32_t iStart, uint32_t nRows,
                                 uint32_t & nLeft, uint32_t & nRight);

    services::Status updateResponse(services::internal::sycl::UniversalBuffer & treeOrder, services::internal::sycl::UniversalBuffer & response,
                                    uint32_t iStart, uint32_t nRows, algorithmFPType inc);

    services::internal::sycl::KernelPtr kernelScan;
    services::internal::sycl::KernelPtr kernelReduce;
    services::internal::sycl::KernelPtr kernelInitializeTreeOrder;
    services::internal::sycl::KernelPtr kernelComputePartialHistograms;
    services::internal::sycl::KernelPtr kernelReducePartialHistograms;
    services::internal::sycl::KernelPtr kernelComputeHistogramDiff;
    services::internal::sycl::KernelPtr kernelComputeOptCoeffs;
    services::internal::sycl::KernelPtr kernelComputeTotalOptCoeffs;
    services::internal::sycl::KernelPtr kernelComputeBestSplitForFeatures;
    services::internal::sycl::KernelPtr kernelPartitionScan;
    services::internal::sycl::KernelPtr kernelPartitionSumScan;
    services::internal::sycl::KernelPtr kernelPartitionReorder;
    services::internal::sycl::KernelPtr kernelPartitionCopy;
    services::internal::sycl::KernelPtr kernelUpdateResponse;

    const uint32_t _maxWorkItemsPerGroup = 128;   // should be a power of two for interal needs
    const uint32_t _maxLocalBuffer       = 30000; // should be less than a half of local memory (two buffers)
    const uint32_t _preferableSubGroup   = 16;    // preferable maximal sub-group size
    const uint32_t _maxLocalSize         = 128;
    const uint32_t _maxLocalSums         = 256;
    const uint32_t _maxLocalHistograms   = 256;
    const uint32_t _preferableGroupSize  = 256;
};

} // namespace internal
} // namespace training
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
