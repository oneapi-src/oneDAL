/* file: df_regression_train_hist_kernel_oneapi.h */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
//  Declaration of structure containing kernels for decision forest
//  training for GPU for the hist method.
//--
*/

#ifndef __DF_REGRESSION_TRAIN_HIST_KERNEL_ONEAPI_H__
#define __DF_REGRESSION_TRAIN_HIST_KERNEL_ONEAPI_H__

#include "services/internal/sycl/types.h"
#include "services/internal/sycl/execution_context.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_model_impl.h"
#include "algorithms/decision_forest/decision_forest_regression_training_types.h"
#include "algorithms/decision_forest/decision_forest_regression_model.h"
#include "src/algorithms/dtrees/forest/oneapi/df_feature_type_helper_oneapi.h"
#include "src/algorithms/dtrees/forest/oneapi/df_tree_level_build_helper_oneapi.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
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
    RegressionTrainBatchKernelOneAPI() {}
    services::Status compute(services::HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y,
                             decision_forest::regression::Model & m, Result & res, const Parameter & par)
    {
        return services::ErrorMethodNotImplemented;
    }
};

template <typename algorithmFPType>
class RegressionTrainBatchKernelOneAPI<algorithmFPType, hist> : public daal::algorithms::Kernel
{
public:
    RegressionTrainBatchKernelOneAPI() : _nRows(0), _nFeatures(0), _nSelectedRows(0), _nMaxBinsAmongFtrs(0), _totalBins(0) {};
    services::Status compute(services::HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y,
                             decision_forest::regression::Model & m, Result & res, const Parameter & par);

private:
    services::Status buildProgram(services::internal::sycl::ClKernelFactoryIface & factory, const char * programName, const char * programSrc,
                                  const char * buildOptions);

    size_t getPartHistRequiredMemSize(size_t nSelectedFeatures, size_t nMaxBinsAmongFtrs);

    services::Status computeBestSplit(const services::internal::sycl::UniversalBuffer & data, services::internal::sycl::UniversalBuffer & treeOrder,
                                      services::internal::sycl::UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
                                      const services::internal::Buffer<algorithmFPType> & response,
                                      services::internal::sycl::UniversalBuffer & nodeOffsets, services::internal::sycl::UniversalBuffer & binOffsets,
                                      services::internal::sycl::UniversalBuffer & splitInfo,
                                      services::internal::sycl::UniversalBuffer & nodeImpDecreaseList, bool updateImpDecreaseRequired,
                                      size_t nFeatures, size_t nNodes, size_t minObservationsInLeafNode, algorithmFPType impurityThreshold);

    services::Status computeBestSplitSinglePass(
        const services::internal::sycl::UniversalBuffer & data, services::internal::sycl::UniversalBuffer & treeOrder,
        services::internal::sycl::UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
        const services::internal::Buffer<algorithmFPType> & response, services::internal::sycl::UniversalBuffer & binOffsets,
        services::internal::sycl::UniversalBuffer & nodeList, services::internal::sycl::UniversalBuffer & nodeIndices, size_t nodeIndicesOffset,
        services::internal::sycl::UniversalBuffer & impList, services::internal::sycl::UniversalBuffer & nodeImpDecreaseList,
        bool updateImpDecreaseRequired, size_t nFeatures, size_t nNodes, size_t minObservationsInLeafNode, algorithmFPType impurityThreshold);

    services::Status computeBestSplitByHistogram(
        const services::internal::sycl::UniversalBuffer & nodeHistogramList, services::internal::sycl::UniversalBuffer & selectedFeatures,
        size_t nSelectedFeatures, services::internal::sycl::UniversalBuffer & nodeList, services::internal::sycl::UniversalBuffer & nodeIndices,
        size_t nodeIndicesOffset, services::internal::sycl::UniversalBuffer & binOffsets, services::internal::sycl::UniversalBuffer & splitInfo,
        services::internal::sycl::UniversalBuffer & nodeImpDecreaseList, bool updateImpDecreaseRequired, size_t nNodes, size_t nMaxBinsAmongFtrs,
        size_t minObservationsInLeafNode, algorithmFPType impurityThreshold);

    services::Status computePartialHistograms(const services::internal::sycl::UniversalBuffer & data,
                                              services::internal::sycl::UniversalBuffer & treeOrder,
                                              services::internal::sycl::UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
                                              const services::internal::Buffer<algorithmFPType> & response,
                                              services::internal::sycl::UniversalBuffer & nodeList,
                                              services::internal::sycl::UniversalBuffer & nodeIndices, size_t nodeIndicesOffset,
                                              services::internal::sycl::UniversalBuffer & binOffsets, size_t nMaxBinsAmongFtrs, size_t nFeatures,
                                              size_t nNodes, services::internal::sycl::UniversalBuffer & partialHistograms,
                                              size_t nPartialHistograms);

    services::Status reducePartialHistograms(services::internal::sycl::UniversalBuffer & partialHistograms,
                                             services::internal::sycl::UniversalBuffer & histograms, size_t nPartialHistograms, size_t nNodes,
                                             size_t nSelectedFeatures, size_t nMaxBinsAmongFtrs, size_t reduceLocalSize);

    services::Status computeResults(const dtrees::internal::Tree & t, const algorithmFPType * x, const algorithmFPType * y, const size_t nRows,
                                    const size_t nFeatures, const services::internal::sycl::UniversalBuffer & oobIndices,
                                    const services::internal::sycl::UniversalBuffer & oobRowsNumList,
                                    services::internal::sycl::UniversalBuffer & oobBuf, algorithmFPType * varImp, algorithmFPType * varImpVariance,
                                    size_t nBuiltTrees, const engines::EnginePtr & engine, size_t nTreesInBlock, size_t treeIndex,
                                    const Parameter & par);

    algorithmFPType computeOOBError(const dtrees::internal::Tree & t, const algorithmFPType * x, const algorithmFPType * y, const size_t nRows,
                                    const size_t nFeatures, const services::internal::sycl::UniversalBuffer & indices, size_t indicesOffset, size_t n,
                                    services::internal::sycl::UniversalBuffer oobBuf, services::Status & status);

    algorithmFPType computeOOBErrorPerm(const dtrees::internal::Tree & t, const algorithmFPType * x, const algorithmFPType * y, const size_t nRows,
                                        const size_t nFeatures, const services::internal::sycl::UniversalBuffer & indices, size_t indicesOffset,
                                        const int * indicesPerm, const size_t testFtrInd, size_t n, services::Status & status);

    services::Status finalizeOOBError(const algorithmFPType * y, const services::internal::sycl::UniversalBuffer & oobBuf, const size_t nRows,
                                      algorithmFPType * res, algorithmFPType * resPerObs, algorithmFPType * resR2, algorithmFPType * resPrediction);

    services::Status finalizeVarImp(const Parameter & par, algorithmFPType * varImp, algorithmFPType * varImpVariance, size_t nFeatures);

    services::internal::sycl::KernelPtr kernelComputePartialHistograms;
    services::internal::sycl::KernelPtr kernelReducePartialHistograms;
    services::internal::sycl::KernelPtr kernelComputeBestSplitByHistogram;
    services::internal::sycl::KernelPtr kernelComputeBestSplitSinglePass;

    decision_forest::internal::TreeLevelBuildHelperOneAPI<algorithmFPType> _treeLevelBuildHelper;

    const size_t _maxWorkItemsPerGroup    = 256; // should be a power of two for interal needs
    const size_t _preferableSubGroup      = 16;  // preferable maximal sub-group size
    const size_t _maxLocalSize            = 128;
    const size_t _maxLocalSums            = 256;
    const size_t _maxLocalHistograms      = 256;
    const size_t _preferableGroupSize     = 256;
    const size_t _minRowsBlock            = 256;
    const size_t _maxBins                 = 256;
    const size_t _reduceLocalSizePartHist = 64;

    const size_t _minPreferableLocalSizeForPartHistKernel = 32;

    const double _globalMemFractionForTreeBlock  = 0.6;        // part of free global mem which can be used for processing block of tree
    const double _globalMemFractionForPartHist   = 0.2;        // part of free global mem which can be used for partial histograms
    const size_t _maxMemAllocSizeForAlgo         = 1073741824; // 1 Gb it showed better efficiency than using just platform info.maxMemAllocSize
    const size_t _minRowsBlocksForMaxPartHistNum = 16384;
    const size_t _minRowsBlocksForOneHist        = 128;

    const size_t _nOOBProps      = 2; // number of props for each OOB row to compute prediction (i.e. mean and num of predictions)
    const size_t _nHistProps     = 3; // number of properties in bins histogram (i.e. n, mean and var)
    const size_t _nNodesGroups   = 3; // all nodes are split on groups (big, medium, small)
    const size_t _nodeGroupProps = 2; // each nodes Group contains props: numOfNodes, maxNumOfBlocks

    static constexpr size_t _int32max = static_cast<size_t>(services::internal::MaxVal<int32_t>::get());

    size_t _nRows;
    size_t _nFeatures;
    size_t _nSelectedRows;
    size_t _nMaxBinsAmongFtrs;
    size_t _totalBins;
    size_t _preferableLocalSizeForPartHistKernel; // local size for histogram collecting kernel, depends on num of selected features
    size_t _maxPartHistCumulativeSize;            // is calculated at the beggining of compute using _globalMemFractionForPartHist
};

} // namespace internal
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
